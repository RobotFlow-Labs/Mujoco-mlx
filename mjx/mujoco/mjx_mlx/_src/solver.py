# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constraint solvers (MLX port)."""

import mlx.core as mx
import mujoco
from mujoco.mjx_mlx._src import math
from mujoco.mjx_mlx._src import smooth
from mujoco.mjx_mlx._src import support
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.dataclasses import PyTreeNode
from mujoco.mjx_mlx._src.dataclasses import tree_map as _tree_map
from mujoco.mjx_mlx._src.types import ConeType
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import DataMLX
from mujoco.mjx_mlx._src.types import DisableBit
from mujoco.mjx_mlx._src.types import Model
from mujoco.mjx_mlx._src.types import ModelMLX
from mujoco.mjx_mlx._src.types import OptionMLX
from mujoco.mjx_mlx._src.types import SolverType
# pylint: enable=g-importing-member

import numpy as np


class Context(PyTreeNode):
  """Data updated during each solver iteration.

  Attributes:
    qacc: acceleration (from Data)                    (nv,)
    qfrc_constraint: constraint force (from Data)     (nv,)
    Jaref: Jac*qacc - aref                            (nefc,)
    efc_force: constraint force in constraint space   (nefc,)
    Ma: M*qacc                                        (nv,)
    grad: gradient of master cost                     (nv,)
    Mgrad: M / grad                                   (nv,)
    search: linesearch vector                         (nv,)
    gauss: gauss Cost
    cost: constraint + Gauss cost
    prev_cost: cost from previous iter
    solver_niter: number of solver iterations
    active: active (quadratic) constraints            (nefc,)
    fri: friction of regularized cone                 (num(con.dim > 1), 6)
    dm: regularized constraint mass                   (num(con.dim > 1))
    u: friction cone (normal and tangents)            (num(con.dim > 1), 6)
    h: cone hessian                                   (num(con.dim > 1), 6, 6)
  """

  qacc: mx.array
  qfrc_constraint: mx.array
  Jaref: mx.array  # pylint: disable=invalid-name
  efc_force: mx.array
  Ma: mx.array  # pylint: disable=invalid-name
  grad: mx.array
  Mgrad: mx.array  # pylint: disable=invalid-name
  search: mx.array
  gauss: mx.array
  cost: mx.array
  prev_cost: mx.array
  solver_niter: mx.array
  active: mx.array
  fri: mx.array
  dm: mx.array
  u: mx.array
  h: mx.array

  @classmethod
  def create(cls, m: Model, d: Data, grad: bool = True) -> 'Context':
    if not isinstance(d._impl, DataMLX) or not isinstance(
        m.opt._impl, OptionMLX
    ):
      pass  # MLX port: backend check removed

    jaref = (d._impl or d).efc_J @ d.qacc - (d._impl or d).efc_aref
    ma = support.mul_m(m, d, d.qacc)
    nv_0 = mx.zeros(m.nv)
    fri = mx.array(0.0)
    if m.opt.cone == ConeType.ELLIPTIC:
      mask = (d._impl or d).contact.dim > 1
      mask_idx = np.nonzero(mask)[0]
      if mask_idx.size > 0:
        friction = (d._impl or d).contact.friction[mask_idx]
        dim = (d._impl or d).contact.dim[mask_idx]
        mu = friction[:, 0] / mx.sqrt(m.opt.impratio)
        fri = mx.concatenate([mu[:, None], friction], axis=1)
        for condim in (3, 4, 6):
          dim_mask = dim == condim
          if dim_mask.any():
            # Zero out columns >= condim
            n_rows, n_cols = fri.shape
            col_mask = mx.arange(n_cols) >= condim
            row_mask = mx.array(dim_mask.astype(np.float32))
            # Combine: zero where both row matches and col >= condim
            zero_mask = (row_mask[:, None] > 0) & col_mask[None, :]
            fri = mx.where(zero_mask, 0.0, fri)

    ctx = Context(
        qacc=d.qacc,
        qfrc_constraint=d.qfrc_constraint,
        Jaref=jaref,
        efc_force=(d._impl or d).efc_force,
        Ma=ma,
        grad=nv_0,
        Mgrad=nv_0,
        search=nv_0,
        gauss=mx.array(0.0),
        cost=mx.array(float('inf')),
        prev_cost=mx.array(0.0),
        solver_niter=mx.array(0),
        active=mx.array(0.0),
        fri=fri,
        dm=mx.array(0.0),
        u=mx.array(0.0),
        h=mx.array(0.0),
    )
    ctx = _update_constraint(m, d, ctx)
    if grad:
      ctx = _update_gradient(m, d, ctx)
      ctx = ctx.replace(search=-ctx.Mgrad)  # start with preconditioned gradient

    return ctx


class _LSPoint(PyTreeNode):
  """Line search evaluation point.

  Attributes:
    alpha: step size that reduces f(x + alpha * p) given search direction p
    cost: line search cost
    deriv_0: first derivative of quadratic
    deriv_1: second derivative of quadratic
  """

  alpha: mx.array
  cost: mx.array
  deriv_0: mx.array
  deriv_1: mx.array

  @classmethod
  def create(
      cls,
      m: Model,
      d: Data,
      ctx: Context,
      alpha: mx.array,
      jv: mx.array,
      quad: mx.array,
      quad_gauss: mx.array,
      uu: mx.array,
      v0: mx.array,
      uv: mx.array,
      vv: mx.array,
  ) -> '_LSPoint':
    """Creates a linesearch point with first and second derivatives."""
    # MLX port: backend check removed

    cost, deriv_0, deriv_1 = mx.array(0.0), mx.array(0.0), mx.array(0.0)
    quad_total = quad_gauss
    x = ctx.Jaref + alpha * jv
    ne_nf = (d._impl or d).ne + (d._impl or d).nf
    active = (x < 0)
    # Set first ne+nf entries to True
    if ne_nf > 0:
      mask_ne_nf = mx.arange(x.shape[0]) < ne_nf
      active = mx.where(mask_ne_nf, True, active)

    dof_fl = (m._impl or m).dof_hasfrictionloss
    ten_fl = (m._impl or m).tendon_hasfrictionloss
    if (dof_fl.any() or ten_fl.any()) and not (
        m.opt.disableflags & DisableBit.FRICTIONLOSS
    ):
      f = (d._impl or d).efc_frictionloss
      r = 1.0 / ((d._impl or d).efc_D + ((d._impl or d).efc_D == 0.0) * mujoco.mjMINVAL)
      rf = r * f
      z = mx.zeros_like(f)

      linear_neg = (x <= -rf)
      linear_pos = (x >= rf)

      # Build qf for negative linear region
      qf_neg = mx.stack([
          f * (-0.5 * rf - ctx.Jaref), -f * jv, z
      ], axis=-1)

      # Build qf for positive linear region
      qf_pos = mx.stack([
          f * (-0.5 * rf + ctx.Jaref), f * jv, z
      ], axis=-1)

      # Apply masks
      use_neg = (linear_neg & (f > 0))[:, None]
      use_pos = (linear_pos & (f > 0))[:, None]
      quad = mx.where(use_neg, qf_neg, quad)
      quad = mx.where(use_pos, qf_pos, quad)

    if m.opt.cone == ConeType.ELLIPTIC:
      mu = ctx.fri[:, 0]
      u0 = ctx.u[:, 0]
      n = u0 + alpha * v0
      tsqr = uu + alpha * (2 * uv + alpha * vv)
      t = mx.sqrt(mx.maximum(tsqr, mx.array(0.0)))

      bottom_zone = ((tsqr <= 0) & (n < 0)) | ((tsqr > 0) & ((mu * n + t) <= 0))
      middle_zone = (tsqr > 0) & (n < (mu * t)) & ((mu * n + t) > 0)

      # quadratic cost for equality, friction, limits, frictionless contacts
      dim1_idx = np.nonzero((d._impl or d).contact.dim == 1)[0]
      dim1_addrs = (d._impl or d).contact.efc_address[dim1_idx]
      nefl = (d._impl or d).ne + (d._impl or d).nf + (d._impl or d).nl
      # Set active: nefl onwards to False, then re-enable dim1
      nefl_mask = mx.arange(active.shape[0]) >= nefl
      active = mx.where(nefl_mask, False, active)
      for addr in dim1_addrs:
        idx_mask = mx.arange(active.shape[0]) == int(addr)
        active = mx.where(idx_mask, (x < 0)[int(addr)], active)

      # quad_efld: multiply quad by active
      quad_efld = quad * active[:, None]
      quad_total = quad_total + mx.sum(quad_efld, axis=0)

      # elliptic bottom zone: quadratic cost
      efc_elliptic_idx = np.nonzero((d._impl or d).contact.dim > 1)[0]
      efc_elliptic_addrs = (d._impl or d).contact.efc_address[efc_elliptic_idx]
      if len(efc_elliptic_addrs) > 0:
        quad_c = quad[efc_elliptic_addrs] * bottom_zone[:, None]
        quad_total = quad_total + mx.sum(quad_c, axis=0)

      # elliptic middle zone
      t = t + (t == 0) * mujoco.mjMINVAL
      tsqr = tsqr + (tsqr == 0) * mujoco.mjMINVAL
      n1 = v0
      t1 = (uv + alpha * vv) / t
      t2 = vv / t - (uv + alpha * vv) * t1 / tsqr
      dm = ctx.dm * middle_zone
      nmt = n - mu * t
      cost = cost + 0.5 * mx.sum(dm * mx.square(nmt))
      deriv_0 = deriv_0 + mx.sum(dm * nmt * (n1 - mu * t1))
      deriv_1 = deriv_1 + mx.sum(dm * (mx.square(n1 - mu * t1) - nmt * mu * t2))
    elif m.opt.cone == ConeType.PYRAMIDAL:
      quad = quad * active[:, None]  # only active
      quad_total = quad_total + mx.sum(quad, axis=0)
    else:
      raise NotImplementedError(f'unsupported cone type: {m.opt.cone}')

    alpha_sq = alpha * alpha
    cost = cost + alpha_sq * quad_total[2] + alpha * quad_total[1] + quad_total[0]
    deriv_0 = deriv_0 + 2 * alpha * quad_total[2] + quad_total[1]
    deriv_1 = deriv_1 + 2 * quad_total[2] + (quad_total[2] == 0) * mujoco.mjMINVAL

    return _LSPoint(alpha=alpha, cost=cost, deriv_0=deriv_0, deriv_1=deriv_1)


class _LSContext(PyTreeNode):
  """Data updated during each line search iteration.

  Attributes:
    lo: low point bounding the line search interval
    hi: high point bounding the line search interval
    swap: True if low or hi was swapped in the line search iteration
    ls_iter: number of linesearch iterations
  """

  lo: _LSPoint
  hi: _LSPoint
  swap: mx.array
  ls_iter: mx.array


def _update_constraint(m: Model, d: Data, ctx: Context) -> Context:
  """Updates constraint force and resulting cost given last solver iteration.

  Corresponds to CGupdateConstraint in mujoco/src/engine/engine_solver.c

  Args:
    m: model defining constraints
    d: data which contains latest qacc and smooth terms
    ctx: current solver context

  Returns:
    context with new constraint force and costs
  """
  # MLX port: backend check removed

  # ne constraints are always active, nf are conditionally active, others are
  # non-negative constraints.
  ne_nf = (d._impl or d).ne + (d._impl or d).nf
  active = (ctx.Jaref < 0)
  if ne_nf > 0:
    mask_ne_nf = mx.arange(ctx.Jaref.shape[0]) < ne_nf
    active = mx.where(mask_ne_nf, True, active)

  floss_force = mx.zeros((d._impl or d).nefc)
  floss_cost = mx.array(0.0)
  dof_fl = (m._impl or m).dof_hasfrictionloss
  ten_fl = (m._impl or m).tendon_hasfrictionloss
  if (dof_fl.any() or ten_fl.any()) and not (
      m.opt.disableflags & DisableBit.FRICTIONLOSS
  ):
    f = (d._impl or d).efc_frictionloss
    r = 1.0 / ((d._impl or d).efc_D + ((d._impl or d).efc_D == 0.0) * mujoco.mjMINVAL)
    linear_neg = (ctx.Jaref <= -r * f) * (f > 0)
    linear_pos = (ctx.Jaref >= r * f) * (f > 0)
    active = active & ~linear_neg & ~linear_pos
    floss_force = linear_neg * f + linear_pos * -f
    floss_cost = linear_neg * (-0.5 * r * f * f - f * ctx.Jaref)
    floss_cost = floss_cost + linear_pos * (-0.5 * r * f * f + f * ctx.Jaref)
    floss_cost = mx.sum(floss_cost)

  if m.opt.cone == ConeType.PYRAMIDAL:
    efc_force = (d._impl or d).efc_D * -ctx.Jaref * active + floss_force
    cost = 0.5 * mx.sum((d._impl or d).efc_D * ctx.Jaref * ctx.Jaref * active)
    dm, u, h = mx.array(0.0), mx.array(0.0), mx.array(0.0)
  elif m.opt.cone == ConeType.ELLIPTIC:
    friction_idx = np.nonzero((d._impl or d).contact.dim > 1)[0]
    friction = (d._impl or d).contact.friction[friction_idx]
    efc_address = (d._impl or d).contact.efc_address[friction_idx]
    dim = (d._impl or d).contact.dim[friction_idx]

    # Gather Jaref slices for each elliptic contact
    jaref_padded = mx.concatenate([ctx.Jaref, mx.zeros(3)])
    u_list = []
    for i in range(len(efc_address)):
      addr = int(efc_address[i])
      u_list.append(jaref_padded[addr:addr + 6])
    if u_list:
      u = mx.stack(u_list) * ctx.fri
    else:
      u = mx.array(0.0)

    if isinstance(u, mx.array) and u.ndim >= 2:
      mu = ctx.fri[:, 0]
      n = u[:, 0]
      t = mx.sqrt(mx.sum(u[:, 1:] * u[:, 1:], axis=1))

      # bottom zone: quadratic
      bottom_zone = ((t <= 0) & (n < 0)) | ((t > 0) & ((mu * n + t) <= 0))
      # Set active for bottom-zone contacts
      for i in range(len(dim)):
        condim = int(dim[i])
        addr = int(efc_address[i])
        for j in range(condim):
          idx_mask = mx.arange(active.shape[0]) == (addr + j)
          active = mx.where(idx_mask, bottom_zone[i], active)

      efc_force = (d._impl or d).efc_D * -ctx.Jaref * active + floss_force
      cost = 0.5 * mx.sum((d._impl or d).efc_D * ctx.Jaref * ctx.Jaref * active)

      # middle zone: cone
      middle_zone = (t > 0) & (n < (mu * t)) & ((mu * n + t) > 0)
      dm = (d._impl or d).efc_D[efc_address] / mx.maximum(
          mu * mu * (1 + mu * mu), mx.array(mujoco.mjMINVAL)
      )
      nmt = n - mu * t
      cost = cost + 0.5 * mx.sum(dm * nmt * nmt * middle_zone)

      # tangent and friction for middle zone
      force = -dm * nmt * mu * middle_zone
      t_safe = t + (~middle_zone) * mujoco.mjMINVAL
      force_fri = (-force / t_safe)[:, None] * u[:, 1:] * friction

      # Scatter forces back to efc_force
      for i in range(len(efc_address)):
        addr = int(efc_address[i])
        idx_n = mx.arange(efc_force.shape[0]) == addr
        efc_force = mx.where(idx_n, efc_force + force[i], efc_force)
        condim = int(dim[i])
        for j in range(condim - 1):
          idx_j = mx.arange(efc_force.shape[0]) == (addr + 1 + j)
          efc_force = mx.where(idx_j, efc_force + force_fri[i, j], efc_force)

      # cone hessian
      h = mx.array(0.0)
      if m.opt.solver == SolverType.NEWTON:
        t_clamped = mx.maximum(t, mx.array(mujoco.mjMINVAL))
        ttt = mx.maximum(t_clamped * t_clamped * t_clamped, mx.array(mujoco.mjMINVAL))

        h_list = []
        for i in range(len(mu)):
          ui = u[i]
          # h = mu*N/T^3 * U*U'
          h_i = (mu[i] * n[i] / ttt[i]) * (ui[:, None] * ui[None, :])
          # add diagonal: (mu^2 - mu*N/T) * I
          h_i = h_i + (mu[i] * mu[i] - mu[i] * n[i] / t_clamped[i]) * mx.eye(6)
          # set first row/col: (1, -mu/T * U)
          h_0 = mx.concatenate([mx.array([1.0]), -mu[i] / t_clamped[i] * ui[1:]])
          # Build with first row replaced
          h_i = mx.concatenate([h_0[None, :], h_i[1:, :]], axis=0)
          # Build with first col replaced
          h_i = mx.concatenate([h_0[:, None], h_i[:, 1:]], axis=1)
          # pre and post multiply by diag(fri), scale by dm
          fri_i = ctx.fri[i]
          h_i = dm[i] * (fri_i[:, None] * h_i * fri_i[None, :])
          # mask by middle_zone
          h_i = h_i * middle_zone[i]
          h_list.append(h_i)

        h = mx.stack(h_list) if h_list else mx.array(0.0)
    else:
      efc_force = (d._impl or d).efc_D * -ctx.Jaref * active + floss_force
      cost = 0.5 * mx.sum((d._impl or d).efc_D * ctx.Jaref * ctx.Jaref * active)
      dm, u, h = mx.array(0.0), mx.array(0.0), mx.array(0.0)
  else:
    raise NotImplementedError(f'unsupported cone type: {m.opt.cone}')

  qfrc_constraint = (d._impl or d).efc_J.T @ efc_force
  gauss = 0.5 * mx.sum((ctx.Ma - d.qfrc_smooth) * (ctx.qacc - d.qacc_smooth))
  ctx = ctx.replace(
      qfrc_constraint=qfrc_constraint,
      gauss=gauss,
      cost=cost + gauss + floss_cost,
      prev_cost=ctx.cost,
      efc_force=efc_force,
      active=active,
      dm=dm,
      u=u,
      h=h,
  )

  return ctx


def _update_gradient(m: Model, d: Data, ctx: Context) -> Context:
  """Updates grad and M / grad given latest solver iteration.

  Corresponds to CGupdateGradient in mujoco/src/engine/engine_solver.c

  Args:
    m: model defining constraints
    d: data which contains latest smooth terms
    ctx: current solver context

  Returns:
    context with new grad and M / grad
  Raises:
    NotImplementedError: for unsupported solver type
  """
  # MLX port: backend check removed

  grad = ctx.Ma - d.qfrc_smooth - ctx.qfrc_constraint

  if m.opt.solver == SolverType.CG:
    mgrad = smooth.solve_m(m, d, grad)
  elif m.opt.solver == SolverType.NEWTON:
    if m.opt.cone == ConeType.ELLIPTIC:
      cm = mx.diag((d._impl or d).efc_D * ctx.active)
      efc_address = (d._impl or d).contact.efc_address[(d._impl or d).contact.dim > 1]
      dim = (d._impl or d).contact.dim[(d._impl or d).contact.dim > 1]
      # set efc of cone H along diagonal
      for i, (condim, addr) in enumerate(zip(dim, efc_address)):
        condim = int(condim)
        addr = int(addr)
        h_cone = ctx.h[i, :condim, :condim]
        # Add h_cone to cm[addr:addr+condim, addr:addr+condim]
        for r in range(condim):
          for c in range(condim):
            row_mask = mx.arange(cm.shape[0]) == (addr + r)
            col_mask = mx.arange(cm.shape[1]) == (addr + c)
            mask_2d = row_mask[:, None] & col_mask[None, :]
            cm = mx.where(mask_2d, cm + h_cone[r, c], cm)
      h = (d._impl or d).efc_J.T @ cm @ (d._impl or d).efc_J
    else:
      h = ((d._impl or d).efc_J.T * ((d._impl or d).efc_D * ctx.active)) @ (d._impl or d).efc_J
    h = support.full_m(m, d) + h
    # Symmetrize to reduce the chance of numerical issues in cholesky
    h_sym = (h + h.T) * 0.5
    # MLX Cholesky solve: L = cholesky(h_sym), solve L L^T x = grad
    L = mx.linalg.cholesky(h_sym)
    # Forward substitution: L y = grad
    y = mx.linalg.solve_triangular(L, grad[:, None], upper=False)
    # Backward substitution: L^T x = y
    mgrad = mx.linalg.solve_triangular(L.T, y, upper=True).squeeze(-1)
  else:
    raise NotImplementedError(f'unsupported solver type: {m.opt.solver}')

  ctx = ctx.replace(grad=grad, Mgrad=mgrad)

  return ctx


def _rescale(m: Model, value: mx.array) -> mx.array:
  return value / (m.stat.meaninertia * max(1, m.nv))


def _linesearch(m: Model, d: Data, ctx: Context) -> Context:
  """Performs a zoom linesearch to find optimal search step size.

  Args:
    m: model defining search options and other needed terms
    d: data with inertia matrix and other needed terms
    ctx: current solver context

  Returns:
    updated context with new qacc, Ma, Jaref
  """
  if (
      False  # MLX port: single backend
      or not isinstance(m.opt._impl, OptionMLX)
  ):
    pass  # MLX port: backend check removed

  smag = math.norm(ctx.search) * m.stat.meaninertia * max(1, m.nv)
  gtol = m.opt.tolerance * m.opt.ls_tolerance * smag

  # compute Mv, Jv
  mv = support.mul_m(m, d, ctx.search)
  jv = (d._impl or d).efc_J @ ctx.search

  # prepare quadratics
  quad_gauss = mx.stack([
      ctx.gauss,
      mx.sum(ctx.search * ctx.Ma) - mx.sum(ctx.search * d.qfrc_smooth),
      0.5 * mx.sum(ctx.search * mv),
  ])
  quad = mx.stack([
      0.5 * ctx.Jaref * ctx.Jaref,
      jv * ctx.Jaref,
      0.5 * jv * jv,
  ])
  quad = (quad * (d._impl or d).efc_D).T

  uu = mx.array(0.0)
  v0 = mx.array(0.0)
  uv = mx.array(0.0)
  vv = mx.array(0.0)
  if m.opt.cone == ConeType.ELLIPTIC:
    mask = (d._impl or d).contact.dim > 1
    mask_idx = np.nonzero(mask)[0]
    if mask_idx.size > 0:
      # complete vector quadratic (for bottom zone)
      for ii in range(mask_idx.size):
        condim = int((d._impl or d).contact.dim[mask_idx[ii]])
        addr = int((d._impl or d).contact.efc_address[mask_idx[ii]])
        for jj in range(1, condim):
          quad_sum = quad[addr] + quad[addr + jj]
          # Scatter back
          quad = mx.concatenate([
              quad[:addr], quad_sum[None, :], quad[addr + 1:]
          ], axis=0)

      # rescale to make primal cone circular
      jv_padded = mx.concatenate([jv, mx.zeros(3)])
      efc_elliptic = (d._impl or d).contact.efc_address[mask_idx]
      v_list = []
      for addr in efc_elliptic:
        addr = int(addr)
        v_list.append(jv_padded[addr:addr + 6])
      v = mx.stack(v_list) * ctx.fri

      uu = mx.sum(ctx.u[:, 1:] * ctx.u[:, 1:], axis=1)
      v0 = v[:, 0]
      uv = mx.sum(ctx.u[:, 1:] * v[:, 1:], axis=1)
      vv = mx.sum(v[:, 1:] * v[:, 1:], axis=1)

  point_fn = lambda a: _LSPoint.create(
      m, d, ctx, a, jv, quad, quad_gauss, uu, v0, uv, vv
  )

  def _ls_cond(lo, hi, swap, ls_iter):
    done = ls_iter >= m.opt.ls_iterations
    done = done or (not swap)
    lo_d0 = float(lo.deriv_0.item())
    hi_d0 = float(hi.deriv_0.item())
    gtol_f = float(gtol.item()) if isinstance(gtol, mx.array) else float(gtol)
    done = done or (lo_d0 < 0 and lo_d0 > -gtol_f)
    done = done or (hi_d0 > 0 and hi_d0 < gtol_f)
    return not done

  def _ls_body(lo, hi, ls_iter):
    lo_next = point_fn(lo.alpha - lo.deriv_0 / lo.deriv_1)
    hi_next = point_fn(hi.alpha - hi.deriv_0 / hi.deriv_1)
    mid = point_fn(0.5 * (lo.alpha + hi.alpha))

    in_bracket = lambda x, y: ((x < y) & (y < 0)) | ((x > y) & (y > 0))

    swap_lo_next = in_bracket(lo.deriv_0, lo_next.deriv_0)
    lo = _tree_map(lambda x, y: mx.where(swap_lo_next, y, x), lo, lo_next)
    swap_lo_mid = in_bracket(lo.deriv_0, mid.deriv_0)
    lo = _tree_map(lambda x, y: mx.where(swap_lo_mid, y, x), lo, mid)
    swap_lo_hi_next = in_bracket(lo.deriv_0, hi_next.deriv_0)
    lo = _tree_map(lambda x, y: mx.where(swap_lo_hi_next, y, x), lo, hi_next)
    swap_hi_next = in_bracket(hi.deriv_0, hi_next.deriv_0)
    hi = _tree_map(lambda x, y: mx.where(swap_hi_next, y, x), hi, hi_next)
    swap_hi_mid = in_bracket(hi.deriv_0, mid.deriv_0)
    hi = _tree_map(lambda x, y: mx.where(swap_hi_mid, y, x), hi, mid)
    swap_hi_lo_next = in_bracket(hi.deriv_0, lo_next.deriv_0)
    hi = _tree_map(lambda x, y: mx.where(swap_hi_lo_next, y, x), hi, lo_next)
    swap = swap_lo_next | swap_lo_mid | swap_lo_hi_next
    swap = swap | swap_hi_next | swap_hi_mid | swap_hi_lo_next

    return lo, hi, swap, ls_iter + 1

  # initialize interval
  p0 = point_fn(mx.array(0.0))
  lo = point_fn(p0.alpha - p0.deriv_0 / p0.deriv_1)
  lesser = lo.deriv_0 < p0.deriv_0
  hi = _tree_map(lambda x, y: mx.where(lesser, x, y), p0, lo)
  lo = _tree_map(lambda x, y: mx.where(lesser, x, y), lo, p0)

  # While loop (replaces jax.lax.scan-based while_loop)
  swap = True
  ls_iter = 0
  for _ in range(m.opt.ls_iterations):
    if not _ls_cond(lo, hi, swap, ls_iter):
      break
    lo, hi, swap_arr, ls_iter = _ls_body(lo, hi, ls_iter)
    swap = bool(swap_arr.item()) if isinstance(swap_arr, mx.array) else bool(swap_arr)

  # move to new solution if improved
  improved = (lo.cost < p0.cost) | (hi.cost < p0.cost)
  alpha = mx.where(lo.cost < hi.cost, lo.alpha, hi.alpha)
  qacc = ctx.qacc + improved * ctx.search * alpha
  ma = ctx.Ma + improved * mv * alpha
  jaref = ctx.Jaref + improved * jv * alpha

  ctx = ctx.replace(qacc=qacc, Ma=ma, Jaref=jaref)

  return ctx


def solve(m: Model, d: Data) -> Data:
  """Finds forces that satisfy constraints using conjugate gradient descent."""
  if not isinstance(m.opt._impl, OptionMLX):
    pass  # MLX port: backend check removed

  def _cond(ctx: Context) -> bool:
    improvement = _rescale(m, ctx.prev_cost - ctx.cost)
    gradient = _rescale(m, math.norm(ctx.grad))

    done = int(ctx.solver_niter.item()) >= m.opt.iterations
    done = done or (float(improvement.item()) < float(m.opt.tolerance.item()))
    done = done or (float(gradient.item()) < float(m.opt.tolerance.item()))

    return not done

  def _body(ctx: Context) -> Context:
    ctx = _linesearch(m, d, ctx)
    prev_grad, prev_Mgrad = ctx.grad, ctx.Mgrad
    ctx = _update_constraint(m, d, ctx)
    ctx = _update_gradient(m, d, ctx)

    if m.opt.solver == SolverType.NEWTON:
      search = -ctx.Mgrad
    else:
      # polak-ribiere:
      beta = mx.sum(ctx.grad * (ctx.Mgrad - prev_Mgrad))
      beta = beta / mx.maximum(mx.array(mujoco.mjMINVAL), mx.sum(prev_grad * prev_Mgrad))
      beta = mx.maximum(mx.array(0.0), beta)
      search = -ctx.Mgrad + beta * ctx.search
    ctx = ctx.replace(search=search, solver_niter=ctx.solver_niter + 1)

    return ctx

  # warmstart:
  qacc = d.qacc_smooth
  if not m.opt.disableflags & DisableBit.WARMSTART:
    warm = Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
    smth = Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
    qacc = mx.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
  d = d.replace(qacc=qacc)

  ctx = Context.create(m, d)

  # while loop (replaces jax.lax.while_loop)
  if m.opt.iterations == 1:
    ctx = _body(ctx)
  else:
    for _ in range(m.opt.iterations):
      if not _cond(ctx):
        break
      ctx = _body(ctx)

  d = d.tree_replace({
      'qfrc_constraint': ctx.qfrc_constraint,
      'qacc': ctx.qacc,
      '_impl.efc_force': ctx.efc_force,
  })

  return d
