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
"""Forward step functions (MLX port)."""

from typing import Optional, Sequence

import mlx.core as mx
import mujoco
from mujoco.mjx_mlx._src import collision_driver
from mujoco.mjx_mlx._src import constraint
from mujoco.mjx_mlx._src import derivative
from mujoco.mjx_mlx._src import math
from mujoco.mjx_mlx._src import passive as passive_mod
from mujoco.mjx_mlx._src import scan
from mujoco.mjx_mlx._src import sensor
from mujoco.mjx_mlx._src import smooth
from mujoco.mjx_mlx._src import solver
from mujoco.mjx_mlx._src import support
from mujoco.mjx_mlx._src.types import BiasType
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import DisableBit
from mujoco.mjx_mlx._src.types import DynType
from mujoco.mjx_mlx._src.types import GainType
from mujoco.mjx_mlx._src.types import IntegratorType
from mujoco.mjx_mlx._src.types import JointType
from mujoco.mjx_mlx._src.types import Model
from mujoco.mjx_mlx._src.types import TrnType
from mujoco.mjx_mlx._src.dataclasses import tree_map
import numpy as np

# RK4 tableau
_RK4_A = np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 1.0],
])
_RK4_B = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])


def fwd_position(m: Model, d: Data) -> Data:
  """Position-dependent computations."""
  d = smooth.kinematics(m, d)
  d = smooth.com_pos(m, d)
  d = smooth.camlight(m, d)
  d = smooth.tendon(m, d)
  d = smooth.crb(m, d)
  d = smooth.tendon_armature(m, d)
  d = smooth.factor_m(m, d)
  d = collision_driver.collision(m, d)
  d = constraint.make_constraint(m, d)
  d = smooth.transmission(m, d)
  return d


def fwd_velocity(m: Model, d: Data) -> Data:
  """Velocity-dependent computations."""
  d = d.tree_replace({
      '_impl.actuator_velocity': (d._impl or d).actuator_moment @ mx.array(d.qvel),
      '_impl.ten_velocity': (d._impl or d).ten_J @ mx.array(d.qvel),
  })
  d = smooth.com_vel(m, d)
  d = passive_mod.passive(m, d)
  d = smooth.rne(m, d)
  d = smooth.tendon_bias(m, d)
  return d


def fwd_actuation(m: Model, d: Data) -> Data:
  """Actuation-dependent computations."""
  if not m.nu or m.opt.disableflags & DisableBit.ACTUATION:
    return d.replace(
        act_dot=mx.zeros((m.na,)),
        qfrc_actuator=mx.zeros((m.nv,)),
    )

  ctrl = mx.array(d.ctrl) if not isinstance(d.ctrl, mx.array) else d.ctrl
  if not m.opt.disableflags & DisableBit.CLAMPCTRL:
    ctrlrange = mx.where(
        m.actuator_ctrllimited[:, None],
        m.actuator_ctrlrange,
        mx.array([-mx.inf, mx.inf]),
    )
    ctrl = mx.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

  # act_dot for stateful actuators
  def get_act_dot(dyn_typ, dyn_prm, ctrl, act):
    if dyn_typ == DynType.NONE:
      act_dot = mx.array(0.0)
    elif dyn_typ == DynType.INTEGRATOR:
      act_dot = ctrl
    elif dyn_typ in (DynType.FILTER, DynType.FILTEREXACT):
      act_dot = (ctrl - act) / mx.clip(dyn_prm[0], a_min=mujoco.mjMINVAL)
    elif dyn_typ == DynType.MUSCLE:
      act_dot = support.muscle_dynamics(ctrl, act, dyn_prm)
    else:
      raise NotImplementedError(f'dyntype {dyn_typ.name} not implemented.')
    return act_dot

  act_dot = mx.zeros((m.na,))
  if m.na:
    act_dot = scan.flat(
        m,
        get_act_dot,
        'uuua',
        'a',
        m.actuator_dyntype,
        m.actuator_dynprm,
        ctrl,
        d.act,
        group_by='u',
    )

  ctrl_act = ctrl
  if m.na:
    act_last_dim = d.act[m.actuator_actadr + m.actuator_actnum - 1]
    ctrl_act = mx.where(m.actuator_actadr == -1, ctrl, act_last_dim)

  def get_force(*args):
    gain_t, gain_p, bias_t, bias_p, len_, vel, ctrl_act, len_range, acc0 = args

    typ, prm = GainType(gain_t), gain_p
    if typ == GainType.FIXED:
      gain = prm[0]
    elif typ == GainType.AFFINE:
      gain = prm[0] + prm[1] * len_ + prm[2] * vel
    elif typ == GainType.MUSCLE:
      gain = support.muscle_gain(len_, vel, len_range, acc0, prm)
    else:
      raise RuntimeError(f'unrecognized gaintype {typ.name}.')

    typ, prm = BiasType(bias_t), bias_p
    bias = mx.array(0.0)
    if typ == BiasType.AFFINE:
      bias = prm[0] + prm[1] * len_ + prm[2] * vel
    elif typ == BiasType.MUSCLE:
      bias = support.muscle_bias(len_, len_range, acc0, prm)

    return gain * ctrl_act + bias

  force = scan.flat(
      m,
      get_force,
      'uuuuuuuuu',
      'u',
      m.actuator_gaintype,
      m.actuator_gainprm,
      m.actuator_biastype,
      m.actuator_biasprm,
      d.actuator_length,
      (d._impl or d).actuator_velocity,
      ctrl_act,
      mx.array(m.actuator_lengthrange),
      mx.array(m.actuator_acc0),
      group_by='u',
  )

  # tendon total force clamping
  if np.any(m.tendon_actfrclimited):
    (tendon_actfrclimited_id,) = np.nonzero(m.tendon_actfrclimited)
    actuator_tendon = m.actuator_trntype == TrnType.TENDON

    force_mask = [
        actuator_tendon & (m.actuator_trnid[:, 0] == tendon_id)
        for tendon_id in tendon_actfrclimited_id
    ]
    force_ids = np.concatenate([np.nonzero(mask)[0] for mask in force_mask])
    force_mat = np.array(force_mask)[:, force_ids]
    tendon_total_force = mx.array(force_mat) @ force[force_ids]

    force_scaling = mx.where(
        tendon_total_force < m.tendon_actfrcrange[tendon_actfrclimited_id, 0],
        m.tendon_actfrcrange[tendon_actfrclimited_id, 0] / tendon_total_force,
        1,
    )
    force_scaling = mx.where(
        tendon_total_force > m.tendon_actfrcrange[tendon_actfrclimited_id, 1],
        m.tendon_actfrcrange[tendon_actfrclimited_id, 1] / tendon_total_force,
        force_scaling,
    )

    tendon_forces = force[force_ids] * (mx.array(force_mat).T @ force_scaling)
    # MLX scatter: convert to numpy, update, convert back
    force_np = np.array(force)
    tendon_np = np.array(tendon_forces)
    force_np[force_ids] = tendon_np
    force = mx.array(force_np)

  forcerange = mx.where(
      m.actuator_forcelimited[:, None],
      m.actuator_forcerange,
      mx.array([-mx.inf, mx.inf]),
  )
  force = mx.clip(force, forcerange[:, 0], forcerange[:, 1])

  qfrc_actuator = (d._impl or d).actuator_moment.T @ force

  if m.ngravcomp:
    # actuator-level gravity compensation, skip if added as passive force
    qfrc_actuator = qfrc_actuator + d.qfrc_gravcomp * m.jnt_actgravcomp[m.dof_jntid]

  # clamp qfrc_actuator
  actfrcrange = mx.where(
      m.jnt_actfrclimited[:, None],
      m.jnt_actfrcrange,
      mx.array([-mx.inf, mx.inf]),
  )
  actfrcrange = mx.take(actfrcrange, mx.array(np.array(m.dof_jntid)), axis=0)
  qfrc_actuator = mx.clip(qfrc_actuator, actfrcrange[:, 0], actfrcrange[:, 1])

  d = d.replace(
      act_dot=act_dot, qfrc_actuator=qfrc_actuator, actuator_force=force
  )
  return d


def fwd_acceleration(m: Model, d: Data) -> Data:
  """Add up all non-constraint forces, compute qacc_smooth."""
  qfrc_applied = d.qfrc_applied + support.xfrc_accumulate(m, d)
  qfrc_smooth = d.qfrc_passive - d.qfrc_bias + d.qfrc_actuator + qfrc_applied
  qacc_smooth = smooth.solve_m(m, d, qfrc_smooth)
  d = d.replace(qfrc_smooth=qfrc_smooth, qacc_smooth=qacc_smooth)
  return d


def _integrate_pos(
    jnt_typs: Sequence[str], qpos: mx.array, qvel: mx.array, dt: mx.array
) -> mx.array:
  """Integrate position given velocity."""
  qs, qi, vi = [], 0, 0

  for jnt_typ in jnt_typs:
    if jnt_typ == JointType.FREE:
      pos = qpos[qi : qi + 3] + dt * qvel[vi : vi + 3]
      quat = math.quat_integrate(
          qpos[qi + 3 : qi + 7], qvel[vi + 3 : vi + 6], dt
      )
      qs.append(mx.concatenate([pos, quat]))
      qi, vi = qi + 7, vi + 6
    elif jnt_typ == JointType.BALL:
      quat = math.quat_integrate(qpos[qi : qi + 4], qvel[vi : vi + 3], dt)
      qs.append(quat)
      qi, vi = qi + 4, vi + 3
    elif jnt_typ in (JointType.HINGE, JointType.SLIDE):
      pos = qpos[qi] + dt * qvel[vi]
      qs.append(mx.array([pos]))
      qi, vi = qi + 1, vi + 1
    else:
      raise RuntimeError(f'unrecognized joint type: {jnt_typ}')

  return mx.concatenate(qs) if qs else mx.zeros((0,))


def _next_activation(m: Model, d: Data, act_dot: mx.array) -> mx.array:
  """Returns the next act given the current act_dot, after clamping."""
  act = d.act

  if not m.na:
    return act

  actrange = mx.where(
      m.actuator_actlimited[:, None],
      m.actuator_actrange,
      mx.array([-mx.inf, mx.inf]),
  )

  def fn(dyntype, dynprm, act, act_dot, actrange):
    if dyntype == DynType.FILTEREXACT:
      tau = mx.clip(dynprm[0], a_min=mujoco.mjMINVAL)
      act = act + act_dot * tau * (1 - mx.exp(-m.opt.timestep / tau))
    else:
      act = act + act_dot * m.opt.timestep
    act = mx.clip(act, actrange[0], actrange[1])
    return act

  args = (m.actuator_dyntype, m.actuator_dynprm, act, act_dot, actrange)
  act = scan.flat(m, fn, 'uuaau', 'a', *args, group_by='u')

  return mx.reshape(act, (m.na,))


def _advance(
    m: Model,
    d: Data,
    act_dot: mx.array,
    qacc: mx.array,
    qvel: Optional[mx.array] = None,
) -> Data:
  """Advance state and time given activation derivatives and acceleration."""
  act = _next_activation(m, d, act_dot)

  # advance velocities
  d = d.replace(qvel=d.qvel + qacc * m.opt.timestep)

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  qvel = d.qvel if qvel is None else qvel
  integrate_fn = lambda *args: _integrate_pos(*args, dt=m.opt.timestep)
  qpos = scan.flat(m, integrate_fn, 'jqv', 'q', m.jnt_type, d.qpos, qvel)

  # advance time
  time = d.time + m.opt.timestep

  # save qacc for next step warmstart
  d = d.replace(qacc_warmstart=d.qacc)

  return d.replace(act=act, qpos=qpos, time=time)


def euler(m: Model, d: Data) -> Data:
  """Euler integrator, semi-implicit in velocity."""

  # integrate damping implicitly
  qacc = d.qacc
  if not m.opt.disableflags & DisableBit.EULERDAMP:
    if support.is_sparse(m):
      # Sparse mass matrix: add damping to diagonal entries
      # Build additive delta and add in one shot
      delta = mx.zeros_like((d._impl or d).qM)
      diag_indices = np.array([int(m.dof_Madr[i]) for i in range(m.nv)])
      damping_vals = m.opt.timestep * m.dof_damping
      # Scatter damping into delta at diagonal addresses
      delta_np = np.zeros((d._impl or d).qM.shape, dtype=np.float32)
      for i in range(m.nv):
        delta_np[diag_indices[i]] += float(damping_vals[i])
      qM = (d._impl or d).qM + mx.array(delta_np)
    else:
      qM = (d._impl or d).qM + mx.diag(mx.array(m.opt.timestep * np.array(m.dof_damping)))
    dh = d.tree_replace({'_impl.qM': qM})
    dh = smooth.factor_m(m, dh)
    qfrc = d.qfrc_smooth + d.qfrc_constraint
    qacc = smooth.solve_m(m, dh, qfrc)
  return _advance(m, d, d.act_dot, qacc)


def rungekutta4(m: Model, d: Data) -> Data:
  """Runge-Kutta explicit order 4 integrator."""
  d0 = d
  A, B = _RK4_A, _RK4_B
  C = np.tril(A).sum(axis=0)
  T = d.time + mx.array(C) * m.opt.timestep

  kqvel = d.qvel  # intermediate RK solution
  # RK solutions sum
  qvel = tree_map(lambda k: B[0] * k, kqvel) if isinstance(kqvel, mx.array) else B[0] * kqvel
  qacc = B[0] * d.qacc
  act_dot_sum = B[0] * d.act_dot

  integrate_fn = lambda *args: _integrate_pos(*args, dt=m.opt.timestep)

  # Unroll 3 RK4 stages (replacing jax.lax.scan with Python loop)
  for stage in range(3):
    a = A[stage][stage]  # diagonal of A
    b = B[stage + 1]
    t = T[stage]

    # get intermediate RK solutions
    dqvel = a * kqvel
    dqacc = a * d.qacc
    dact_dot = a * d.act_dot

    kqpos = scan.flat(m, integrate_fn, 'jqv', 'q', m.jnt_type, d0.qpos, dqvel)
    kact = d0.act + dact_dot * m.opt.timestep
    kqvel = d0.qvel + dqacc * m.opt.timestep
    d = d.replace(qpos=kqpos, qvel=kqvel, act=kact, time=t)
    d = forward(m, d)

    qvel = qvel + b * kqvel
    qacc = qacc + b * d.qacc
    act_dot_sum = act_dot_sum + b * d.act_dot

  d = d.replace(qpos=d0.qpos, qvel=d0.qvel, act=d0.act, time=d0.time)
  d = _advance(m, d, act_dot_sum, qacc, qvel)
  return d


def implicit(m: Model, d: Data) -> Data:
  """Integrates fully implicit in velocity."""
  qderiv = derivative.deriv_smooth_vel(m, d)

  qacc = d.qacc
  if qderiv is not None:
    qm = support.full_m(m, d) if support.is_sparse(m) else (d._impl or d).qM
    qm = qm - m.opt.timestep * qderiv
    # Cholesky factorization and solve (MLX does not have scipy-style cho_factor)
    # Use direct linear solve: qacc = qm^{-1} @ qfrc
    qfrc = d.qfrc_smooth + d.qfrc_constraint
    # MLX linalg solve: solve qm @ qacc = qfrc
    qacc = mx.linalg.solve(qm, qfrc)

  return _advance(m, d, d.act_dot, qacc)


def forward(m: Model, d: Data) -> Data:
  """Forward dynamics."""
  d = fwd_position(m, d)
  d = sensor.sensor_pos(m, d)
  d = fwd_velocity(m, d)
  d = sensor.sensor_vel(m, d)
  d = fwd_actuation(m, d)
  d = fwd_acceleration(m, d)

  if (d._impl or d).efc_J.size == 0:
    d = d.replace(qacc=d.qacc_smooth)
    return d

  d = solver.solve(m, d)
  d = sensor.sensor_acc(m, d)

  return d


def step(m: Model, d: Data) -> Data:
  """Advance simulation."""
  d = forward(m, d)

  if m.opt.integrator == IntegratorType.EULER:
    d = euler(m, d)
  elif m.opt.integrator == IntegratorType.RK4:
    d = rungekutta4(m, d)
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    d = implicit(m, d)
  else:
    raise NotImplementedError(f'integrator {m.opt.integrator} not implemented.')

  return d
