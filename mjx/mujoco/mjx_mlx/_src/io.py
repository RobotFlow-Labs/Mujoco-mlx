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
"""Functions to initialize, load, or save data.

MLX port: removes JAX device placement, WARP/C/CPP backends, and CUDA-specific
code. All arrays are created via mlx.core (mx) instead of jax.numpy (jnp).
Only the MLX (pure-array) backend is supported.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mujoco
from mujoco.mjx_mlx._src import collision_driver
from mujoco.mjx_mlx._src import constraint
from mujoco.mjx_mlx._src import mesh
from mujoco.mjx_mlx._src import support
from mujoco.mjx_mlx._src import types
import numpy as np
import scipy


def _numpy_to_mlx(tree):
  """Convert all numpy arrays in a dataclass tree to MLX arrays."""
  from mujoco.mjx_mlx._src import dataclasses as mjx_dataclasses  # pylint: disable=g-import-not-at-top

  def _convert(value):
    if isinstance(value, mx.array):
      return value
    if isinstance(value, np.ndarray):
      return mx.array(value)
    return value

  if hasattr(tree, '__dataclass_fields__'):
    return mjx_dataclasses.tree_map(_convert, tree)
  return tree


def _put_option(
    o: mujoco.MjOption,
) -> types.Option:
  """Returns mjx_mlx.Option given mujoco.MjOption."""
  if o.integrator not in set(types.IntegratorType):
    raise NotImplementedError(f'{mujoco.mjtIntegrator(o.integrator)}')

  if o.cone not in set(types.ConeType):
    raise NotImplementedError(f'{mujoco.mjtCone(o.cone)}')

  if o.jacobian not in set(types.JacobianType):
    raise NotImplementedError(f'{mujoco.mjtJacobian(o.jacobian)}')

  if o.solver not in set(types.SolverType):
    raise NotImplementedError(f'{mujoco.mjtSolver(o.solver)}')

  for i in range(mujoco.mjtEnableBit.mjNENABLE):
    if o.enableflags & 2**i and 2 ** i not in set(types.EnableBit):
      raise NotImplementedError(f'{mujoco.mjtEnableBit(2**i)}')

  fields = {
      f.name: getattr(o, f.name, None)
      for f in types.Option.fields()
      if f.name != 'has_fluid_params'
  }
  fields['integrator'] = types.IntegratorType(o.integrator)
  fields['cone'] = types.ConeType(o.cone)
  fields['solver'] = types.SolverType(o.solver)
  fields['disableflags'] = types.DisableBit(o.disableflags)
  fields['enableflags'] = types.EnableBit(o.enableflags)
  fields['jacobian'] = types.JacobianType(o.jacobian)

  has_fluid_params = o.density > 0 or o.viscosity > 0 or o.wind.any()
  implicitfast = o.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  if implicitfast and has_fluid_params:
    raise NotImplementedError('implicitfast not implemented for fluid drag.')
  fields['has_fluid_params'] = has_fluid_params

  return types.Option(**fields)


def _put_statistic(s: mujoco.MjStatistic) -> types.Statistic:
  """Puts mujoco.MjStatistic into mjx_mlx.Statistic."""
  return types.Statistic(
      meaninertia=s.meaninertia,
      meanmass=s.meanmass,
      meansize=s.meansize,
      extent=s.extent,
      center=s.center,
  )


def put_model(
    m: mujoco.MjModel,
) -> types.Model:
  """Converts mujoco.MjModel into an mjx_mlx.Model with MLX arrays.

  Args:
    m: the model to convert

  Returns:
    an mjx_mlx.Model with MLX arrays
  """
  if m.nflex:
    raise NotImplementedError('Flex not implemented for MLX backend.')

  # contact sensor checks
  is_contact_sensor = m.sensor_type == types.SensorType.CONTACT
  if is_contact_sensor.any():
    objtype = m.sensor_objtype[is_contact_sensor]
    reftype = m.sensor_reftype[is_contact_sensor]
    contact_sensor_type = set(np.concatenate([objtype, reftype]))

    if types.ObjType.SITE in set(objtype):
      raise NotImplementedError(
          'Contact sensor with site matching semantics not implemented for MLX'
          ' backend.'
      )
    if types.ObjType.BODY in contact_sensor_type:
      raise NotImplementedError(
          'Contact sensor with body matching semantics not implemented for MLX'
          ' backend.'
      )
    if types.ObjType.XBODY in contact_sensor_type:
      raise NotImplementedError(
          'Contact sensor with subtree matching semantics not implemented for'
          ' MLX backend.'
      )
    if (m.sensor_intprm[is_contact_sensor, 1] == 3).any():
      raise NotImplementedError(
          'Contact sensor with netforce reduction not implemented for MLX'
          ' backend.'
      )

  mesh_geomid = set()
  for g1, g2, ip in collision_driver.geom_pairs(m):
    t1, t2 = m.geom_type[[g1, g2]]
    if not collision_driver.has_collision_fn(t1, t2):
      t1, t2 = mujoco.mjtGeom(t1), mujoco.mjtGeom(t2)
      raise NotImplementedError(f'({t1}, {t2}) collisions not implemented.')
    no_margin = {mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_HFIELD}
    if no_margin.intersection({t1, t2}):
      if ip != -1:
        margin = m.pair_margin[ip]
      else:
        margin = m.geom_margin[g1] + m.geom_margin[g2]
      if margin.any():
        t1, t2 = mujoco.mjtGeom(t1), mujoco.mjtGeom(t2)
        raise NotImplementedError(f'({t1}, {t2}) margin/gap not implemented.')
    for t, g in [(t1, g1), (t2, g2)]:
      if t == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_geomid.add(g)

  for enum_field, enum_type, mj_type in (
      (m.actuator_biastype, types.BiasType, mujoco.mjtBias),
      (m.actuator_dyntype, types.DynType, mujoco.mjtDyn),
      (m.actuator_gaintype, types.GainType, mujoco.mjtGain),
      (m.actuator_trntype, types.TrnType, mujoco.mjtTrn),
      (m.eq_type, types.EqType, mujoco.mjtEq),
      (m.sensor_type, types.SensorType, mujoco.mjtSensor),
      (m.wrap_type, types.WrapType, mujoco.mjtWrap),
  ):
    missing = set(enum_field) - set(enum_type)
    if missing:
      raise NotImplementedError(
          f'{[mj_type(m) for m in missing]} not supported'
      )

  mj_field_names = {f.name for f in types.Model.fields() if f.name not in (
      'dof_hasfrictionloss', 'tendon_hasfrictionloss', 'geom_rbound_hfield',
      'wrap_inside_maxiter', 'wrap_inside_tolerance', 'wrap_inside_z_init',
      'is_wrap_inside', 'mesh_convex', 'opt', 'stat',
  )}
  fields = {f: getattr(m, f) for f in mj_field_names}
  fields['cam_mat0'] = fields['cam_mat0'].reshape((-1, 3, 3))
  fields['opt'] = _put_option(m.opt)
  fields['stat'] = _put_statistic(m.stat)

  # MLX-specific fields (equivalent to JAX impl fields, but inlined)
  fields['dof_hasfrictionloss'] = fields['dof_frictionloss'] > 0
  fields['tendon_hasfrictionloss'] = fields['tendon_frictionloss'] > 0
  fields['geom_rbound_hfield'] = fields['geom_rbound']

  # spatial tendon wrap inside
  fields['wrap_inside_maxiter'] = 5
  fields['wrap_inside_tolerance'] = 1.0e-4
  fields['wrap_inside_z_init'] = 1.0 - 1.0e-5
  fields['is_wrap_inside'] = np.zeros(0, dtype=bool)
  if m.nsite:
    (wrap_id_geom,) = np.nonzero(
        (m.wrap_type == mujoco.mjtWrap.mjWRAP_SPHERE)
        | (m.wrap_type == mujoco.mjtWrap.mjWRAP_CYLINDER)
    )
    wrap_objid_geom = m.wrap_objid[wrap_id_geom]
    geom_pos = m.geom_pos[wrap_objid_geom]
    geom_size = m.geom_size[wrap_objid_geom, 0]

    side_id = np.round(m.wrap_prm[wrap_id_geom]).astype(int)
    side = m.site_pos[side_id]

    fields['is_wrap_inside'] = np.array(
        (np.linalg.norm(side - geom_pos, axis=1) < geom_size) & (side_id >= 0)
    )

  # Pre-compile meshes for MJX collisions.
  fields['mesh_convex'] = [None] * m.nmesh
  for i in mesh_geomid:
    dataid = m.geom_dataid[i]
    if fields['mesh_convex'][dataid] is None:
      fields['mesh_convex'][dataid] = mesh.convex(m, dataid)
  fields['mesh_convex'] = tuple(fields['mesh_convex'])

  model = types.Model(**{k: copy.copy(v) for k, v in fields.items()})

  # Convert numpy arrays to MLX arrays
  model = _numpy_to_mlx(model)
  return model


def _make_data_public_fields(m: types.Model) -> Dict[str, Any]:
  """Create public fields for the Data object."""
  float_ = np.float32
  zero_fields = {
      'time': (float_,),
      'qvel': (m.nv, float_),
      'act': (m.na, float_),
      'history': (m.nhistory, float_),
      'plugin_state': (m.npluginstate, float_),
      'qacc_warmstart': (m.nv, float_),
      'ctrl': (m.nu, float_),
      'qfrc_applied': (m.nv, float_),
      'xfrc_applied': (m.nbody, 6, float_),
      'mocap_pos': (m.nmocap, 3, float_),
      'mocap_quat': (m.nmocap, 4, float_),
      'qacc': (m.nv, float_),
      'act_dot': (m.na, float_),
      'userdata': (m.nuserdata, float_),
      'sensordata': (m.nsensordata, float_),
      'xpos': (m.nbody, 3, float_),
      'xquat': (m.nbody, 4, float_),
      'xmat': (m.nbody, 3, 3, float_),
      'xipos': (m.nbody, 3, float_),
      'ximat': (m.nbody, 3, 3, float_),
      'xanchor': (m.njnt, 3, float_),
      'xaxis': (m.njnt, 3, float_),
      'geom_xpos': (m.ngeom, 3, float_),
      'geom_xmat': (m.ngeom, 3, 3, float_),
      'site_xpos': (m.nsite, 3, float_),
      'site_xmat': (m.nsite, 3, 3, float_),
      'cam_xpos': (m.ncam, 3, float_),
      'cam_xmat': (m.ncam, 3, 3, float_),
      'subtree_com': (m.nbody, 3, float_),
      'actuator_force': (m.nu, float_),
      'actuator_length': (m.nu, float_),
      'qfrc_bias': (m.nv, float_),
      'qfrc_gravcomp': (m.nv, float_),
      'qfrc_fluid': (m.nv, float_),
      'qfrc_passive': (m.nv, float_),
      'qfrc_actuator': (m.nv, float_),
      'qfrc_smooth': (m.nv, float_),
      'qacc_smooth': (m.nv, float_),
      'qfrc_constraint': (m.nv, float_),
      'qfrc_inverse': (m.nv, float_),
      'cvel': (m.nbody, 6, float_),
      'cdof': (m.nv, 6, float_),
      'cdof_dot': (m.nv, 6, float_),
      'ten_length': (m.ntendon, float_),
  }
  zero_fields = {
      k: np.zeros(v[:-1], dtype=v[-1]) for k, v in zero_fields.items()
  }
  return zero_fields


def _make_data_contact(
    condim: np.ndarray, efc_address: np.ndarray
) -> types.Contact:
  """Create contact for the Data object."""
  ncon = condim.size
  float_ = np.float32
  int_ = np.int32
  contact = types.Contact(
      dist=np.zeros((ncon,), dtype=float_),
      pos=np.zeros((ncon, 3), dtype=float_),
      frame=np.zeros((ncon, 3, 3), dtype=float_),
      includemargin=np.zeros((ncon,), dtype=float_),
      friction=np.zeros((ncon, 5), dtype=float_),
      solref=np.zeros((ncon, mujoco.mjNREF), dtype=float_),
      solreffriction=np.zeros((ncon, mujoco.mjNREF), dtype=float_),
      solimp=np.zeros((ncon, mujoco.mjNIMP), dtype=float_),
      dim=condim,
      geom1=np.full((ncon,), -1, dtype=int_),
      geom2=np.full((ncon,), -1, dtype=int_),
      geom=np.full((ncon, 2), -1, dtype=int_),
      efc_address=efc_address,
  )
  return contact


def make_data(
    m: Union[types.Model, mujoco.MjModel],
) -> types.Data:
  """Allocate and initialize Data for the MLX backend.

  Args:
    m: the model to use (mjx_mlx.Model or mujoco.MjModel)

  Returns:
    an initialized mjx_mlx.Data with MLX arrays
  """
  # Use the original JAX collision_driver and constraint modules which
  # operate on numpy/MjModel data before array placement.
  from mujoco.mjx_mlx._src import types as jax_types  # pylint: disable=g-import-not-at-top

  dim = collision_driver.make_condim(m, impl=jax_types.Impl.JAX)
  efc_type = constraint.make_efc_type(m, dim)
  ne, nf, nl, nc = constraint.counts(efc_type)
  ncon, nefc = dim.size, ne + nf + nl + nc
  efc_address = constraint.make_efc_address(m, dim, efc_type)

  float_ = np.float32
  int_ = np.int32
  contact = _make_data_contact(dim, efc_address)

  if m.opt.cone == types.ConeType.ELLIPTIC and np.any(contact.dim == 1):
    raise NotImplementedError(
        'condim=1 with ConeType.ELLIPTIC not implemented.'
    )

  zero_impl_fields = {
      'solver_niter': (int_,),
      'cinert': (m.nbody, 10, float_),
      'ten_wrapadr': (m.ntendon, np.int32),
      'ten_wrapnum': (m.ntendon, np.int32),
      'ten_J': (m.ntendon, m.nv, float_),
      'wrap_obj': (m.nwrap, 2, np.int32),
      'wrap_xpos': (m.nwrap, 6, float_),
      'actuator_moment': (m.nu, m.nv, float_),
      'crb': (m.nbody, 10, float_),
      'qM': (m.nM, float_) if support.is_sparse(m) else (m.nv, m.nv, float_),
      'M': (m.nC, float_),
      'qLD': (m.nC, float_) if support.is_sparse(m) else (m.nv, m.nv, float_),
      'qLDiagInv': (m.nv, float_) if support.is_sparse(m) else (0, float_),
      'ten_velocity': (m.ntendon, float_),
      'actuator_velocity': (m.nu, float_),
      'cacc': (m.nbody, 6, float_),
      'cfrc_int': (m.nbody, 6, float_),
      'cfrc_ext': (m.nbody, 6, float_),
      'subtree_linvel': (m.nbody, 3, float_),
      'subtree_angmom': (m.nbody, 3, float_),
      'efc_J': (nefc, m.nv, float_),
      'efc_pos': (nefc, float_),
      'efc_margin': (nefc, float_),
      'efc_frictionloss': (nefc, float_),
      'efc_D': (nefc, float_),
      'efc_aref': (nefc, float_),
      'efc_force': (nefc, float_),
  }
  zero_impl_fields = {
      k: np.zeros(v[:-1], dtype=v[-1]) for k, v in zero_impl_fields.items()
  }

  d = types.Data(
      ne=ne,
      nf=nf,
      nl=nl,
      nefc=nefc,
      ncon=ncon,
      contact=contact,
      efc_type=efc_type,
      qpos=np.array(m.qpos0, dtype=float_),
      eq_active=m.eq_active0,
      **_make_data_public_fields(m),
      **zero_impl_fields,
  )

  if m.nmocap:
    body_mask = m.body_mocapid >= 0
    body_pos = m.body_pos[body_mask]
    body_quat = m.body_quat[body_mask]
    d = d.replace(
        mocap_pos=body_pos[m.body_mocapid[body_mask]],
        mocap_quat=body_quat[m.body_mocapid[body_mask]],
    )

  # Convert numpy arrays to MLX arrays
  d = _numpy_to_mlx(d)
  return d


def _put_contact(
    c: mujoco._structs._MjContactList,
    dim: np.ndarray,
    efc_address: np.ndarray,
) -> Tuple[types.Contact, np.ndarray]:
  """Converts mujoco.structs._MjContactList into mjx_mlx.Contact."""
  fields = {f.name: getattr(c, f.name) for f in types.Contact.fields()}
  fields['frame'] = fields['frame'].reshape((-1, 3, 3))

  contact_map = -np.ones_like(dim)
  for i, di in enumerate(fields['dim']):
    space = [j for j, dj in enumerate(dim) if di == dj and contact_map[j] == -1]
    if not space:
      raise ValueError(f'unable to place Contact[{i}], no space in condim {di}')
    contact_map[space[0]] = i

  if contact_map.size > 0:
    # reorganize contact with a zero contact at the end for -1
    zero = {
        k: np.zeros((1,) + v.shape[1:], dtype=v.dtype) for k, v in fields.items()
    }
    zero['dist'][:] = 1e10
    fields = {k: np.concatenate([fields[k], zero[k]]) for k in fields}
    fields = {k: v[contact_map] for k, v in fields.items()}

  fields['dim'] = dim
  fields['efc_address'] = efc_address

  return types.Contact(**fields), contact_map


def _put_data_public_fields(d: mujoco.MjData) -> Dict[str, Any]:
  """Returns public fields from mujoco.MjData in a dictionary."""
  fields = {
      f.name: getattr(d, f.name)
      for f in types.Data.fields()
      if f.name not in (
          'ne', 'nf', 'nl', 'nefc', 'ncon', 'contact', 'efc_type',
          'solver_niter', 'cinert', 'ten_wrapadr', 'ten_wrapnum', 'ten_J',
          'wrap_obj', 'wrap_xpos', 'actuator_moment', 'crb', 'qM', 'M',
          'qLD', 'qLDiagInv', 'ten_velocity', 'actuator_velocity', 'cacc',
          'cfrc_int', 'cfrc_ext', 'subtree_linvel', 'subtree_angmom',
          'efc_J', 'efc_pos', 'efc_margin', 'efc_frictionloss', 'efc_D',
          'efc_aref', 'efc_force',
      )
  }
  # MJX uses square matrices for these fields:
  for fname in ('xmat', 'ximat', 'geom_xmat', 'site_xmat', 'cam_xmat'):
    if fname in fields:
      fields[fname] = fields[fname].reshape((-1, 3, 3))

  return fields


def put_data(
    m: mujoco.MjModel,
    d: mujoco.MjData,
) -> types.Data:
  """Puts mujoco.MjData into an mjx_mlx.Data with MLX arrays.

  Args:
    m: the model to use
    d: the data to convert

  Returns:
    an mjx_mlx.Data with MLX arrays
  """
  from mujoco.mjx_mlx._src import types as jax_types  # pylint: disable=g-import-not-at-top

  dim = collision_driver.make_condim(m, impl=jax_types.Impl.JAX)
  efc_type = constraint.make_efc_type(m, dim)
  efc_address = constraint.make_efc_address(m, dim, efc_type)
  ne, nf, nl, nc = constraint.counts(efc_type)
  ncon, nefc = dim.size, ne + nf + nl + nc

  for d_val, val, name in (
      (d.ncon, ncon, 'ncon'),
      (d.ne, ne, 'ne'),
      (d.nf, nf, 'nf'),
      (d.nl, nl, 'nl'),
      (d.nefc, nefc, 'nefc'),
  ):
    if d_val > val:
      raise ValueError(f'd.{name} too high, d.{name} = {d_val}, model = {val}')

  fields = _put_data_public_fields(d)

  # Implementation specific fields.
  impl_field_names = (
      'solver_niter', 'cinert', 'ten_wrapadr', 'ten_wrapnum', 'ten_J',
      'wrap_obj', 'wrap_xpos', 'actuator_moment', 'crb', 'qM', 'M',
      'qLD', 'qLDiagInv', 'ten_velocity', 'actuator_velocity', 'cacc',
      'cfrc_int', 'cfrc_ext', 'subtree_linvel', 'subtree_angmom',
      'efc_J', 'efc_pos', 'efc_margin', 'efc_frictionloss', 'efc_D',
      'efc_aref', 'efc_force',
  )
  impl_fields = {
      name: getattr(d, name)
      for name in impl_field_names
      if hasattr(d, name)
  }
  # MJX does not support islanding, so only transfer the first solver_niter
  impl_fields['solver_niter'] = impl_fields['solver_niter'][0]

  # convert sparse actuator_moment to dense matrix
  moment = np.zeros((m.nu, m.nv))
  mujoco.mju_sparse2dense(
      moment,
      d.actuator_moment,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
  )
  impl_fields['actuator_moment'] = moment

  # convert ten_J to dense matrix
  if m.ntendon:
    ten_J = np.zeros((m.ntendon, m.nv))
    mujoco.mju_sparse2dense(
        ten_J,
        d.ten_J,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
    )
  else:
    ten_J = np.zeros((m.ntendon, m.nv))
  impl_fields['ten_J'] = ten_J

  contact, contact_map = _put_contact(d.contact, dim, efc_address)

  # pad efc fields: MuJoCo efc arrays are sparse for inactive constraints.
  # efc_J is also optionally column-sparse (typically for large nv). MJX-MLX
  # is neither: it contains zeros for inactive constraints, and efc_J is always
  # (nefc, nv). This may change in the future.
  if mujoco.mj_isSparse(m):
    efc_j = np.zeros((d.efc_J_rownnz.shape[0], m.nv))
    mujoco.mju_sparse2dense(
        efc_j,
        impl_fields['efc_J'],
        d.efc_J_rownnz,
        d.efc_J_rowadr,
        d.efc_J_colind,
    )
    impl_fields['efc_J'] = efc_j
  else:
    impl_fields['efc_J'] = impl_fields['efc_J'].reshape(
        (-1 if m.nv else 0, m.nv)
    )

  # move efc rows to their correct offsets
  for fname in (
      'efc_J',
      'efc_pos',
      'efc_margin',
      'efc_frictionloss',
      'efc_D',
      'efc_aref',
      'efc_force',
  ):
    value = np.zeros((nefc, m.nv)) if fname == 'efc_J' else np.zeros(nefc)
    for i in range(3):
      value_beg = sum([ne, nf][:i])
      d_beg = sum([d.ne, d.nf][:i])
      size = [d.ne, d.nf, d.nl][i]
      value[value_beg : value_beg + size] = impl_fields[fname][
          d_beg : d_beg + size
      ]

    for id_to, id_from in enumerate(contact_map):
      if id_from == -1:
        continue
      num_rows = dim[id_to]
      if num_rows > 1 and m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
        num_rows = (num_rows - 1) * 2
      efc_i, efc_o = d.contact.efc_address[id_from], efc_address[id_to]
      if efc_i == -1:
        continue
      value[efc_o : efc_o + num_rows] = impl_fields[fname][
          efc_i : efc_i + num_rows
      ]

    impl_fields[fname] = value

  # convert qM and qLD if jacobian is dense
  if not support.is_sparse(m):
    impl_fields['qM'] = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, impl_fields['qM'], d.qM)
    try:
      impl_fields['qLD'], _ = scipy.linalg.cho_factor(impl_fields['qM'])
    except scipy.linalg.LinAlgError:
      impl_fields['qLD'] = np.zeros((m.nv, m.nv))
    impl_fields['qLDiagInv'] = np.zeros(0)

  # copy because conversion is from numpy
  data = types.Data(
      ne=ne,
      nf=nf,
      nl=nl,
      nefc=nefc,
      ncon=ncon,
      contact=contact,
      efc_type=efc_type,
      **{k: copy.copy(v) for k, v in impl_fields.items()},
      **{k: copy.copy(v) for k, v in fields.items()},
  )

  # Convert numpy arrays to MLX arrays
  data = _numpy_to_mlx(data)
  return data


def _get_contact(c: mujoco._structs._MjContactList, cx: types.Contact):
  """Converts mjx_mlx.Contact to mujoco._structs._MjContactList."""
  # Convert MLX arrays to numpy for writing back to MuJoCo structs
  dist = np.array(cx.dist) if isinstance(cx.dist, mx.array) else cx.dist
  con_id = np.nonzero(dist <= 0)[0]
  for field in types.Contact.fields():
    value = getattr(cx, field.name)
    if isinstance(value, mx.array):
      value = np.array(value)
    value = value[con_id]
    if field.name == 'frame':
      value = value.reshape((-1, 9))
    getattr(c, field.name)[:] = value


def get_data_into(
    result: Union[mujoco.MjData, List[mujoco.MjData]],
    m: mujoco.MjModel,
    d: types.Data,
):
  """Gets mjx_mlx.Data into an existing mujoco.MjData or list."""
  is_batched = isinstance(result, list)
  if is_batched:
    qpos = np.array(d.qpos) if isinstance(d.qpos, mx.array) else d.qpos
    if len(qpos.shape) < 2:
      raise ValueError('destination is a list, but d is not batched.')
  else:
    qpos = np.array(d.qpos) if isinstance(d.qpos, mx.array) else d.qpos
    if len(qpos.shape) >= 2:
      raise ValueError('destination is an MjData, but d is batched.')

  # Convert the full tree to numpy
  from mujoco.mjx_mlx._src import dataclasses as mjx_dataclasses  # pylint: disable=g-import-not-at-top
  d = mjx_dataclasses.tree_map(
      lambda x: np.array(x) if isinstance(x, mx.array) else x, d
  )

  batch_size = d.qpos.shape[0] if is_batched else 1

  dof_i, dof_j = [], []
  for i in range(m.nv):
    j = i
    while j > -1:
      dof_i.append(i)
      dof_j.append(j)
      j = m.dof_parentid[j]

  for i in range(batch_size):
    if is_batched:
      # Slice batch dimension from all fields
      d_i = mjx_dataclasses.tree_map(lambda x, i=i: x[i], d)
    else:
      d_i = d
    result_i = result[i] if is_batched else result

    ncon = (d_i.contact.dist <= 0).sum()
    efc_active = (d_i.efc_J != 0).any(axis=1)
    nefc = int(efc_active.sum())
    nj = (d_i.efc_J != 0).sum() if support.is_sparse(m) else nefc * m.nv

    if ncon != result_i.ncon or nefc != result_i.nefc or nj != result_i.nJ:
      mujoco._functions._realloc_con_efc(result_i, ncon=ncon, nefc=nefc, nJ=nj)  # pylint: disable=protected-access

    all_field_names = {f.name for f in types.Data.fields()}

    for field in types.Data.fields():
      if field.name not in mujoco.MjData.__dict__.keys():
        continue

      if field.name == 'contact':
        _get_contact(result_i.contact, d_i.contact)
        efc_map = np.cumsum(efc_active) - 1
        result_i.contact.efc_address[:] = efc_map[result_i.contact.efc_address]
        continue

      # MuJoCo actuator_moment is sparse, MJX uses a dense representation.
      if field.name == 'actuator_moment':
        moment_rownnz = np.zeros(m.nu, dtype=np.int32)
        moment_rowadr = np.zeros(m.nu, dtype=np.int32)
        moment_colind = np.zeros(m.nJmom, dtype=np.int32)
        actuator_moment = np.zeros(m.nJmom)
        if m.nu:
          mujoco.mju_dense2sparse(
              actuator_moment,
              d_i.actuator_moment,
              moment_rownnz,
              moment_rowadr,
              moment_colind,
          )
        result_i.moment_rownnz[:] = moment_rownnz
        result_i.moment_rowadr[:] = moment_rowadr
        result_i.moment_colind[:] = moment_colind
        result_i.actuator_moment[:] = actuator_moment
        continue

      # MuJoCo ten_J is sparse, MJX uses a dense representation.
      if field.name == 'ten_J':
        ten_j_rownnz = np.zeros(m.ntendon, dtype=np.int32)
        ten_j_rowadr = np.zeros(m.ntendon, dtype=np.int32)
        ten_j_colind = np.zeros(m.nJten, dtype=np.int32)
        ten_j = np.zeros(m.nJten)
        if m.ntendon:
          mujoco.mju_dense2sparse(
              ten_j,
              d_i.ten_J,
              ten_j_rownnz,
              ten_j_rowadr,
              ten_j_colind,
          )
        result_i.ten_J[:] = ten_j
        continue

      value = getattr(d_i, field.name)

      if field.name in ('nefc', 'ncon'):
        value = {'nefc': nefc, 'ncon': ncon}[field.name]
      elif field.name.endswith('xmat') or field.name == 'ximat':
        value = value.reshape((-1, 9))
      elif field.name == 'efc_J':
        value = value[efc_active]
        if support.is_sparse(m):
          efc_J_rownnz = np.zeros(nefc, dtype=np.int32)
          efc_J_rowadr = np.zeros(nefc, dtype=np.int32)
          efc_J_colind = np.zeros(nj, dtype=np.int32)
          efc_J = np.zeros(nj)
          mujoco.mju_dense2sparse(
              efc_J,
              value,
              efc_J_rownnz,
              efc_J_rowadr,
              efc_J_colind,
          )
          result_i.efc_J_rownnz[:] = efc_J_rownnz
          result_i.efc_J_rowadr[:] = efc_J_rowadr
          result_i.efc_J_colind[:] = efc_J_colind
          value = efc_J
        else:
          value = value.reshape(-1)
      elif field.name.startswith('efc_'):
        value = value[efc_active]
      if field.name == 'qM' and not support.is_sparse(m):
        value = value[dof_i, dof_j]
      elif field.name == 'qLD':
        value = np.zeros(m.nC)
      elif field.name == 'qLDiagInv' and not support.is_sparse(m):
        value = np.ones(m.nv)

      if isinstance(value, np.ndarray) and value.shape:
        result_field = getattr(result_i, field.name)
        if result_field.shape != value.shape:
          raise ValueError(
              f'Input field {field.name} has shape {value.shape}, but output'
              f' has shape {result_field.shape}'
          )
        result_field[:] = value
      else:
        setattr(result_i, field.name, value)

    # map inertia (sparse) to reduced inertia (compressed sparse)
    result_i.M[:] = result_i.qM[m.mapM2M]

    # recalculate qLD and qLDiagInv
    mujoco.mj_factorM(m, result_i)


def get_data(
    m: mujoco.MjModel,
    d: types.Data,
) -> Union[mujoco.MjData, List[mujoco.MjData]]:
  """Gets mjx_mlx.Data, resulting in mujoco.MjData or List[MjData]."""
  qpos = np.array(d.qpos) if isinstance(d.qpos, mx.array) else d.qpos
  batched = len(qpos.shape) > 1
  batch_size = qpos.shape[0] if batched else 1

  if batched:
    result = [mujoco.MjData(m) for _ in range(batch_size)]
  else:
    result = mujoco.MjData(m)

  get_data_into(result, m, d)
  return result


# ---------------------------------------------------------------------------
# State get/set utilities
# ---------------------------------------------------------------------------

_STATE_MAP = {
    mujoco.mjtState.mjSTATE_TIME: 'time',
    mujoco.mjtState.mjSTATE_QPOS: 'qpos',
    mujoco.mjtState.mjSTATE_QVEL: 'qvel',
    mujoco.mjtState.mjSTATE_ACT: 'act',
    mujoco.mjtState.mjSTATE_HISTORY: 'history',
    mujoco.mjtState.mjSTATE_WARMSTART: 'qacc_warmstart',
    mujoco.mjtState.mjSTATE_CTRL: 'ctrl',
    mujoco.mjtState.mjSTATE_QFRC_APPLIED: 'qfrc_applied',
    mujoco.mjtState.mjSTATE_XFRC_APPLIED: 'xfrc_applied',
    mujoco.mjtState.mjSTATE_EQ_ACTIVE: 'eq_active',
    mujoco.mjtState.mjSTATE_MOCAP_POS: 'mocap_pos',
    mujoco.mjtState.mjSTATE_MOCAP_QUAT: 'mocap_quat',
    mujoco.mjtState.mjSTATE_USERDATA: 'userdata',
    mujoco.mjtState.mjSTATE_PLUGIN: 'plugin_state',
}


def _state_elem_size(m: types.Model, state_enum: mujoco.mjtState) -> int:
  """Returns the size of a state component."""
  if state_enum not in _STATE_MAP:
    raise ValueError(f'Invalid state element {state_enum}')
  name = _STATE_MAP[state_enum]
  if name == 'time':
    return 1
  if name in (
      'qpos',
      'qvel',
      'act',
      'history',
      'qacc_warmstart',
      'ctrl',
      'qfrc_applied',
      'eq_active',
      'mocap_pos',
      'mocap_quat',
      'userdata',
      'plugin_state',
  ):
    val = getattr(
        m,
        {
            'qpos': 'nq',
            'qvel': 'nv',
            'act': 'na',
            'history': 'nhistory',
            'qacc_warmstart': 'nv',
            'ctrl': 'nu',
            'qfrc_applied': 'nv',
            'eq_active': 'neq',
            'mocap_pos': 'nmocap',
            'mocap_quat': 'nmocap',
            'userdata': 'nuserdata',
            'plugin_state': 'npluginstate',
        }[name],
    )
    if name == 'mocap_pos':
      val *= 3
    if name == 'mocap_quat':
      val *= 4
    return val
  if name == 'xfrc_applied':
    return 6 * m.nbody

  raise NotImplementedError(f'state component {name} not implemented')


def state_size(m: types.Model, spec: Union[int, mujoco.mjtState]) -> int:
  """Returns the size of a state vector for a given spec.

  Args:
    m: model describing the simulation
    spec: int bitmask or mjtState enum specifying which state components to
      include

  Returns:
    size of the state vector
  """
  size = 0
  spec_int = int(spec)
  for i in range(mujoco.mjtState.mjNSTATE.value):
    element = mujoco.mjtState(1 << i)
    if element & spec_int:
      size += _state_elem_size(m, element)
  return size


def get_state(
    m: types.Model, d: types.Data, spec: Union[int, mujoco.mjtState]
) -> mx.array:
  """Gets state from mjx_mlx.Data. Equivalent to `mujoco.mj_getState`.

  Args:
    m: model describing the simulation
    d: data for the simulation
    spec: int bitmask or mjtState enum specifying which state components to
      include

  Returns:
    a flat MLX array of state values
  """
  spec_int = int(spec)
  if spec_int >= (1 << mujoco.mjtState.mjNSTATE.value):
    raise ValueError(f'Invalid state spec {spec}')

  state = []
  for i in range(mujoco.mjtState.mjNSTATE.value):
    element = mujoco.mjtState(1 << i)
    if element & spec_int:
      if element not in _STATE_MAP:
        raise ValueError(f'Invalid state element {element}')
      name = _STATE_MAP[element]
      value = getattr(d, name)
      if not isinstance(value, mx.array):
        value = mx.array(value)
      if element == mujoco.mjtState.mjSTATE_EQ_ACTIVE:
        value = value.astype(mx.float32)
      state.append(mx.reshape(value, (-1,)))

  return mx.concatenate(state) if state else mx.array([])


def set_state(
    m: types.Model,
    d: types.Data,
    state: mx.array,
    spec: Union[int, mujoco.mjtState],
) -> types.Data:
  """Sets state in mjx_mlx.Data. Equivalent to `mujoco.mj_setState`.

  Args:
    m: model describing the simulation
    d: data for the simulation
    state: a flat MLX array of state values
    spec: int bitmask or mjtState enum specifying which state components to
      include

  Returns:
    data with state set to provided values
  """
  spec_int = int(spec)
  if spec_int >= (1 << mujoco.mjtState.mjNSTATE.value):
    raise ValueError(f'Invalid state spec {spec}')

  expected_size = state_size(m, spec)
  if state.size != expected_size:
    raise ValueError(
        f'state has size {state.size} but expected {expected_size}'
    )

  updates = {}
  offset = 0
  for i in range(mujoco.mjtState.mjNSTATE.value):
    element = mujoco.mjtState(1 << i)
    if element & spec_int:
      if element not in _STATE_MAP:
        raise ValueError(f'Invalid state element {element}')
      name = _STATE_MAP[element]
      size = _state_elem_size(m, element)
      value = state[offset : offset + size]
      if name == 'time':
        value = value[0]
      else:
        orig = getattr(d, name)
        orig_shape = orig.shape if isinstance(orig, mx.array) else np.array(orig).shape
        value = mx.reshape(value, orig_shape)
      if element == mujoco.mjtState.mjSTATE_EQ_ACTIVE:
        value = value.astype(mx.bool_)
      updates[name] = value
      offset += size

  return d.replace(**updates)
