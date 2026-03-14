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
"""Base types used in MJX-MLX (Apple MLX port)."""

import dataclasses
import enum
from typing import Any, Tuple, Union
import warnings

import mlx.core as mx
import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Helper base class: replaces flax/JAX PyTreeNode with a plain dataclass.
# ---------------------------------------------------------------------------
import copy
from typing import Dict, Optional, Sequence as _Seq


def _tree_replace_impl(base, attr, val):
  """Sets attributes in a dataclass with values (dot-path traversal)."""
  if not attr:
    return base
  # special case for List attribute
  if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
    lst = copy.copy(getattr(base, attr[0]))
    for i, g in enumerate(lst):
      if not hasattr(g, attr[1]):
        continue
      v = val if not hasattr(val, '__iter__') else val[i]
      lst[i] = _tree_replace_impl(g, attr[1:], v)
    return dataclasses.replace(base, **{attr[0]: lst})
  if len(attr) == 1:
    return dataclasses.replace(base, **{attr[0]: val})
  return dataclasses.replace(
      base,
      **{attr[0]: _tree_replace_impl(getattr(base, attr[0]), attr[1:], val)}
  )


@dataclasses.dataclass
class MLXNode:
  """Plain dataclass base that replaces JAX PyTreeNode for MLX."""

  @classmethod
  def fields(cls):
    """Return dataclass fields (compatibility with flax.struct API)."""
    return dataclasses.fields(cls)

  def replace(self, **overrides):
    """Return a copy with specified fields replaced."""
    return dataclasses.replace(self, **overrides)

  def tree_replace(self, params: Dict[str, Any]) -> 'MLXNode':
    """Replace nested fields using dot-separated keys."""
    new = self
    for k, v in params.items():
      new = _tree_replace_impl(new, k.split('.'), v)
    return new

  @property
  def _impl_or_self(self):
    """Return _impl if it exists and is not None, else self."""
    impl = object.__getattribute__(self, '_impl') if '_impl' in {f.name for f in dataclasses.fields(self)} else None
    return impl if impl is not None else self


class DisableBit(enum.IntFlag):
  """Disable default feature bitflags.

  Attributes:
    CONSTRAINT:   entire constraint solver
    EQUALITY:     equality constraints
    FRICTIONLOSS: joint and tendon frictionloss constraints
    LIMIT:        joint and tendon limit constraints
    CONTACT:      contact constraints
    SPRING:       passive spring forces
    DAMPER:       passive damper forces
    GRAVITY:      gravitational forces
    CLAMPCTRL:    clamp control to specified range
    WARMSTART:    warmstart constraint solver
    ACTUATION:    apply actuation forces
    REFSAFE:      integrator safety: make ref[0]>=2*timestep
    SENSOR:       sensors
  """

  CONSTRAINT = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
  EQUALITY = mujoco.mjtDisableBit.mjDSBL_EQUALITY
  FRICTIONLOSS = mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
  LIMIT = mujoco.mjtDisableBit.mjDSBL_LIMIT
  CONTACT = mujoco.mjtDisableBit.mjDSBL_CONTACT
  SPRING = mujoco.mjtDisableBit.mjDSBL_SPRING
  DAMPER = mujoco.mjtDisableBit.mjDSBL_DAMPER
  GRAVITY = mujoco.mjtDisableBit.mjDSBL_GRAVITY
  CLAMPCTRL = mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
  WARMSTART = mujoco.mjtDisableBit.mjDSBL_WARMSTART
  ACTUATION = mujoco.mjtDisableBit.mjDSBL_ACTUATION
  REFSAFE = mujoco.mjtDisableBit.mjDSBL_REFSAFE
  SENSOR = mujoco.mjtDisableBit.mjDSBL_SENSOR
  EULERDAMP = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  FILTERPARENT = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
  # unsupported: MIDPHASE


class EnableBit(enum.IntFlag):
  """Enable optional feature bitflags.

  Attributes:
    INVDISCRETE: discrete-time inverse dynamics
  """

  INVDISCRETE = mujoco.mjtEnableBit.mjENBL_INVDISCRETE
  # unsupported: OVERRIDE, ENERGY, FWDINV, ISLAND
  # required by the C implementation only, ignored otherwise: MULTICCD
  MULTICCD = mujoco.mjtEnableBit.mjENBL_MULTICCD
  SLEEP = mujoco.mjtEnableBit.mjENBL_SLEEP


class JointType(enum.IntEnum):
  """Type of degree of freedom.

  Attributes:
    FREE:  global position and orientation (quat)       (7,)
    BALL:  orientation (quat) relative to parent        (4,)
    SLIDE: sliding distance along body-fixed axis       (1,)
    HINGE: rotation angle (rad) around body-fixed axis  (1,)
  """

  FREE = mujoco.mjtJoint.mjJNT_FREE
  BALL = mujoco.mjtJoint.mjJNT_BALL
  SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
  HINGE = mujoco.mjtJoint.mjJNT_HINGE

  def dof_width(self) -> int:
    return {0: 6, 1: 3, 2: 1, 3: 1}[self.value]

  def qpos_width(self) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[self.value]


class IntegratorType(enum.IntEnum):
  """Integrator mode.

  Attributes:
    EULER: semi-implicit Euler
    RK4: 4th-order Runge Kutta
    IMPLICITFAST: implicit in velocity, no rne derivative
  """

  EULER = mujoco.mjtIntegrator.mjINT_EULER
  RK4 = mujoco.mjtIntegrator.mjINT_RK4
  IMPLICITFAST = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  # unsupported: IMPLICIT


class GeomType(enum.IntEnum):
  """Type of geometry.

  Attributes:
    PLANE: plane
    HFIELD: height field
    SPHERE: sphere
    CAPSULE: capsule
    ELLIPSOID: ellipsoid
    CYLINDER: cylinder
    BOX: box
    MESH: mesh
    SDF: signed distance field
  """

  PLANE = mujoco.mjtGeom.mjGEOM_PLANE
  HFIELD = mujoco.mjtGeom.mjGEOM_HFIELD
  SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
  CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
  ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
  CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
  BOX = mujoco.mjtGeom.mjGEOM_BOX
  MESH = mujoco.mjtGeom.mjGEOM_MESH
  # unsupported: NGEOMTYPES, ARROW*, LINE, SKIN, LABEL, NONE


@dataclasses.dataclass
class ConvexMesh(MLXNode):
  """Geom properties for convex meshes.

  Attributes:
    vert: vertices of the convex mesh
    face: faces of the convex mesh
    face_normal: normal vectors for the faces
    edge: edge indexes for all edges in the convex mesh
    edge_face_normal: indexes for face normals adjacent to edges in `edge`
  """

  vert: mx.array
  face: mx.array
  face_normal: mx.array
  edge: mx.array
  edge_face_normal: mx.array


class ConeType(enum.IntEnum):
  """Type of friction cone.

  Attributes:
    PYRAMIDAL: pyramidal
    ELLIPTIC: elliptic
  """

  PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL
  ELLIPTIC = mujoco.mjtCone.mjCONE_ELLIPTIC


class JacobianType(enum.IntEnum):
  """Type of constraint Jacobian.

  Attributes:
    DENSE: dense
    SPARSE: sparse
    AUTO: sparse if nv>60, dense otherwise
  """

  DENSE = mujoco.mjtJacobian.mjJAC_DENSE
  SPARSE = mujoco.mjtJacobian.mjJAC_SPARSE
  AUTO = mujoco.mjtJacobian.mjJAC_AUTO


class SolverType(enum.IntEnum):
  """Constraint solver algorithm.

  Attributes:
    CG: Conjugate gradient (primal)
    NEWTON: Newton (primal)
  """

  # unsupported: PGS
  CG = mujoco.mjtSolver.mjSOL_CG
  NEWTON = mujoco.mjtSolver.mjSOL_NEWTON


class EqType(enum.IntEnum):
  """Type of equality constraint.

  Attributes:
    CONNECT: connect two bodies at a point (ball joint)
    WELD: fix relative position and orientation of two bodies
    JOINT: couple the values of two scalar joints with cubic
    TENDON: couple the lengths of two tendons with cubic
  """

  CONNECT = mujoco.mjtEq.mjEQ_CONNECT
  WELD = mujoco.mjtEq.mjEQ_WELD
  JOINT = mujoco.mjtEq.mjEQ_JOINT
  TENDON = mujoco.mjtEq.mjEQ_TENDON
  # unsupported: DISTANCE


class WrapType(enum.IntEnum):
  """Type of tendon wrap object.

  Attributes:
    JOINT: constant moment arm
    PULLEY: pulley used to split tendon
    SITE: pass through site
    SPHERE: wrap around sphere
    CYLINDER: wrap around (infinite) cylinder
  """

  JOINT = mujoco.mjtWrap.mjWRAP_JOINT
  PULLEY = mujoco.mjtWrap.mjWRAP_PULLEY
  SITE = mujoco.mjtWrap.mjWRAP_SITE
  SPHERE = mujoco.mjtWrap.mjWRAP_SPHERE
  CYLINDER = mujoco.mjtWrap.mjWRAP_CYLINDER


class TrnType(enum.IntEnum):
  """Type of actuator transmission.

  Attributes:
    JOINT: force on joint
    JOINTINPARENT: force on joint, expressed in parent frame
    TENDON: force on tendon
    SITE: force on site
  """

  JOINT = mujoco.mjtTrn.mjTRN_JOINT
  JOINTINPARENT = mujoco.mjtTrn.mjTRN_JOINTINPARENT
  SITE = mujoco.mjtTrn.mjTRN_SITE
  TENDON = mujoco.mjtTrn.mjTRN_TENDON
  # unsupported: SLIDERCRANK, BODY


class DynType(enum.IntEnum):
  """Type of actuator dynamics.

  Attributes:
    NONE: no internal dynamics; ctrl specifies force
    INTEGRATOR: integrator: da/dt = u
    FILTER: linear filter: da/dt = (u-a) / tau
    FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
    MUSCLE: piece-wise linear filter with two time constants
  """

  NONE = mujoco.mjtDyn.mjDYN_NONE
  INTEGRATOR = mujoco.mjtDyn.mjDYN_INTEGRATOR
  FILTER = mujoco.mjtDyn.mjDYN_FILTER
  FILTEREXACT = mujoco.mjtDyn.mjDYN_FILTEREXACT
  MUSCLE = mujoco.mjtDyn.mjDYN_MUSCLE
  # unsupported: USER


class GainType(enum.IntEnum):
  """Type of actuator gain.

  Attributes:
    FIXED: fixed gain
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle FLV curve computed by muscle_gain
  """

  FIXED = mujoco.mjtGain.mjGAIN_FIXED
  AFFINE = mujoco.mjtGain.mjGAIN_AFFINE
  MUSCLE = mujoco.mjtGain.mjGAIN_MUSCLE
  # unsupported: USER


class BiasType(enum.IntEnum):
  """Type of actuator bias.

  Attributes:
    NONE: no bias
    AFFINE: const + kp*length + kv*velocity
    MUSCLE: muscle passive force computed by muscle_bias
  """

  NONE = mujoco.mjtBias.mjBIAS_NONE
  AFFINE = mujoco.mjtBias.mjBIAS_AFFINE
  MUSCLE = mujoco.mjtBias.mjBIAS_MUSCLE
  # unsupported: USER


class ConstraintType(enum.IntEnum):
  """Type of constraint.

  Attributes:
    EQUALITY: equality constraint
    LIMIT_JOINT: joint limit
    LIMIT_TENDON: tendon limit
    CONTACT_FRICTIONLESS: frictionless contact
    CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
  """

  EQUALITY = mujoco.mjtConstraint.mjCNSTR_EQUALITY
  FRICTION_DOF = mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
  FRICTION_TENDON = mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON
  LIMIT_JOINT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
  LIMIT_TENDON = mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
  CONTACT_FRICTIONLESS = mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS
  CONTACT_PYRAMIDAL = mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
  CONTACT_ELLIPTIC = mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC


class CamLightType(enum.IntEnum):
  """Type of camera light.

  Attributes:
    FIXED: pos and rot fixed in body
    TRACK: pos tracks body, rot fixed in global
    TRACKCOM: pos tracks subtree com, rot fixed in body
    TARGETBODY: pos fixed in body, rot tracks target body
    TARGETBODYCOM: pos fixed in body, rot tracks target subtree com
  """

  FIXED = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
  TRACK = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
  TRACKCOM = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
  TARGETBODY = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  TARGETBODYCOM = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM


class SensorType(enum.IntEnum):
  """Type of sensor.

  Attributes:
    MAGNETOMETER: magnetometer
    CAMPROJECTION: camera projection
    RANGEFINDER: rangefinder
    JOINTPOS: joint position
    TENDONPOS: scalar tendon position
    ACTUATORPOS: actuator position
    BALLQUAT: ball joint orientation
    FRAMEPOS: frame position
    FRAMEXAXIS: frame x-axis
    FRAMEYAXIS: frame y-axis
    FRAMEZAXIS: frame z-axis
    FRAMEQUAT: frame orientation, represented as quaternion
    SUBTREECOM: subtree centor of mass
    CLOCK: simulation time
    VELOCIMETER: 3D linear velocity, in local frame
    GYRO: 3D angular velocity, in local frame
    JOINTVEL: joint velocity
    TENDONVEL: scalar tendon velocity
    ACTUATORVEL: actuator velocity
    BALLANGVEL: ball joint angular velocity
    FRAMELINVEL: 3D linear velocity
    FRAMEANGVEL: 3D angular velocity
    SUBTREELINVEL: subtree linear velocity
    SUBTREEANGMOM: subtree angular momentum
    TOUCH: scalar contact normal forces summed over the sensor zone
    CONTACT: contacts which occurred during the simulation
    ACCELEROMETER: accelerometer
    FORCE: force
    TORQUE: torque
    ACTUATORFRC: scalar actuator force
    JOINTACTFRC: scalar actuator force, measured at the joint
    TENDONACTFRC: scalar actuator force, measured at the tendon
    FRAMELINACC: 3D linear acceleration
    FRAMEANGACC: 3D angular acceleration
  """

  MAGNETOMETER = mujoco.mjtSensor.mjSENS_MAGNETOMETER
  CAMPROJECTION = mujoco.mjtSensor.mjSENS_CAMPROJECTION
  RANGEFINDER = mujoco.mjtSensor.mjSENS_RANGEFINDER
  JOINTPOS = mujoco.mjtSensor.mjSENS_JOINTPOS
  TENDONPOS = mujoco.mjtSensor.mjSENS_TENDONPOS
  ACTUATORPOS = mujoco.mjtSensor.mjSENS_ACTUATORPOS
  BALLQUAT = mujoco.mjtSensor.mjSENS_BALLQUAT
  FRAMEPOS = mujoco.mjtSensor.mjSENS_FRAMEPOS
  FRAMEXAXIS = mujoco.mjtSensor.mjSENS_FRAMEXAXIS
  FRAMEYAXIS = mujoco.mjtSensor.mjSENS_FRAMEYAXIS
  FRAMEZAXIS = mujoco.mjtSensor.mjSENS_FRAMEZAXIS
  FRAMEQUAT = mujoco.mjtSensor.mjSENS_FRAMEQUAT
  SUBTREECOM = mujoco.mjtSensor.mjSENS_SUBTREECOM
  CLOCK = mujoco.mjtSensor.mjSENS_CLOCK
  VELOCIMETER = mujoco.mjtSensor.mjSENS_VELOCIMETER
  GYRO = mujoco.mjtSensor.mjSENS_GYRO
  JOINTVEL = mujoco.mjtSensor.mjSENS_JOINTVEL
  TENDONVEL = mujoco.mjtSensor.mjSENS_TENDONVEL
  ACTUATORVEL = mujoco.mjtSensor.mjSENS_ACTUATORVEL
  BALLANGVEL = mujoco.mjtSensor.mjSENS_BALLANGVEL
  FRAMELINVEL = mujoco.mjtSensor.mjSENS_FRAMELINVEL
  FRAMEANGVEL = mujoco.mjtSensor.mjSENS_FRAMEANGVEL
  SUBTREELINVEL = mujoco.mjtSensor.mjSENS_SUBTREELINVEL
  SUBTREEANGMOM = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
  TOUCH = mujoco.mjtSensor.mjSENS_TOUCH
  CONTACT = mujoco.mjtSensor.mjSENS_CONTACT
  ACCELEROMETER = mujoco.mjtSensor.mjSENS_ACCELEROMETER
  FORCE = mujoco.mjtSensor.mjSENS_FORCE
  TORQUE = mujoco.mjtSensor.mjSENS_TORQUE
  ACTUATORFRC = mujoco.mjtSensor.mjSENS_ACTUATORFRC
  JOINTACTFRC = mujoco.mjtSensor.mjSENS_JOINTACTFRC
  TENDONACTFRC = mujoco.mjtSensor.mjSENS_TENDONACTFRC
  FRAMELINACC = mujoco.mjtSensor.mjSENS_FRAMELINACC
  FRAMEANGACC = mujoco.mjtSensor.mjSENS_FRAMEANGACC


class ObjType(enum.IntEnum):
  """Type of object.

  Attributes:
    UNKNOWN: unknown object type
    BODY: body
    XBODY: body, used to access regular frame instead of i-frame
    GEOM: geom
    SITE: site
    CAMERA: camera
  """

  UNKNOWN = mujoco.mjtObj.mjOBJ_UNKNOWN
  BODY = mujoco.mjtObj.mjOBJ_BODY
  XBODY = mujoco.mjtObj.mjOBJ_XBODY
  GEOM = mujoco.mjtObj.mjOBJ_GEOM
  SITE = mujoco.mjtObj.mjOBJ_SITE
  CAMERA = mujoco.mjtObj.mjOBJ_CAMERA


@dataclasses.dataclass
class Statistic(MLXNode):
  """Model statistics (in qpos0).

  Attributes:
    meaninertia: mean diagonal inertia
    meanmass: mean body mass (not used)
    meansize: mean body size (not used)
    extent: spatial extent (not used)
    center: center of model (not used)
  """

  meaninertia: mx.array
  meanmass: mx.array
  meansize: mx.array
  extent: mx.array
  center: mx.array


@dataclasses.dataclass
class OptionMLX(MLXNode):
  """MLX-specific option."""

  o_margin: mx.array
  o_solref: mx.array
  o_solimp: mx.array
  o_friction: mx.array
  disableactuator: int
  sdf_initpoints: int
  has_fluid_params: bool


@dataclasses.dataclass
class Option(MLXNode):
  """Physics options."""

  iterations: int
  ls_iterations: int
  tolerance: mx.array
  ls_tolerance: mx.array
  impratio: mx.array
  gravity: mx.array
  density: mx.array
  viscosity: mx.array
  magnetic: mx.array
  wind: mx.array
  jacobian: JacobianType
  cone: ConeType
  disableflags: DisableBit
  enableflags: int
  integrator: IntegratorType
  solver: SolverType
  timestep: mx.array
  _impl: OptionMLX = None


@dataclasses.dataclass
class Contact(MLXNode):
  """Result of collision detection functions.

  Attributes:
    dist: distance between nearest points; neg: penetration
    pos: position of contact point: midpoint between geoms            (3,)
    frame: normal is in [0-2]                                         (9,)
    includemargin: include if dist<includemargin=margin-gap           (1,)
    friction: tangent1, 2, spin, roll1, 2                             (5,)
    solref: constraint solver reference, normal direction             (mjNREF,)
    solreffriction: constraint solver reference, friction directions  (mjNREF,)
    solimp: constraint solver impedance                               (mjNIMP,)
    dim: contact space dimensionality: 1, 3, 4, or 6
    geom1: id of geom 1; deprecated, use geom[0]
    geom2: id of geom 2; deprecated, use geom[1]
    geom: geom ids                                                    (2,)
    efc_address: address in efc; -1: not included
  """  # fmt: skip
  dist: mx.array
  pos: mx.array
  frame: mx.array
  includemargin: mx.array
  friction: mx.array
  solref: mx.array
  solreffriction: mx.array
  solimp: mx.array
  # unsupported: mu, H (calculated locally in solver.py)
  dim: np.ndarray
  geom1: mx.array
  geom2: mx.array
  geom: mx.array
  # unsupported: flex, elem, vert, exclude
  efc_address: np.ndarray


@dataclasses.dataclass
class ModelMLX(MLXNode):
  """MLX-specific model data."""

  dof_hasfrictionloss: np.ndarray
  geom_rbound_hfield: np.ndarray
  mesh_convex: Tuple[ConvexMesh, ...]
  tendon_hasfrictionloss: np.ndarray


@dataclasses.dataclass
class Model(MLXNode):
  """Static model of the scene that remains unchanged with each physics step.

  Attributes:
    nq: number of generalized coordinates
    nv: number of degrees of freedom
    nu: number of actuators/controls
    na: number of activation states
    nbody: number of bodies
    njnt: number of joints
    ngeom: number of geoms
    nsite: number of sites
    ncam: number of cameras
    nlight: number of lights
    nmesh: number of meshes
    nmeshvert: number of vertices for all meshes
    nmeshnormal: number of normals in all meshes
    nmeshtexcoord: number of texcoords in all meshes
    nmeshface: number of faces for all meshes
    nmeshgraph: number of ints in mesh auxiliary data
    nmeshpoly: number of polygons in all meshes
    nmeshpolyvert: number of vertices in all polygons
    nmeshpolymap: number of polygons in vertex map
    nhfield: number of heightfields
    nhfielddata: size of elevation data
    ntex: number of textures
    ntexdata: size of texture data
    nmat: number of materials
    npair: number of predefined geom pairs
    nexclude: number of excluded geom pairs
    neq: number of equality constraints
    ntendon: number of tendons
    nwrap: number of wrap objects in all tendon paths
    nsensor: number of sensors
    nnumeric: number of numeric custom fields
    ntuple: number of tuple custom fields
    nkey: number of keyframes
    nmocap: number of mocap bodies
    nM: number of non-zeros in sparse inertia matrix
    nB: number of non-zeros in B matrix
    nC: number of non-zeros in C matrix
    nD: number of non-zeros in D matrix
    nJmom: number of non-zeros in Jacobian momentum matrix
    nJten: number of non-zeros in sparse tendon Jacobian
    ngravcomp: number of bodies with nonzero gravcomp
    nuserdata: number of elements in userdata
    nsensordata: number of elements in sensor data vector
    npluginstate: number of plugin state values
    nhistory: number of history buffer elements
    opt: physics options
    stat: model statistics
    qpos0: qpos values at default pose
    qpos_spring: reference pose for springs
  """

  nq: int
  nv: int
  nu: int
  na: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  ncam: int
  nlight: int
  nmesh: int
  nmeshvert: int
  nmeshnormal: int
  nmeshtexcoord: int
  nmeshface: int
  nmeshgraph: int
  nmeshpoly: int
  nmeshpolyvert: int
  nmeshpolymap: int
  nhfield: int
  nhfielddata: int
  ntex: int
  ntexdata: int
  nmat: int
  npair: int
  nexclude: int
  neq: int
  ntendon: int
  nwrap: int
  nsensor: int
  nnumeric: int
  ntuple: int
  nkey: int
  nmocap: int
  nM: int
  nB: int
  nC: int
  nD: int
  nJmom: int
  nJten: int
  ngravcomp: int
  nuserdata: int
  nsensordata: int
  npluginstate: int
  nhistory: int
  opt: Option
  stat: Statistic
  qpos0: mx.array
  qpos_spring: mx.array
  body_parentid: np.ndarray
  body_mocapid: np.ndarray
  body_rootid: np.ndarray
  body_weldid: np.ndarray
  body_jntnum: np.ndarray
  body_jntadr: np.ndarray
  body_sameframe: np.ndarray
  body_dofnum: np.ndarray
  body_dofadr: np.ndarray
  body_treeid: np.ndarray
  body_geomnum: np.ndarray
  body_geomadr: np.ndarray
  body_simple: np.ndarray
  body_pos: mx.array
  body_quat: mx.array
  body_ipos: mx.array
  body_iquat: mx.array
  body_mass: mx.array
  body_subtreemass: mx.array
  body_inertia: mx.array
  body_gravcomp: mx.array
  body_margin: np.ndarray
  body_contype: np.ndarray
  body_conaffinity: np.ndarray
  body_invweight0: mx.array
  jnt_type: np.ndarray
  jnt_qposadr: np.ndarray
  jnt_dofadr: np.ndarray
  jnt_bodyid: np.ndarray
  jnt_limited: np.ndarray
  jnt_actfrclimited: np.ndarray
  jnt_actgravcomp: np.ndarray
  jnt_solref: mx.array
  jnt_solimp: mx.array
  jnt_pos: mx.array
  jnt_axis: mx.array
  jnt_stiffness: mx.array
  jnt_range: mx.array
  jnt_actfrcrange: mx.array
  jnt_margin: mx.array
  dof_bodyid: np.ndarray
  dof_jntid: np.ndarray
  dof_parentid: np.ndarray
  dof_treeid: np.ndarray
  dof_Madr: np.ndarray
  dof_simplenum: np.ndarray
  dof_solref: mx.array
  dof_solimp: mx.array
  dof_frictionloss: mx.array
  dof_armature: mx.array
  dof_damping: mx.array
  dof_invweight0: mx.array
  dof_M0: mx.array
  geom_type: np.ndarray
  geom_contype: np.ndarray
  geom_conaffinity: np.ndarray
  geom_condim: np.ndarray
  geom_bodyid: np.ndarray
  geom_sameframe: np.ndarray
  geom_dataid: np.ndarray
  geom_group: np.ndarray
  geom_matid: mx.array
  geom_priority: np.ndarray
  geom_solmix: mx.array
  geom_solref: mx.array
  geom_solimp: mx.array
  geom_size: mx.array
  geom_aabb: mx.array
  geom_rbound: mx.array
  geom_pos: mx.array
  geom_quat: mx.array
  geom_friction: mx.array
  geom_margin: mx.array
  geom_gap: mx.array
  geom_fluid: np.ndarray
  geom_rgba: mx.array
  site_type: np.ndarray
  site_bodyid: np.ndarray
  site_sameframe: np.ndarray
  site_size: np.ndarray
  site_pos: mx.array
  site_quat: mx.array
  cam_mode: np.ndarray
  cam_bodyid: np.ndarray
  cam_targetbodyid: np.ndarray
  cam_pos: mx.array
  cam_quat: mx.array
  cam_poscom0: mx.array
  cam_pos0: mx.array
  cam_mat0: mx.array
  cam_fovy: mx.array
  cam_resolution: np.ndarray
  cam_sensorsize: np.ndarray
  cam_intrinsic: mx.array
  light_mode: np.ndarray
  light_type: mx.array
  light_castshadow: mx.array
  light_pos: mx.array
  light_dir: mx.array
  light_poscom0: mx.array
  light_pos0: mx.array
  light_dir0: mx.array
  light_cutoff: mx.array
  mesh_vertadr: np.ndarray
  mesh_vertnum: np.ndarray
  mesh_faceadr: np.ndarray
  mesh_bvhadr: np.ndarray
  mesh_bvhnum: np.ndarray
  mesh_octadr: np.ndarray
  mesh_octnum: np.ndarray
  mesh_normaladr: np.ndarray
  mesh_normalnum: np.ndarray
  mesh_graphadr: np.ndarray
  mesh_vert: np.ndarray
  mesh_normal: np.ndarray
  mesh_face: np.ndarray
  mesh_graph: np.ndarray
  mesh_pos: np.ndarray
  mesh_quat: np.ndarray
  mesh_texcoordadr: np.ndarray
  mesh_texcoordnum: np.ndarray
  mesh_texcoord: np.ndarray
  hfield_size: np.ndarray
  hfield_nrow: np.ndarray
  hfield_ncol: np.ndarray
  hfield_adr: np.ndarray
  hfield_data: mx.array
  tex_type: np.ndarray
  tex_height: np.ndarray
  tex_width: np.ndarray
  tex_nchannel: np.ndarray
  tex_adr: np.ndarray
  tex_data: np.ndarray
  mat_rgba: mx.array
  mat_texid: mx.array
  pair_dim: np.ndarray
  pair_geom1: np.ndarray
  pair_geom2: np.ndarray
  pair_signature: np.ndarray
  pair_solref: mx.array
  pair_solreffriction: mx.array
  pair_solimp: mx.array
  pair_margin: mx.array
  pair_gap: mx.array
  pair_friction: mx.array
  exclude_signature: np.ndarray
  eq_type: np.ndarray
  eq_obj1id: np.ndarray
  eq_obj2id: np.ndarray
  eq_objtype: np.ndarray
  eq_active0: np.ndarray
  eq_solref: mx.array
  eq_solimp: mx.array
  eq_data: mx.array
  tendon_adr: np.ndarray
  tendon_num: np.ndarray
  tendon_limited: np.ndarray
  tendon_actfrclimited: np.ndarray
  tendon_solref_lim: mx.array
  tendon_solimp_lim: mx.array
  tendon_solref_fri: mx.array
  tendon_solimp_fri: mx.array
  tendon_range: mx.array
  tendon_actfrcrange: mx.array
  tendon_margin: mx.array
  tendon_stiffness: mx.array
  tendon_damping: mx.array
  tendon_armature: mx.array
  tendon_frictionloss: mx.array
  tendon_lengthspring: mx.array
  tendon_length0: mx.array
  tendon_invweight0: mx.array
  wrap_type: np.ndarray
  wrap_objid: np.ndarray
  wrap_prm: np.ndarray
  actuator_trntype: np.ndarray
  actuator_dyntype: np.ndarray
  actuator_gaintype: np.ndarray
  actuator_biastype: np.ndarray
  actuator_trnid: np.ndarray
  actuator_actadr: np.ndarray
  actuator_actnum: np.ndarray
  actuator_group: np.ndarray
  actuator_ctrllimited: np.ndarray
  actuator_forcelimited: np.ndarray
  actuator_actlimited: np.ndarray
  actuator_dynprm: mx.array
  actuator_gainprm: mx.array
  actuator_biasprm: mx.array
  actuator_actearly: np.ndarray
  actuator_ctrlrange: mx.array
  actuator_forcerange: mx.array
  actuator_actrange: mx.array
  actuator_gear: mx.array
  actuator_cranklength: np.ndarray
  actuator_acc0: mx.array
  actuator_lengthrange: np.ndarray
  sensor_type: np.ndarray
  sensor_datatype: np.ndarray
  sensor_needstage: np.ndarray
  sensor_objtype: np.ndarray
  sensor_objid: np.ndarray
  sensor_reftype: np.ndarray
  sensor_refid: np.ndarray
  sensor_intprm: np.ndarray
  sensor_dim: np.ndarray
  sensor_adr: np.ndarray
  sensor_cutoff: np.ndarray
  numeric_adr: np.ndarray
  numeric_data: np.ndarray
  tuple_adr: np.ndarray
  tuple_size: np.ndarray
  tuple_objtype: np.ndarray
  tuple_objid: np.ndarray
  tuple_objprm: np.ndarray
  key_time: np.ndarray
  key_qpos: np.ndarray
  key_qvel: np.ndarray
  key_act: np.ndarray
  key_mpos: np.ndarray
  key_mquat: np.ndarray
  key_ctrl: np.ndarray
  name_bodyadr: np.ndarray
  name_jntadr: np.ndarray
  name_geomadr: np.ndarray
  name_siteadr: np.ndarray
  name_camadr: np.ndarray
  name_meshadr: np.ndarray
  name_hfieldadr: np.ndarray
  name_pairadr: np.ndarray
  name_eqadr: np.ndarray
  name_tendonadr: np.ndarray
  name_actuatoradr: np.ndarray
  name_sensoradr: np.ndarray
  name_numericadr: np.ndarray
  name_tupleadr: np.ndarray
  name_keyadr: np.ndarray
  names: bytes
  signature: np.uint64
  _sizes: mx.array
  _impl: ModelMLX = None


@dataclasses.dataclass
class DataMLX(MLXNode):
  """MLX-specific data."""

  ne: int
  nf: int
  nl: int
  nefc: int
  ncon: int
  solver_niter: mx.array
  cinert: mx.array
  ten_wrapadr: mx.array
  ten_wrapnum: mx.array
  ten_J: mx.array
  wrap_obj: mx.array
  wrap_xpos: mx.array
  actuator_moment: mx.array
  crb: mx.array
  qM: mx.array
  M: mx.array
  qLD: mx.array
  qLDiagInv: mx.array
  ten_velocity: mx.array
  actuator_velocity: mx.array

  cacc: mx.array
  cfrc_int: mx.array
  cfrc_ext: mx.array
  subtree_linvel: mx.array
  subtree_angmom: mx.array
  # dynamically sized data which are made static
  contact: Contact
  efc_type: mx.array
  efc_J: mx.array
  efc_pos: mx.array
  efc_margin: mx.array
  efc_frictionloss: mx.array
  efc_D: mx.array
  efc_aref: mx.array
  efc_force: mx.array


@dataclasses.dataclass
class Data(MLXNode):
  """Dynamic state that updates each step.

  Attributes:
    time: simulation time
    qpos: position
    qvel: velocity
    act: actuator activation
    history: actuator history buffer
    qacc_warmstart: warm start for solver
    plugin_state: plugin state values
    ctrl: control input
    qfrc_applied: applied generalized force
    xfrc_applied: applied Cartesian force/torque
    eq_active: enable/disable equality constraints
    mocap_pos: positions of mocap bodies
    mocap_quat: orientations of mocap bodies
    qacc: acceleration
    act_dot: time-derivative of actuator activation
    userdata: user data
    sensordata: sensor data output
    xpos: Cartesian position of body frame
    xquat: Cartesian orientation of body frame
    xmat: rotation matrix of body frame
    xipos: Cartesian position of body com
    ximat: rotation matrix of body inertia
    xanchor: Cartesian position of joint anchor
    xaxis: Cartesian joint axis
    ten_length: tendon lengths
    geom_xpos: Cartesian position of geoms
    geom_xmat: rotation matrix of geoms
    site_xpos: Cartesian position of sites
    site_xmat: rotation matrix of sites
    cam_xpos: camera positions
    cam_xmat: camera rotation matrices
    subtree_com: com of each subtree
    cvel: center of mass based velocity
    cdof: center of mass based jacobian
    cdof_dot: time-derivative of cdof
    qfrc_bias: C(qpos,qvel)
    qfrc_gravcomp: gravity compensation term
    qfrc_fluid: fluid drag and buoyancy forces
    qfrc_passive: passive force
    qfrc_actuator: actuator force
    actuator_force: actuator force in actuation space
    actuator_length: actuator lengths
    qfrc_smooth: smooth dynamics force
    qacc_smooth: acceleration without constraints
    qfrc_constraint: constraint force
    qfrc_inverse: net external force for inverse dynamics
  """

  # global properties:
  time: mx.array
  # state:
  qpos: mx.array
  qvel: mx.array
  act: mx.array
  history: mx.array
  qacc_warmstart: mx.array
  plugin_state: mx.array
  # control:
  ctrl: mx.array
  qfrc_applied: mx.array
  xfrc_applied: mx.array
  eq_active: mx.array
  # mocap data:
  mocap_pos: mx.array
  mocap_quat: mx.array
  # dynamics:
  qacc: mx.array
  act_dot: mx.array
  # user data:
  userdata: mx.array
  sensordata: mx.array
  # position dependent:
  xpos: mx.array
  xquat: mx.array
  xmat: mx.array
  xipos: mx.array
  ximat: mx.array
  xanchor: mx.array
  xaxis: mx.array
  ten_length: mx.array
  geom_xpos: mx.array
  geom_xmat: mx.array
  site_xpos: mx.array
  site_xmat: mx.array
  cam_xpos: mx.array
  cam_xmat: mx.array
  subtree_com: mx.array
  cvel: mx.array
  cdof: mx.array
  cdof_dot: mx.array
  qfrc_bias: mx.array
  qfrc_gravcomp: mx.array
  qfrc_fluid: mx.array
  qfrc_passive: mx.array
  qfrc_actuator: mx.array
  actuator_force: mx.array
  actuator_length: mx.array
  qfrc_smooth: mx.array
  qacc_smooth: mx.array
  qfrc_constraint: mx.array
  qfrc_inverse: mx.array
  _impl: DataMLX = None
