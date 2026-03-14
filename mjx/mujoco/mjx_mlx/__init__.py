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
"""Public API for MJX-MLX (Apple MLX port of MuJoCo MJX)."""

import warnings as _warnings

# ---------------------------------------------------------------------------
# Core types  (always available -- these have no heavy compute dependencies)
# ---------------------------------------------------------------------------
# isort: off
try:
    from mujoco.mjx_mlx._src.types import Model
    from mujoco.mjx_mlx._src.types import Data
    from mujoco.mjx_mlx._src.types import Contact
    from mujoco.mjx_mlx._src.types import Option
    from mujoco.mjx_mlx._src.types import Statistic
except ImportError as _e:  # pragma: no cover
    _warnings.warn(f"mjx_mlx: failed to import core types: {_e}")

# Enum types (wildcard re-export mirrors upstream MJX behaviour)
try:
    from mujoco.mjx_mlx._src.types import DisableBit
    from mujoco.mjx_mlx._src.types import EnableBit
    from mujoco.mjx_mlx._src.types import JointType
    from mujoco.mjx_mlx._src.types import IntegratorType
    from mujoco.mjx_mlx._src.types import GeomType
    from mujoco.mjx_mlx._src.types import ConeType
    from mujoco.mjx_mlx._src.types import JacobianType
    from mujoco.mjx_mlx._src.types import SolverType
    from mujoco.mjx_mlx._src.types import EqType
    from mujoco.mjx_mlx._src.types import WrapType
    from mujoco.mjx_mlx._src.types import TrnType
    from mujoco.mjx_mlx._src.types import DynType
    from mujoco.mjx_mlx._src.types import GainType
    from mujoco.mjx_mlx._src.types import BiasType
    from mujoco.mjx_mlx._src.types import ConstraintType
    from mujoco.mjx_mlx._src.types import CamLightType
    from mujoco.mjx_mlx._src.types import SensorType
    from mujoco.mjx_mlx._src.types import ObjType
except ImportError as _e:  # pragma: no cover
    _warnings.warn(f"mjx_mlx: failed to import enum types: {_e}")

# Internal-only dataclass helpers (ConvexMesh, MLXNode, etc.)
try:
    from mujoco.mjx_mlx._src.types import ConvexMesh
    from mujoco.mjx_mlx._src.types import MLXNode
except ImportError:
    pass
# isort: on

# ---------------------------------------------------------------------------
# I/O: model & data conversion
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.io import put_model
    from mujoco.mjx_mlx._src.io import put_data
    from mujoco.mjx_mlx._src.io import make_data
    from mujoco.mjx_mlx._src.io import get_data
    from mujoco.mjx_mlx._src.io import get_data_into
    from mujoco.mjx_mlx._src.io import get_state
    from mujoco.mjx_mlx._src.io import set_state
    from mujoco.mjx_mlx._src.io import state_size
except ImportError as _e:  # pragma: no cover
    _warnings.warn(f"mjx_mlx: failed to import io module: {_e}")

# ---------------------------------------------------------------------------
# Forward dynamics & stepping
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.forward import step
    from mujoco.mjx_mlx._src.forward import forward
    from mujoco.mjx_mlx._src.forward import euler
    from mujoco.mjx_mlx._src.forward import rungekutta4
    from mujoco.mjx_mlx._src.forward import implicit
    from mujoco.mjx_mlx._src.forward import fwd_acceleration
    from mujoco.mjx_mlx._src.forward import fwd_actuation
    from mujoco.mjx_mlx._src.forward import fwd_position
    from mujoco.mjx_mlx._src.forward import fwd_velocity
except ImportError as _e:  # pragma: no cover
    _warnings.warn(f"mjx_mlx: failed to import forward module: {_e}")

# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.collision_driver import collision
except ImportError as _e:  # pragma: no cover
    _warnings.warn(f"mjx_mlx: failed to import collision_driver: {_e}")

try:
    from mujoco.mjx_mlx._src.bvh import refit_bvh
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.constraint import make_constraint
except ImportError as _e:  # pragma: no cover
    _warnings.warn(f"mjx_mlx: failed to import constraint: {_e}")

# ---------------------------------------------------------------------------
# Derivative
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.derivative import deriv_smooth_vel
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Inverse dynamics
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.inverse import inverse
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Passive forces
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.passive import passive
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.ray import ray
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.render import render
except ImportError:
    pass

try:
    from mujoco.mjx_mlx._src.render_util import get_depth
    from mujoco.mjx_mlx._src.render_util import get_rgb
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.sensor import sensor_acc
    from mujoco.mjx_mlx._src.sensor import sensor_pos
    from mujoco.mjx_mlx._src.sensor import sensor_vel
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Smooth dynamics (kinematics, RNE, tendon, etc.)
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.smooth import camlight
    from mujoco.mjx_mlx._src.smooth import com_pos
    from mujoco.mjx_mlx._src.smooth import com_vel
    from mujoco.mjx_mlx._src.smooth import crb
    from mujoco.mjx_mlx._src.smooth import factor_m
    from mujoco.mjx_mlx._src.smooth import kinematics
    from mujoco.mjx_mlx._src.smooth import rne
    from mujoco.mjx_mlx._src.smooth import rne_postconstraint
    from mujoco.mjx_mlx._src.smooth import subtree_vel
    from mujoco.mjx_mlx._src.smooth import tendon
    from mujoco.mjx_mlx._src.smooth import tendon_armature
    from mujoco.mjx_mlx._src.smooth import tendon_bias
    from mujoco.mjx_mlx._src.smooth import transmission
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.solver import solve
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Support utilities
# ---------------------------------------------------------------------------
try:
    from mujoco.mjx_mlx._src.support import apply_ft
    from mujoco.mjx_mlx._src.support import full_m
    from mujoco.mjx_mlx._src.support import id2name
    from mujoco.mjx_mlx._src.support import is_sparse
    from mujoco.mjx_mlx._src.support import jac
    from mujoco.mjx_mlx._src.support import mul_m
    from mujoco.mjx_mlx._src.support import name2id
    from mujoco.mjx_mlx._src.support import xfrc_accumulate
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Convenience aliases (match JAX MJX API surface)
# ---------------------------------------------------------------------------
def device_put(m, d=None):
    """Alias that mirrors ``mjx.device_put``.

    In the MLX backend all arrays already live on the Metal device, so this
    simply calls ``put_model`` / ``put_data`` as appropriate.
    """
    if d is not None:
        return put_model(m), put_data(m, d)
    return put_model(m)
