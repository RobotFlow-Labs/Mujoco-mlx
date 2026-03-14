# Copyright 2026 DeepMind Technologies Limited
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
"""BVH helpers for MJX-MLX."""

from typing import Any
# pylint: disable=g-importing-member
from mujoco.mjx_mlx._src.types import Data
from mujoco.mjx_mlx._src.types import Model
# pylint: enable=g-importing-member


def refit_bvh(m: Model, d: Data, ctx: Any):
  """Refit the scene BVH for the current pose.

  Note: The original JAX implementation delegates to the Warp backend.
  MLX does not support Warp, so this is left as a stub that raises
  NotImplementedError. It is included for API compatibility.
  """
  raise NotImplementedError(
      'refit_bvh is not implemented for the MLX backend. '
      'The original implementation requires the MuJoCo Warp backend.'
  )
