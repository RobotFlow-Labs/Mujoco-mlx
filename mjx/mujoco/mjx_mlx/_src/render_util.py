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
"""JAX render utilities for unpacking render output from MuJoCo Warp."""

from typing import TYPE_CHECKING

import mlx.core as mx
import mujoco.mjx.warp as mjxw

if TYPE_CHECKING:
  from mujoco.mjx.warp.render_context import RenderContextPytree


def _dynamic_slice_in_dim(
    arr: mx.array, start: int, size: int, axis: int = 0
) -> mx.array:
  """Small MLX replacement for jax.lax.dynamic_slice_in_dim."""
  if axis != 0:
    raise NotImplementedError('_dynamic_slice_in_dim currently supports axis=0')
  return arr[start : start + size]

def get_rgb(
    rc: 'RenderContextPytree',
    cam_id: int,
    rgb_data: mx.array,
) -> mx.array:
  """Unpack uint32 ABGR pixel data into float32 RGB.

  Args:
    rc: RenderContextPytree.
    cam_id: Camera index to extract.
    rgb_data: Packed render output, shape (total_pixels,) as uint32.

  Returns:
    Float32 RGB array with shape (H, W, 3), values in [0, 1].

  Raises:
    RuntimeError: If Warp is not installed.
  """
  if not mjxw.WARP_INSTALLED:
    raise RuntimeError('Warp not installed.')

  import mujoco.mjx.warp.render_context as mjxw_rc  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  if not isinstance(rc, mjxw_rc.RenderContextPytree):
    raise TypeError(
        f'Expected RenderContextPytree, got {type(rc).__name__}.'
        ' Use rc.pytree() to get the JAX-compatible handle.'
    )

  warp_rc = mjxw_rc._MJX_RENDER_CONTEXT_BUFFERS[(rc.key, None)]  # pylint: disable=protected-access
  rgb_adr = int(warp_rc.rgb_adr.numpy()[cam_id])
  width = int(warp_rc.cam_res.numpy()[cam_id][0])
  height = int(warp_rc.cam_res.numpy()[cam_id][1])

  packed = _dynamic_slice_in_dim(
      rgb_data, rgb_adr, width * height, axis=0
  )

  b = (packed & 0xFF).astype(mx.float32) / 255.0
  g = ((packed >> 8) & 0xFF).astype(mx.float32) / 255.0
  r = ((packed >> 16) & 0xFF).astype(mx.float32) / 255.0
  rgb = mx.stack([r, g, b], axis=-1)
  return rgb.reshape(height, width, 3)

def get_depth(
    rc: 'RenderContextPytree',
    cam_id: int,
    depth_data: mx.array,
    depth_scale: float,
) -> mx.array:
  """Extract and normalize depth data for a camera.

  Args:
    rc: RenderContextPytree.
    cam_id: Camera index to extract.
    depth_data: Raw depth output, shape (total_pixels,) as float32.
    depth_scale: Scale factor for normalizing depth values.

  Returns:
    Float32 depth array with shape (H, W), clamped to [0, 1].

  Raises:
    RuntimeError: If Warp is not installed.
  """
  if not mjxw.WARP_INSTALLED:
    raise RuntimeError('Warp not installed.')

  import mujoco.mjx.warp.render_context as mjxw_rc  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  if not isinstance(rc, mjxw_rc.RenderContextPytree):
    raise TypeError(
        f'Expected RenderContextPytree, got {type(rc).__name__}.'
        ' Use rc.pytree() to get the JAX-compatible handle.'
    )

  warp_rc = mjxw_rc._MJX_RENDER_CONTEXT_BUFFERS[(rc.key, None)]  # pylint: disable=protected-access
  depth_adr = int(warp_rc.depth_adr.numpy()[cam_id])
  width = int(warp_rc.cam_res.numpy()[cam_id][0])
  height = int(warp_rc.cam_res.numpy()[cam_id][1])

  raw = _dynamic_slice_in_dim(
      depth_data, depth_adr, width * height, axis=0
  )

  depth = mx.clip(raw / depth_scale, 0.0, 1.0)
  return depth.reshape(height, width, 1)
