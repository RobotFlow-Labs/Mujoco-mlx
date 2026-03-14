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
"""Some useful math functions."""

from typing import Optional, Tuple, Union

import mlx.core as mx
import mujoco

# MLX port notes:
# - jax.Array -> mx.array
# - jnp.XXX -> mx.XXX
# - jnp.dot(a, b) -> a @ b or mx.sum(a * b) for 1D dot products
# - jnp.cross -> _cross (manual implementation)
# - jnp.linalg.norm -> _norm_raw (manual implementation)
# - jnp.insert -> mx.concatenate based reconstruction
# - jnp.outer -> manual outer product
# - jp.clip -> mx.clip
# - jp.allclose -> manual allclose


def _cross(a: mx.array, b: mx.array) -> mx.array:
  """Manual cross product for 3D vectors (MLX has no built-in cross)."""
  return mx.array([
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  ])


def _norm_raw(x: mx.array, axis: Optional[Union[Tuple[int, ...], int]] = None) -> mx.array:
  """Raw norm computation (equivalent to jnp.linalg.norm)."""
  return mx.sqrt(mx.sum(mx.square(x), axis=axis))


def _allclose(x: mx.array, val: float, atol: float = 1e-8) -> mx.array:
  """Check if all elements are close to val."""
  return mx.all(mx.abs(x - val) <= atol)


def _outer(a: mx.array, b: mx.array) -> mx.array:
  """Outer product of two 1D arrays."""
  return a[:, None] * b[None, :]


def safe_div(
    num: Union[float, mx.array], den: Union[float, mx.array]
) -> Union[float, mx.array]:
  """Safe division for case where denominator is zero."""
  return num / (den + mujoco.mjMINVAL * (den == 0))


def matmul_unroll(a: mx.array, b: mx.array) -> mx.array:
  """Calculates a @ b via explicit cell value operations.

  This is faster than XLA matmul for small matrices (e.g. 3x3, 4x4).

  Args:
    a: left hand of matmul operand
    b: right hand of matmul operand

  Returns:
    the matrix product of the inputs.
  """
  c = []
  for i in range(a.shape[0]):
    row = []
    for j in range(b.shape[1]):
      s = 0.0
      for k in range(a.shape[1]):
        s += a[i, k] * b[k, j]
      row.append(s)
    c.append(row)

  return mx.array(c)


def norm(
    x: mx.array, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> mx.array:
  """Calculates a linalg.norm(x) that's safe for gradients at x=0.

  Avoids a poorly defined gradient for jnp.linal.norm(0) see
  https://github.com/jax-ml/jax/issues/3058 for details
  Args:
    x: A mx.array
    axis: The axis along which to compute the norm

  Returns:
    Norm of the array x.
  """

  is_zero = _allclose(x, 0.0)
  # temporarily swap x with ones if is_zero, then swap back
  x = mx.where(is_zero, mx.ones_like(x), x)
  n = _norm_raw(x, axis=axis)
  n = mx.where(is_zero, 0.0, n)
  return n


def normalize_with_norm(
    x: mx.array, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> Tuple[mx.array, mx.array]:
  """Normalizes an array.

  Args:
    x: A mx.array
    axis: The axis along which to compute the norm

  Returns:
    A tuple of (normalized array x, the norm).
  """
  n = norm(x, axis=axis)
  x = x / (n + 1e-6 * (n == 0.0))
  return x, n


def normalize(
    x: mx.array, axis: Optional[Union[Tuple[int, ...], int]] = None
) -> mx.array:
  """Normalizes an array.

  Args:
    x: A mx.array
    axis: The axis along which to compute the norm

  Returns:
    normalized array x
  """
  return normalize_with_norm(x, axis=axis)[0]


def rotate(vec: mx.array, quat: mx.array) -> mx.array:
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  if len(vec.shape) != 1:
    raise ValueError('vec must have no batch dimensions.')
  s, u = quat[0], quat[1:]
  r = 2 * (mx.sum(u * vec) * u) + (s * s - mx.sum(u * u)) * vec
  r = r + 2 * s * _cross(u, vec)
  return r


def quat_inv(q: mx.array) -> mx.array:
  """Calculates the inverse of quaternion q.

  Args:
    q: (4,) quaternion [w, x, y, z]

  Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
  """
  return q * mx.array([1.0, -1.0, -1.0, -1.0])


def quat_sub(u: mx.array, v: mx.array) -> mx.array:
  """Subtracts two quaternions (u - v) as a 3D velocity."""
  q = quat_mul(quat_inv(v), u)
  axis, angle = quat_to_axis_angle(q)
  return axis * angle


def quat_mul(u: mx.array, v: mx.array) -> mx.array:
  """Multiplies two quaternions.

  Args:
    u: (4,) quaternion (w,x,y,z)
    v: (4,) quaternion (w,x,y,z)

  Returns:
    A quaternion u * v.
  """
  return mx.array([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])


def quat_mul_axis(q: mx.array, axis: mx.array) -> mx.array:
  """Multiplies a quaternion and an axis.

  Args:
    q: (4,) quaternion (w,x,y,z)
    axis: (3,) axis (x,y,z)

  Returns:
    A quaternion q * axis
  """
  return mx.array([
      -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
      q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
      q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
      q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
  ])


# TODO(erikfrey): benchmark this against brax's quat_to_3x3
def quat_to_mat(q: mx.array) -> mx.array:
  """Converts a quaternion into a 9-dimensional rotation matrix."""
  q = _outer(q, q)

  return mx.array([
      [
          q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
          2 * (q[1, 2] - q[0, 3]),
          2 * (q[1, 3] + q[0, 2]),
      ],
      [
          2 * (q[1, 2] + q[0, 3]),
          q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
          2 * (q[2, 3] - q[0, 1]),
      ],
      [
          2 * (q[1, 3] - q[0, 2]),
          2 * (q[2, 3] + q[0, 1]),
          q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
      ],
  ])


def quat_to_axis_angle(q: mx.array) -> Tuple[mx.array, mx.array]:
  """Converts a quaternion into axis and angle."""
  axis, sin_a_2 = normalize_with_norm(q[1:])
  angle = 2 * mx.arctan2(sin_a_2, q[0])
  angle = mx.where(angle > mx.array(3.141592653589793), angle - 2 * mx.array(3.141592653589793), angle)

  return axis, angle


def axis_angle_to_quat(axis: mx.array, angle: mx.array) -> mx.array:
  """Provides a quaternion that describes rotating around axis by angle.

  Args:
    axis: (3,) axis (x,y,z)
    angle: () float angle to rotate by

  Returns:
    A quaternion that rotates around axis by angle
  """
  s, c = mx.sin(angle * 0.5), mx.cos(angle * 0.5)
  # jp.insert(axis * s, 0, c) -> concatenate c at front
  return mx.concatenate([mx.array([c]), axis * s])


def quat_integrate(q: mx.array, v: mx.array, dt: mx.array) -> mx.array:
  """Integrates a quaternion given angular velocity and dt."""
  v, norm_ = normalize_with_norm(v)
  angle = dt * norm_
  q_res = axis_angle_to_quat(v, angle)
  q_res = quat_mul(q, q_res)
  return normalize(q_res)


def inert_mul(i: mx.array, v: mx.array) -> mx.array:
  """Multiply inertia by motion, producing force.

  Args:
    i: (10,) inertia (inertia matrix, position, mass)
    v: (6,) spatial motion

  Returns:
    resultant force
  """
  tri_id = mx.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])  # cinert inr order
  inr, pos, mass = i[tri_id], i[6:9], i[9]
  ang = inr @ v[:3] + _cross(pos, v[3:])
  vel = mass * v[3:] - _cross(pos, v[:3])
  return mx.concatenate([ang, vel])


def sign(x: mx.array) -> mx.array:
  """Returns the sign of x in the set {-1, 1}."""
  return mx.where(x < 0, -1, 1)


def transform_motion(vel: mx.array, offset: mx.array, rotmat: mx.array):
  """Transform spatial motion.

  Args:
    vel: (6,) spatial motion (3 angular, 3 linear)
    offset: (3,) translation
    rotmat: (3, 3) rotation

  Returns:
    6d spatial velocity
  """
  # TODO(robotics-simulation): are quaternions faster here
  ang, vel = vel[:3], vel[3:]
  vel = rotmat.T @ (vel - _cross(offset, ang))
  ang = rotmat.T @ ang
  return mx.concatenate([ang, vel])


def motion_cross(u, v):
  """Cross product of two motions.

  Args:
    u: (6,) spatial motion
    v: (6,) spatial motion

  Returns:
    resultant spatial motion
  """
  ang = _cross(u[:3], v[:3])
  vel = _cross(u[3:], v[:3]) + _cross(u[:3], v[3:])
  return mx.concatenate([ang, vel])


def motion_cross_force(v, f):
  """Cross product of a motion and force.

  Args:
    v: (6,) spatial motion
    f: (6,) force

  Returns:
    resultant force
  """
  ang = _cross(v[:3], f[:3]) + _cross(v[3:], f[3:])
  vel = _cross(v[:3], f[3:])
  return mx.concatenate([ang, vel])


def orthogonals(a: mx.array) -> Tuple[mx.array, mx.array]:
  """Returns orthogonal vectors `b` and `c`, given a vector `a`."""
  y, z = mx.array([0.0, 1.0, 0.0]), mx.array([0.0, 0.0, 1.0])
  b = mx.where((-0.5 < a[1]) & (a[1] < 0.5), y, z)
  b = b - a * mx.sum(a * b)
  # normalize b. however if a is a zero vector, zero b as well.
  b = normalize(b) * mx.any(a)
  return b, _cross(a, b)


def make_frame(a: mx.array) -> mx.array:
  """Makes a right-handed 3D frame given a direction."""
  a = normalize(a)
  b, c = orthogonals(a)
  return mx.array([a, b, c])


# Geometry.


def closest_segment_point(
    a: mx.array, b: mx.array, pt: mx.array
) -> mx.array:
  """Returns the closest point on the a-b line segment to a point pt."""
  ab = b - a
  t = mx.sum((pt - a) * ab) / (mx.sum(ab * ab) + 1e-6)
  return a + mx.clip(t, 0.0, 1.0) * ab


def closest_segment_point_and_dist(
    a: mx.array, b: mx.array, pt: mx.array
) -> Tuple[mx.array, mx.array]:
  """Returns closest point on the line segment and the distance squared."""
  closest = closest_segment_point(a, b, pt)
  diff = pt - closest
  dist = mx.sum(diff * diff)
  return closest, dist


def closest_segment_to_segment_points(
    a0: mx.array, a1: mx.array, b0: mx.array, b1: mx.array
) -> Tuple[mx.array, mx.array]:
  """Returns closest points between two line segments."""
  # Gets the closest segment points by first finding the closest points
  # between two lines. Points are then clipped to be on the line segments
  # and edge cases with clipping are handled.
  dir_a, len_a = normalize_with_norm(a1 - a0)
  dir_b, len_b = normalize_with_norm(b1 - b0)

  # Segment mid-points.
  half_len_a = len_a * 0.5
  half_len_b = len_b * 0.5
  a_mid = a0 + dir_a * half_len_a
  b_mid = b0 + dir_b * half_len_b

  # Translation between two segment mid-points.
  trans = a_mid - b_mid

  # Parametrize points on each line as follows:
  #  point_on_a = a_mid + t_a * dir_a
  #  point_on_b = b_mid + t_b * dir_b
  # and analytically minimize the distance between the two points.
  dira_dot_dirb = mx.sum(dir_a * dir_b)
  dira_dot_trans = mx.sum(dir_a * trans)
  dirb_dot_trans = mx.sum(dir_b * trans)
  denom = 1 - dira_dot_dirb * dira_dot_dirb

  orig_t_a = (-dira_dot_trans + dira_dot_dirb * dirb_dot_trans) / (denom + 1e-6)
  orig_t_b = dirb_dot_trans + orig_t_a * dira_dot_dirb
  t_a = mx.clip(orig_t_a, -half_len_a, half_len_a)
  t_b = mx.clip(orig_t_b, -half_len_b, half_len_b)

  best_a = a_mid + dir_a * t_a
  best_b = b_mid + dir_b * t_b

  # Resolve edge cases where both closest points are clipped to the segment
  # endpoints by recalculating the closest segment points for the current
  # clipped points, and then picking the pair of points with smallest
  # distance. An example of this edge case is when lines intersect but line
  # segments don't.
  new_a, d1 = closest_segment_point_and_dist(a0, a1, best_b)
  new_b, d2 = closest_segment_point_and_dist(b0, b1, best_a)
  best_a = mx.where(d1 < d2, new_a, best_a)
  best_b = mx.where(d1 < d2, best_b, new_b)

  return best_a, best_b
