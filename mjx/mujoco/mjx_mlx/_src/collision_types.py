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
"""Collision base types – MLX port."""

import dataclasses
from typing import Tuple

import mlx.core as mx
from mujoco.mjx_mlx._src.dataclasses import PyTreeNode  # pylint: disable=g-importing-member
import numpy as np

# Collision returned by collision functions:
#  - distance          distance between nearest points; neg: penetration
#  - position  (3,)    position of contact point: midpoint between geoms
#  - frame     (3, 3)  normal is in [0, :], points from geom[0] to geom[1]
Collision = Tuple[mx.array, mx.array, mx.array]


class GeomInfo(PyTreeNode):
  """Geom properties for primitive shapes."""

  pos: mx.array
  mat: mx.array
  size: mx.array


class ConvexInfo(PyTreeNode):
  """Geom properties for convex meshes."""

  pos: mx.array
  mat: mx.array
  size: mx.array
  vert: mx.array
  face: mx.array
  face_normal: mx.array
  edge: mx.array
  edge_face_normal: mx.array


class HFieldInfo(PyTreeNode):
  """Geom properties for height fields."""

  pos: mx.array
  mat: mx.array
  size: np.ndarray
  nrow: int
  ncol: int
  data: mx.array


@dataclasses.dataclass(frozen=True)
class FunctionKey:
  """Specifies how geom pairs group into collision_driver's function table.

  Attributes:
    types: geom type pair, which determines the collision function
    data_ids: geom data id pair: mesh id for mesh geoms, otherwise -1. Meshes
      have distinct face/vertex counts, so must occupy distinct entries in the
      collision function table.
    condim: grouping by condim of the collision ensures that the size of the
      resulting constraint jacobian is determined at compile time.
    subgrid_size: the size determines the hfield subgrid to collide with
  """

  types: Tuple[int, int]
  data_ids: Tuple[int, int]
  condim: int
  subgrid_size: Tuple[int, int] = (-1, -1)
