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
"""MLX dataclass utilities for MJX.

Ported from JAX pytree dataclasses to plain Python dataclasses with MLX array
support. Instead of JAX pytree registration, we provide a `tree_map` helper
that applies a function to all mlx.core.array fields in a dataclass.
"""

import copy
import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar

import mlx.core as mx
import numpy as np

_T = TypeVar('_T')


def _is_array_field(value: Any) -> bool:
  """Check if a value is an MLX array or a nested dataclass containing arrays."""
  if isinstance(value, mx.array):
    return True
  if isinstance(value, np.ndarray):
    return False  # numpy arrays are treated as metadata, not mapped over
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return True
  if isinstance(value, (list, tuple)):
    return any(_is_array_field(v) for v in value)
  if isinstance(value, dict):
    return any(_is_array_field(v) for v in value.values())
  return False


def _apply_to_arrays(fn: Callable, value: Any) -> Any:
  """Recursively apply fn to all mx.array leaves in a nested structure."""
  if isinstance(value, mx.array):
    return fn(value)
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return tree_map(fn, value)
  if isinstance(value, tuple):
    mapped = tuple(_apply_to_arrays(fn, v) for v in value)
    # Preserve namedtuple types
    if hasattr(type(value), '_fields'):
      return type(value)(*mapped)
    return mapped
  if isinstance(value, list):
    return [_apply_to_arrays(fn, v) for v in value]
  if isinstance(value, dict):
    return {k: _apply_to_arrays(fn, v) for k, v in value.items()}
  return value


def tree_map(fn: Callable, node: _T, *rest: Any) -> _T:
  """Apply fn to all mx.array fields in a dataclass, returning a new instance.

  This is the MLX replacement for jax.tree_util.tree_map. It recursively
  traverses dataclass fields and applies fn to every mx.array leaf.

  Args:
    fn: function to apply to each mx.array leaf.
    node: a dataclass instance to map over.
    *rest: additional dataclass instances to map over in parallel (like
      jax.tree_util.tree_map with multiple trees).

  Returns:
    A new dataclass instance with fn applied to all array fields.
  """
  if not dataclasses.is_dataclass(node) or isinstance(node, type):
    raise TypeError(f'tree_map expects a dataclass instance, got {type(node)}')

  if rest:
    # Multi-tree map: apply fn(leaf_from_node, leaf_from_rest0, ...)
    return _tree_map_multi(fn, node, *rest)

  updates = {}
  for field in dataclasses.fields(node):
    value = getattr(node, field.name)
    mapped = _apply_to_arrays(fn, value)
    if mapped is not value:
      updates[field.name] = mapped

  if updates:
    return dataclasses.replace(node, **updates)
  return node


def _apply_to_arrays_multi(fn: Callable, *values: Any) -> Any:
  """Multi-tree version of _apply_to_arrays."""
  first = values[0]
  if isinstance(first, mx.array):
    return fn(*values)
  if dataclasses.is_dataclass(first) and not isinstance(first, type):
    return _tree_map_multi(fn, *values)
  if isinstance(first, tuple):
    mapped = tuple(
        _apply_to_arrays_multi(fn, *(v[i] for v in values))
        for i in range(len(first))
    )
    if hasattr(type(first), '_fields'):
      return type(first)(*mapped)
    return mapped
  if isinstance(first, list):
    return [
        _apply_to_arrays_multi(fn, *(v[i] for v in values))
        for i in range(len(first))
    ]
  if isinstance(first, dict):
    return {
        k: _apply_to_arrays_multi(fn, *(v[k] for v in values))
        for k in first
    }
  return first


def _tree_map_multi(fn: Callable, node: _T, *rest: Any) -> _T:
  """Multi-tree map implementation."""
  updates = {}
  for field in dataclasses.fields(node):
    values = [getattr(node, field.name)] + [
        getattr(r, field.name) for r in rest
    ]
    mapped = _apply_to_arrays_multi(fn, *values)
    if mapped is not values[0]:
      updates[field.name] = mapped

  if updates:
    return dataclasses.replace(node, **updates)
  return node


def dataclass(clz: _T) -> _T:
  """Create a frozen dataclass with a replace() method.

  MLX port: no pytree registration needed. We just create a standard frozen
  dataclass and attach `replace` as a convenience method.

  Args:
    clz: the class to register as a dataclass.

  Returns:
    The resulting frozen dataclass.
  """
  data_clz = dataclasses.dataclass(frozen=True)(clz)
  data_clz.replace = dataclasses.replace
  return data_clz


TNode = TypeVar('TNode', bound='PyTreeNode')


class PyTreeNode:
  """Base class for dataclasses that should act like array-carrying nodes.

  MLX port of the JAX PyTreeNode. Instead of registering with JAX's pytree
  system, this simply creates a frozen dataclass with convenience methods.
  The `tree_map` function can be used to map over array fields.
  """

  def __init_subclass__(cls, **kwargs):
    # Pop register_as_pytree if passed (for compatibility), but ignore it
    kwargs.pop('register_as_pytree', None)
    super().__init_subclass__(**kwargs)
    dataclass(cls)

  def __init__(self, *args, **kwargs):
    # stub for type checkers
    raise NotImplementedError

  def replace(self: TNode, **overrides) -> TNode:
    # stub for type checkers
    raise NotImplementedError

  @classmethod
  def fields(cls) -> Tuple[dataclasses.Field, ...]:  # pylint: disable=g-bare-generic
    return dataclasses.fields(cls)

  def tree_replace(
      self, params: Dict[str, Optional[mx.array]]
  ) -> 'PyTreeNode':
    """Replace nested fields using dot-separated keys."""
    new = self
    for k, v in params.items():
      new = _tree_replace(new, k.split('.'), v)
    return new


def _tree_replace(
    base: PyTreeNode,
    attr: Sequence[str],
    val: Optional[mx.array],
) -> PyTreeNode:
  """Sets attributes in a dataclass with values."""
  if not attr:
    return base

  # special case for List attribute
  if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
    lst = copy.deepcopy(getattr(base, attr[0]))

    for i, g in enumerate(lst):
      if not hasattr(g, attr[1]):
        continue
      v = val if not hasattr(val, '__iter__') else val[i]
      lst[i] = _tree_replace(g, attr[1:], v)

    return base.replace(**{attr[0]: lst})

  if len(attr) == 1:
    return base.replace(**{attr[0]: val})

  return base.replace(
      **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
  )
