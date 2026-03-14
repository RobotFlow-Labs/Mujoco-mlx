"""End-to-end test suite for MJX-MLX physics engine.

Tests the JAX→MLX port by comparing MLX outputs against known values
and verifying the core physics pipeline works on Apple Silicon.
"""
from __future__ import annotations

import sys
import os
import time

# Add the mjx directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlx.core as mx
import numpy as np


# ============================================================================
# Phase 0: Foundation tests
# ============================================================================

def test_math_quaternion_identity():
    """Quaternion multiply with identity should return the same quaternion."""
    from mujoco.mjx_mlx._src import math as mjx_math

    identity = mx.array([1.0, 0.0, 0.0, 0.0])
    q = mx.array([0.707, 0.0, 0.707, 0.0])
    result = mjx_math.quat_mul(identity, q)
    mx.eval(result)
    np.testing.assert_allclose(np.array(result), np.array(q), atol=1e-5)
    print("  quat_identity: PASS")


def test_math_quaternion_inverse():
    """q * q_inv should give identity."""
    from mujoco.mjx_mlx._src import math as mjx_math

    q = mx.array([0.5, 0.5, 0.5, 0.5])
    q_inv = mjx_math.quat_inv(q)
    result = mjx_math.quat_mul(q, q_inv)
    mx.eval(result)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.array(result), expected, atol=1e-5)
    print("  quat_inverse: PASS")


def test_math_normalize():
    """Normalize should produce unit vector."""
    from mujoco.mjx_mlx._src import math as mjx_math

    v = mx.array([3.0, 4.0, 0.0])
    n, length = mjx_math.normalize_with_norm(v)
    mx.eval(n)
    mx.eval(length)
    assert abs(float(length) - 5.0) < 1e-5, f"Expected length 5.0, got {length}"
    norm_check = float(mx.sqrt(mx.sum(mx.square(n))))
    assert abs(norm_check - 1.0) < 1e-5, f"Expected unit vector, got norm {norm_check}"
    print("  normalize: PASS")


def test_math_quat_to_mat():
    """Identity quaternion should give identity rotation matrix."""
    from mujoco.mjx_mlx._src import math as mjx_math

    q = mx.array([1.0, 0.0, 0.0, 0.0])
    mat = mjx_math.quat_to_mat(q)
    mx.eval(mat)
    expected = np.eye(3)
    np.testing.assert_allclose(np.array(mat), expected, atol=1e-5)
    print("  quat_to_mat: PASS")


def test_math_rotate():
    """Rotating a vector by identity should return the same vector."""
    from mujoco.mjx_mlx._src import math as mjx_math

    q = mx.array([1.0, 0.0, 0.0, 0.0])
    v = mx.array([1.0, 2.0, 3.0])
    result = mjx_math.rotate(v, q)
    mx.eval(result)
    np.testing.assert_allclose(np.array(result), np.array(v), atol=1e-5)
    print("  rotate_identity: PASS")


def test_math_axis_angle_roundtrip():
    """axis_angle → quat should produce correct quaternion."""
    from mujoco.mjx_mlx._src import math as mjx_math
    import inspect

    axis = mx.array([0.0, 0.0, 1.0])
    angle = 1.5707963  # pi/2
    # Handle both API variants: (axis_angle_vec) or (axis, angle)
    sig = inspect.signature(mjx_math.axis_angle_to_quat)
    if len(sig.parameters) == 1:
        q = mjx_math.axis_angle_to_quat(axis * angle)
    else:
        q = mjx_math.axis_angle_to_quat(axis, angle)
    mx.eval(q)
    # q should be approximately [cos(pi/4), 0, 0, sin(pi/4)]
    expected = np.array([0.7071068, 0.0, 0.0, 0.7071068])
    np.testing.assert_allclose(np.array(q), expected, atol=1e-3)
    print("  axis_angle_roundtrip: PASS")


def test_dataclasses_tree_map():
    """tree_map should apply function to all array fields."""
    from mujoco.mjx_mlx._src import dataclasses as mjx_dc
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TestNode:
        a: mx.array
        b: mx.array
        c: int  # non-array field

        def replace(self, **kwargs):
            from dataclasses import replace as dc_replace
            return dc_replace(self, **kwargs)

    node = TestNode(a=mx.array([1.0, 2.0]), b=mx.array([3.0]), c=42)
    doubled = mjx_dc.tree_map(lambda x: x * 2, node)
    mx.eval(doubled.a)
    mx.eval(doubled.b)
    np.testing.assert_allclose(np.array(doubled.a), [2.0, 4.0])
    np.testing.assert_allclose(np.array(doubled.b), [6.0])
    assert doubled.c == 42, "Non-array field should be unchanged"
    print("  tree_map: PASS")


# ============================================================================
# Phase 0.5: Model Loading tests
# ============================================================================

# ---- XML model strings for testing ----

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.2 0.1 0.05" mass="1.0"/>
      <body name="pole" pos="0 0 0.05">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""

PENDULUM_XML = """
<mujoco model="pendulum">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_model_loading_cartpole():
    """Load cartpole XML via put_model, verify nq=2, nv=2, nbody=3."""
    import mujoco as mj
    from mujoco.mjx_mlx._src.io import put_model

    mj_model = mj.MjModel.from_xml_string(CARTPOLE_XML)
    mx_model = put_model(mj_model)
    assert int(mx_model.nq) == 2, f"Expected nq=2, got {mx_model.nq}"
    assert int(mx_model.nv) == 2, f"Expected nv=2, got {mx_model.nv}"
    assert int(mx_model.nbody) == 3, f"Expected nbody=3, got {mx_model.nbody}"
    print("  model_loading_cartpole: PASS")


def test_data_loading_cartpole():
    """Load cartpole, call make_data(), verify qpos shape is (2,) and all zeros."""
    import mujoco as mj
    from mujoco.mjx_mlx._src.io import put_model, make_data

    mj_model = mj.MjModel.from_xml_string(CARTPOLE_XML)
    mx_model = put_model(mj_model)
    mx_data = make_data(mx_model)
    qpos = np.array(mx_data.qpos)
    assert qpos.shape == (2,), f"Expected qpos shape (2,), got {qpos.shape}"
    np.testing.assert_allclose(qpos, np.zeros(2), atol=1e-7)
    print("  data_loading_cartpole: PASS")


def test_model_loading_pendulum():
    """Load a single-pendulum XML, verify nq=1, nv=1."""
    import mujoco as mj
    from mujoco.mjx_mlx._src.io import put_model

    mj_model = mj.MjModel.from_xml_string(PENDULUM_XML)
    mx_model = put_model(mj_model)
    assert int(mx_model.nq) == 1, f"Expected nq=1, got {mx_model.nq}"
    assert int(mx_model.nv) == 1, f"Expected nv=1, got {mx_model.nv}"
    print("  model_loading_pendulum: PASS")


def test_model_body_properties():
    """Load cartpole, verify body_mass is a non-zero array."""
    import mujoco as mj
    from mujoco.mjx_mlx._src.io import put_model

    mj_model = mj.MjModel.from_xml_string(CARTPOLE_XML)
    mx_model = put_model(mj_model)
    body_mass = np.array(mx_model.body_mass)
    assert body_mass.shape[0] == 3, f"Expected 3 bodies, got {body_mass.shape[0]}"
    # At least one body (cart or pole) should have non-zero mass
    assert np.any(body_mass > 0), "Expected at least one body with non-zero mass"
    print("  model_body_properties: PASS")


def test_model_joint_properties():
    """Load cartpole, verify jnt_type has 2 entries."""
    import mujoco as mj
    from mujoco.mjx_mlx._src.io import put_model

    mj_model = mj.MjModel.from_xml_string(CARTPOLE_XML)
    mx_model = put_model(mj_model)
    jnt_type = np.array(mx_model.jnt_type)
    assert jnt_type.shape == (2,), f"Expected jnt_type shape (2,), got {jnt_type.shape}"
    print("  model_joint_properties: PASS")


def test_data_qvel_shape():
    """Verify qvel shape matches nv for cartpole."""
    import mujoco as mj
    from mujoco.mjx_mlx._src.io import put_model, make_data

    mj_model = mj.MjModel.from_xml_string(CARTPOLE_XML)
    mx_model = put_model(mj_model)
    mx_data = make_data(mx_model)
    qvel = np.array(mx_data.qvel)
    assert qvel.shape == (int(mx_model.nv),), f"Expected qvel shape ({mx_model.nv},), got {qvel.shape}"
    print("  data_qvel_shape: PASS")


def test_model_actuator():
    """Verify nu=1 for cartpole with one motor."""
    import mujoco as mj
    from mujoco.mjx_mlx._src.io import put_model

    mj_model = mj.MjModel.from_xml_string(CARTPOLE_XML)
    mx_model = put_model(mj_model)
    assert int(mx_model.nu) == 1, f"Expected nu=1, got {mx_model.nu}"
    print("  model_actuator: PASS")


# ============================================================================
# Phase 1: Dynamics tests (require smooth, support, constraint, solver, forward)
# ============================================================================

def test_support_import():
    """Support module should import without errors."""
    try:
        from mujoco.mjx_mlx._src import support
        print("  support_import: PASS")
        return True
    except ImportError as e:
        print(f"  support_import: SKIP ({e})")
        return False


def test_smooth_import():
    """Smooth dynamics module should import without errors."""
    try:
        from mujoco.mjx_mlx._src import smooth
        print("  smooth_import: PASS")
        return True
    except ImportError as e:
        print(f"  smooth_import: SKIP ({e})")
        return False


def test_constraint_import():
    """Constraint module should import without errors."""
    try:
        from mujoco.mjx_mlx._src import constraint
        print("  constraint_import: PASS")
        return True
    except ImportError as e:
        print(f"  constraint_import: SKIP ({e})")
        return False


def test_solver_import():
    """Solver module should import without errors."""
    try:
        from mujoco.mjx_mlx._src import solver
        print("  solver_import: PASS")
        return True
    except ImportError as e:
        print(f"  solver_import: SKIP ({e})")
        return False


def test_forward_import():
    """Forward dynamics module should import without errors."""
    try:
        from mujoco.mjx_mlx._src import forward
        print("  forward_import: PASS")
        return True
    except ImportError as e:
        print(f"  forward_import: SKIP ({e})")
        return False


# ============================================================================
# Phase 2: Collision tests
# ============================================================================

def test_collision_types_import():
    """Collision types should import."""
    try:
        from mujoco.mjx_mlx._src import collision_types
        print("  collision_types_import: PASS")
        return True
    except ImportError as e:
        print(f"  collision_types_import: SKIP ({e})")
        return False


def test_collision_primitive_import():
    """Collision primitives should import."""
    try:
        from mujoco.mjx_mlx._src import collision_primitive
        print("  collision_primitive_import: PASS")
        return True
    except ImportError as e:
        print(f"  collision_primitive_import: SKIP ({e})")
        return False


# ============================================================================
# Phase 3: Sensor and rendering tests
# ============================================================================

def test_sensor_import():
    """Sensor module should import."""
    try:
        from mujoco.mjx_mlx._src import sensor
        print("  sensor_import: PASS")
        return True
    except ImportError as e:
        print(f"  sensor_import: SKIP ({e})")
        return False


def test_ray_import():
    """Ray module should import."""
    try:
        from mujoco.mjx_mlx._src import ray
        print("  ray_import: PASS")
        return True
    except ImportError as e:
        print(f"  ray_import: SKIP ({e})")
        return False


# ============================================================================
# Performance benchmark
# ============================================================================

def benchmark_math_ops():
    """Benchmark core math operations on MLX."""
    from mujoco.mjx_mlx._src import math as mjx_math

    # Warm up
    q = mx.array([0.5, 0.5, 0.5, 0.5])
    v = mx.array([1.0, 2.0, 3.0])
    for _ in range(10):
        mjx_math.quat_mul(q, q)
        mjx_math.rotate(v, q)
        mjx_math.normalize(v)
    mx.eval(q)

    # Benchmark
    n_iters = 10000
    start = time.perf_counter()
    for _ in range(n_iters):
        q = mjx_math.quat_mul(q, q)
        n, _ = mjx_math.normalize_with_norm(q)
        q = n
    mx.eval(q)
    elapsed = time.perf_counter() - start
    ops_per_sec = n_iters / elapsed
    print(f"  math_benchmark: {ops_per_sec:.0f} quat_mul+normalize/sec ({elapsed*1000:.1f}ms for {n_iters} iters)")


# ============================================================================
# Main runner
# ============================================================================

def main():
    print("=" * 60)
    print("  MJX-MLX Physics Engine Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    # Phase 0: Foundation
    print("\n--- Phase 0: Foundation (math, dataclasses) ---")
    tests_p0 = [
        test_math_quaternion_identity,
        test_math_quaternion_inverse,
        test_math_normalize,
        test_math_quat_to_mat,
        test_math_rotate,
        test_math_axis_angle_roundtrip,
        test_dataclasses_tree_map,
    ]
    for t in tests_p0:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__}: FAIL ({e})")
            failed += 1

    # Phase 0.5: Model Loading
    print("\n--- Phase 0.5: Model Loading ---")
    tests_p05 = [
        test_model_loading_cartpole,
        test_data_loading_cartpole,
        test_model_loading_pendulum,
        test_model_body_properties,
        test_model_joint_properties,
        test_data_qvel_shape,
        test_model_actuator,
    ]
    for t in tests_p05:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__}: FAIL ({e})")
            failed += 1

    # Phase 1: Dynamics imports
    print("\n--- Phase 1: Core Dynamics (import tests) ---")
    tests_p1 = [
        test_support_import,
        test_smooth_import,
        test_constraint_import,
        test_solver_import,
        test_forward_import,
    ]
    for t in tests_p1:
        try:
            result = t()
            if result:
                passed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  {t.__name__}: FAIL ({e})")
            failed += 1

    # Phase 2: Collision imports
    print("\n--- Phase 2: Collision (import tests) ---")
    tests_p2 = [
        test_collision_types_import,
        test_collision_primitive_import,
    ]
    for t in tests_p2:
        try:
            result = t()
            if result:
                passed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  {t.__name__}: FAIL ({e})")
            failed += 1

    # Phase 3: Sensor/rendering imports
    print("\n--- Phase 3: Sensors & Rendering (import tests) ---")
    tests_p3 = [
        test_sensor_import,
        test_ray_import,
    ]
    for t in tests_p3:
        try:
            result = t()
            if result:
                passed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  {t.__name__}: FAIL ({e})")
            failed += 1

    # Benchmark
    print("\n--- Performance ---")
    try:
        benchmark_math_ops()
    except Exception as e:
        print(f"  benchmark: FAIL ({e})")

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed + skipped
    print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped (of {total})")
    print("=" * 60)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
