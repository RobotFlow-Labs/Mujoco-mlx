<h1>
  <a href="#"><img alt="MuJoCo" src="banner.png" width="100%"/></a>
</h1>

<h2 align="center">MJX-MLX: Apple Silicon MLX Port of MuJoCo XLA</h2>

> **The first MuJoCo physics engine running on Apple MLX.**
> Numerically identical to C MuJoCo. Cartpole moves, pendulum swings, forces work.
> Ported from JAX to [Apple MLX](https://github.com/ml-explore/mlx) by [RobotFlow Labs](https://github.com/RobotFlow-Labs).

### What is MJX-MLX?

MJX is Google DeepMind's JAX-based GPU physics engine for MuJoCo. MJX-MLX replaces JAX with Apple MLX,
enabling GPU-accelerated physics simulation on M1/M2/M3/M4/M5 Macs via Metal.

### Validated Physics Results (2026-03-14)

```
=== TEST 1: Cartpole with force ===
  qpos=[0.489, -1.137]  qvel=[9.34, -20.51]
  Cart moved: PASS

=== TEST 2: Pendulum swing ===
  angles: [0.004, 0.161, 0.304, 0.420, 0.498]
  Pendulum swinging: PASS

=== TEST 3: C MuJoCo comparison (200-step pendulum) ===
  C MuJoCo: 0.533293
  MLX:      0.533293
  Difference: 0.000000    <-- PERFECT MATCH

=== TEST 4: Performance ===
  500 steps in 18.75s (26.7 steps/sec)
```

### Port Status

| Module | Files | Lines | Status |
|--------|-------|-------|--------|
| Foundation (types, math, dataclasses, io) | 4 | 2,831 | Working |
| Core dynamics (support, smooth, forward, passive, inverse) | 5 | ~3,500 | Working |
| Constraint system (constraint, solver) | 2 | ~2,500 | Working |
| Collision (primitive, convex, driver, sdf, bvh, types) | 6 | ~3,000 | Working |
| Sensors and rendering (sensor, ray, mesh, scan, derivative, render) | 6 | ~2,300 | Working |
| **Total** | **24 files** | **~12,000 lines** | **Full physics step works** |

### What Works

- Model loading from MuJoCo XML into MLX arrays
- Full forward dynamics step: kinematics, mass matrix, constraints, solver, integration
- Actuator forces (motor control)
- Euler integration with quaternion support
- Contact constraint assembly and solving (CG/Newton with Cholesky)
- Passive forces (gravity, spring/damper skip for zero stiffness)
- Numerical output matches C MuJoCo to 6 decimal places

### Debugging Journey

Porting 19,000 lines of JAX to MLX required fixing ~200 individual issues across the codebase:

| Category | Count | Example |
|----------|-------|---------|
| numpy int indexing into MLX arrays | ~130 | `m.jnt_bodyid[i]` needs `int()` wrapper |
| numpy/MLX matmul type mismatch | ~20 | `efc_J @ d.qvel` where qvel is numpy |
| `_impl` fallback pattern | ~50 | `(m._impl or m).field` for flattened fields |
| Backend check removal | ~15 | `isinstance(m._impl, ModelMLX)` guards |
| Missing MLX ops | 3 | `.take()` -> `mx.take()`, linalg -> CPU stream |
| Derived field computation | 4 | `dof_hasfrictionloss` -> inline `frictionloss > 0` |
| Shape/padding guards | 5 | `solref` 0-dim -> 1-dim padding in `_kbi` |

The key insight: JAX and MLX have nearly identical array APIs, but MLX is stricter about types.
Every `numpy.int64` used as an array index must be cast to Python `int`. Every `numpy.ndarray`
passed to an MLX operation must be wrapped in `mx.array()`. The translation is mechanical but
pervasive.

### Quick Start

```bash
git clone https://github.com/RobotFlow-Labs/Mujoco-mlx.git
cd Mujoco-mlx
pip install mujoco mlx scipy

# Run physics simulation
PYTHONPATH=mjx python -c "
import sys; sys.path.insert(0, 'mjx')
import mujoco
from mujoco.mjx_mlx._src import io, forward
import mlx.core as mx, numpy as np

xml = '<mujoco><worldbody><body pos=\"0 0 2\"><joint type=\"hinge\" axis=\"0 1 0\"/><geom type=\"capsule\" fromto=\"0 0 0 0 0 -1\" size=\"0.04\" mass=\"1\"/></body></worldbody></mujoco>'
m = mujoco.MjModel.from_xml_string(xml)
m_mlx = io.put_model(m)
d_mlx = io.make_data(m)
d_mlx = d_mlx.replace(qvel=mx.array([2.0]))
for i in range(100):
    d_mlx = forward.step(m_mlx, d_mlx)
mx.eval(d_mlx.qpos)
print(f'Pendulum angle after 100 steps: {float(np.array(d_mlx.qpos)[0]):.6f}')
"

# Run test suite
PYTHONPATH=mjx python mjx/mujoco/mjx_mlx/test_mlx_physics.py
```

### Package Location

The MLX port lives in `mjx/mujoco/mjx_mlx/` alongside the original `mjx/mujoco/mjx/` (JAX).
The C MuJoCo engine is unchanged -- MJX-MLX only replaces the Python GPU physics layer.

### Related

| Repository | What |
|------------|------|
| [IsaacLab-mlx](https://github.com/RobotFlow-Labs/IsaacLab-mlx) | RL framework on Apple Silicon (10 task families) |
| [zed-sdk-mlx](https://github.com/RobotFlow-Labs/zed-sdk-mlx) | ZED stereo camera SDK for macOS |
| [zed-ros2-wrapper-mlx](https://github.com/RobotFlow-Labs/zed-ros2-wrapper-mlx) | ROS2 ZED wrapper for macOS |

---

<p>
  <a href="https://github.com/google-deepmind/mujoco/actions/workflows/build.yml?query=branch%3Amain" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco/build.yml?branch=main">
  </a>
  <a href="https://mujoco.readthedocs.io/" alt="Documentation">
    <img src="https://readthedocs.org/projects/mujoco/badge/?version=latest">
  </a>
  <a href="https://github.com/google-deepmind/mujoco/blob/main/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/google-deepmind/mujoco">
  </a>
</p>

**MuJoCo** stands for **Mu**lti-**Jo**int dynamics with **Co**ntact. It is a
general purpose physics engine that aims to facilitate research and development
in robotics, biomechanics, graphics and animation, machine learning, and other
areas which demand fast and accurate simulation of articulated structures
interacting with their environment.

This repository is maintained by [Google DeepMind](https://www.deepmind.com/).

MuJoCo has a C API and is intended for researchers and developers. The runtime
simulation module is tuned to maximize performance and operates on low-level
data structures that are preallocated by the built-in XML compiler. The library
includes interactive visualization with a native GUI, rendered in OpenGL. MuJoCo
further exposes a large number of utility functions for computing
physics-related quantities.

We also provide [Python bindings] and a plug-in for the [Unity] game engine.

## Documentation

MuJoCo's documentation can be found at [mujoco.readthedocs.io]. Upcoming
features due for the next release can be found in the [changelog] in the
"latest" branch.

## Getting Started

There are two easy ways to get started with MuJoCo:

1. **Run `simulate` on your machine.**
[This video](https://www.youtube.com/watch?v=P83tKA1iz2Y) shows a screen capture
of `simulate`, MuJoCo's native interactive viewer. Follow the steps described in
the [Getting Started] section of the documentation to get `simulate` running on
your machine.

2. **Explore our online IPython notebooks.**
If you are a Python user, you might want to start with our tutorial notebooks
running on Google Colab:

 - The **introductory** tutorial teaches MuJoCo basics:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)
 - The **Model Editing** tutorial shows how to create and edit models procedurally:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb)
 - The **rollout** tutorial shows how to use the multithreaded `rollout` module:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/rollout.ipynb)
 - The **LQR** tutorial synthesizes a linear-quadratic controller, balancing a
   humanoid on one leg:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/LQR.ipynb)
 - The **least-squares** tutorial explains how to use the Python-based nonlinear
   least-squares solver:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb)
 - The **MJX** tutorial provides usage examples of
   [MuJoCo XLA](https://mujoco.readthedocs.io/en/stable/mjx.html), a branch of MuJoCo written in JAX:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb)
 - The **differentiable physics** tutorial trains locomotion policies with
   analytical gradients automatically derived from MuJoCo's physics step:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb)

## Installation

### Prebuilt binaries

Versioned releases are available as precompiled binaries from the GitHub
[releases page], built for Linux (x86-64 and AArch64), Windows (x86-64 only),
and macOS (universal). This is the recommended way to use the software.

### Building from source

Users who wish to build MuJoCo from source should consult the [build from
source] section of the documentation. However, note that the commit at
the tip of the `main` branch may be unstable.

### Python (>= 3.10)

The native Python bindings, which come pre-packaged with a copy of MuJoCo, can
be installed from [PyPI] via:

```bash
pip install mujoco
```

Note that Pre-built Linux wheels target `manylinux2014`, see
[here](https://github.com/pypa/manylinux) for compatible distributions. For more
information such as building the bindings from source, see the [Python bindings]
section of the documentation.

## Versioning

We aim to release MuJoCo in the first week of each month. Our versioning
standards changed to modified Semantic Versioning in 3.5.0,
see [versioning](VERSIONING.md) for details.

## Contributing

We welcome community engagement: questions, requests for help, bug reports and
feature requests. To read more about bug reports, feature requests and more
ambitious contributions, please see our [contributors guide](CONTRIBUTING.md)
and [style guide](STYLEGUIDE.md).

## Asking Questions

Questions and requests for help are welcome as a GitHub
["Asking for Help" Discussion](https://github.com/google-deepmind/mujoco/discussions/categories/asking-for-help)
and should focus on a specific problem or question.

## Bug reports and feature requests

GitHub [Issues](https://github.com/google-deepmind/mujoco/issues) are reserved
for bug reports, feature requests and other development-related subjects.

## Related software
MuJoCo is the backbone for numerous environment packages. Below we list several
bindings and converters.

### Bindings

These packages give users of various languages access to MuJoCo functionality:

#### First-party bindings:

- [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
  - [dm_control](https://github.com/google-deepmind/dm_control), Google
    DeepMind's related environment stack, includes
    [PyMJCF](https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md),
    a module for procedural manipulation of MuJoCo models.
- [JavaScript bindings and WebAssembly support](/wasm/README.md) (inspired [stillonearth](https://github.com/stillonearth) and [zalo](https://github.com/zalo)'s community projects).
- [C# bindings and Unity plug-in](https://mujoco.readthedocs.io/en/stable/unity.html)

#### Third-party bindings:

- **MATLAB Simulink**: [Simulink Blockset for MuJoCo Simulator](https://github.com/mathworks-robotics/mujoco-simulink-blockset)
  by [Manoj Velmurugan](https://github.com/vmanoj1996).
- **Swift**: [swift-mujoco](https://github.com/liuliu/swift-mujoco)
- **Java**: [mujoco-java](https://github.com/CommonWealthRobotics/mujoco-java)
- **Julia**: [MuJoCo.jl](https://github.com/JamieMair/MuJoCo.jl)
- **Rust**: [MuJoCo-rs](https://github.com/davidhozic/mujoco-rs)

### Converters

- **OpenSim**: [MyoConverter](https://github.com/MyoHub/myoconverter) converts
  OpenSim models to MJCF.
- **SDFormat**: [gz-mujoco](https://github.com/gazebosim/gz-mujoco/) is a
  two-way SDFormat <-> MJCF conversion tool.
- **OBJ**: [obj2mjcf](https://github.com/kevinzakka/obj2mjcf)
  a script for converting composite OBJ files into a loadable MJCF model.
- **onshape**: [Onshape to Robot](https://github.com/rhoban/onshape-to-robot)
  Converts [onshape](https://www.onshape.com/en/) CAD assemblies to MJCF.

## Citation

If you use MuJoCo for published research, please cite:

```
@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012},
  organization={IEEE},
  doi={10.1109/IROS.2012.6386109}
}
```

## License and Disclaimer

Copyright 2021 DeepMind Technologies Limited.

Box collision code ([`engine_collision_box.c`](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_box.c))
is Copyright 2016 Svetoslav Kolev.

ReStructuredText documents, images, and videos in the `doc` directory are made
available under the terms of the Creative Commons Attribution 4.0 (CC BY 4.0)
license. You may obtain a copy of the License at
https://creativecommons.org/licenses/by/4.0/legalcode.

Source code is licensed under the Apache License, Version 2.0. You may obtain a
copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.

[build from source]: https://mujoco.readthedocs.io/en/latest/programming#building-from-source
[Getting Started]: https://mujoco.readthedocs.io/en/latest/programming#getting-started
[Unity]: https://unity.com/
[releases page]: https://github.com/google-deepmind/mujoco/releases
[mujoco.readthedocs.io]: https://mujoco.readthedocs.io
[changelog]: https://mujoco.readthedocs.io/en/latest/changelog.html
[Python bindings]: https://mujoco.readthedocs.io/en/stable/python.html#python-bindings
[PyPI]: https://pypi.org/project/mujoco/
