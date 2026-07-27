"""ROS2 fusion package — headless tests. 실행: pytest -q

Exercises the ROS-free estimation core (``FusionCore``) WITHOUT importing rclpy,
and checks that the node module imports behind its rclpy guard even when ROS2 is
absent. Same spirit as scripts/02_imu_fusion.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "ros2" / "kalman_fusion"))

from kalman_fusion.fusion_core import FusionCore  # noqa: E402
from sensor_fusion.sim import noisy_accel, noisy_position, true_trajectory  # noqa: E402


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def test_fusion_core_beats_raw_position():
    dt = 0.1
    n = 300
    _, pos, _, acc = true_trajectory(n, dt=dt)
    z_pos = noisy_position(pos, sigma=2.0)
    z_acc = noisy_accel(acc, sigma=0.5, bias=0.0)

    core = FusionCore(pos_sigma=2.0, imu_sigma=0.5)
    core.seed_position(z_pos[0])

    outage = set(range(120, 180))  # position sensor drops out; IMU carries it
    fused = np.zeros((n, 2))
    for k in range(n):
        core.predict(dt)
        core.update_imu(z_acc[k])
        if k not in outage:
            core.update_position(z_pos[k])
        fused[k] = core.position

    # fused estimate must track ground truth better than the raw position sensor
    assert _rmse(fused, pos) < _rmse(z_pos, pos)
    assert np.all(np.isfinite(core.state))

    # covariance stays symmetric / positive-definite
    P = core.covariance
    assert np.allclose(P, P.T, atol=1e-8)
    assert np.all(np.linalg.eigvals(P) > 0)


def test_node_module_imports_without_rclpy():
    # The node module must import even where rclpy is missing (guard works).
    import kalman_fusion.fusion_node as fn

    assert hasattr(fn, "main")
    # main() should return a nonzero exit code (not raise) when ROS2 is absent.
    if not fn._ROS2_AVAILABLE:
        assert fn.main() == 1
