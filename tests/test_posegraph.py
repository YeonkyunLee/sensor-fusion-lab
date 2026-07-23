"""pose-graph SLAM 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensor_fusion.posegraph import Edge, optimize, t2v, v2t


def test_v2t_t2v_roundtrip():
    for v in [np.array([1.0, 2.0, 0.5]), np.array([-3.0, 0.4, -2.9])]:
        assert np.allclose(t2v(v2t(v)), v, atol=1e-9)


def test_optimize_reduces_error_with_loop_closure():
    spec = importlib.util.spec_from_file_location("pg07", ROOT / "scripts" / "07_pose_graph_slam.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rmse_odo, rmse_opt = mod.main()
    # 루프 클로저 최적화가 오도메트리보다 뚜렷이 정확
    assert rmse_opt < rmse_odo * 0.5


def test_vio_graph_backend_corrects_drift():
    spec = importlib.util.spec_from_file_location("vg10", ROOT / "scripts" / "10_vio_graph_slam.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vio_rmse, opt_rmse = mod.main()
    # 팩터그래프 백엔드가 VIO 드리프트를 크게 줄여야 함
    assert opt_rmse < vio_rmse * 0.2


def test_robust_rejects_false_loop_closures():
    spec = importlib.util.spec_from_file_location("rs11", ROOT / "scripts" / "11_robust_slam.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    naive_rmse, robust_rmse = mod.main()
    # 강건 백엔드가 거짓 루프클로저 오염을 뚜렷이 줄여야 함
    assert robust_rmse < naive_rmse * 0.6


def test_full_graph_slam_joint_optimization():
    spec = importlib.util.spec_from_file_location("gsl12", ROOT / "scripts" / "12_graph_slam_landmarks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pose_odo, pose_opt, map_opt = mod.main()
    assert pose_opt < pose_odo * 0.25   # 공동 최적화가 pose를 크게 개선
    assert map_opt < 1.0                 # 지도도 정밀 복원


def test_optimize_perfect_when_consistent():
    # 무잡음 상대 측정이면 최적화 후 chi2가 ~0
    poses = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, np.pi / 2]], float)
    edges = []
    for k in range(len(poses) - 1):
        z = t2v(np.linalg.inv(v2t(poses[k])) @ v2t(poses[k + 1]))
        edges.append(Edge(k, k + 1, z, np.eye(3)))
    x0 = poses + np.array([0, 0, 0])  # 정확한 초기값
    x, hist = optimize(x0.copy(), edges, iters=5)
    assert hist[-1] < 1e-6
