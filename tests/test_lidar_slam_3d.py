"""전체 3D LiDAR SLAM(점-대-평면 ICP 프론트엔드 + SE(3) pose-graph 백엔드) 테스트.

실행: pytest -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "slam29", ROOT / "scripts" / "29_lidar_slam_3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corner_cloud(rng, n=900):
    """세 직교 벽(법선이 x·y·z를 모두 덮음) 표면 점군 — 6-DOF 관측성 확보."""
    a = rng.uniform(0, 6, (n, 2))
    z0 = np.zeros((n, 1))
    floor = np.hstack([a, z0])                     # z=0 평면 (법선 z)
    wall_x = np.hstack([z0, a])                    # x=0 평면 (법선 x)
    wall_y = np.hstack([a[:, :1], z0, a[:, 1:]])   # y=0 평면 (법선 y)
    return np.vstack([floor, wall_x, wall_y])


def test_point_to_plane_icp_recovers_known_transform():
    m = _load()
    rng = np.random.default_rng(0)
    cloud = _corner_cloud(rng)

    # 알려진 SE(3): 회전 phi + 병진 rho
    from sensor_fusion.se3 import se3_exp, se3_inv, se3_log
    xi_true = np.array([0.25, -0.18, 0.30, 0.06, -0.05, 0.08])  # [rho, phi]
    Ttrue = se3_exp(xi_true)
    moved = m.apply_T(Ttrue, cloud)                # cloud → moved = Ttrue

    # icp(src=cloud, dst=moved) 는 cloud→moved 변환 = Ttrue 를 복원해야 함
    T, err, _ = m.point_to_plane_icp(cloud, moved, max_corr_dist=5.0, max_iter=60)
    resid = se3_log(se3_inv(Ttrue) @ T)            # 남은 SE(3) 오차
    assert np.linalg.norm(resid[:3]) < 1e-2        # 병진 오차
    assert np.linalg.norm(resid[3:]) < 1e-2        # 회전 오차
    assert err < 1e-6                              # 완전 정렬(잡음 없음)


def test_backend_reduces_frontend_drift_3d():
    m = _load()
    odom_rmse, opt_rmse, n_lc, chi2_before, chi2_after = m.main(plot=False)

    # 프론트엔드가 눈에 띄게 3D 드리프트(백엔드로 고칠 여지)
    assert odom_rmse > 0.2
    # 장소재인식이 실제 루프클로저를 여럿 찾음
    assert n_lc >= 5
    # 백엔드가 3D 드리프트를 뚜렷이 줄임(0.6배 미만) + 절대 기준 통과
    assert opt_rmse < 0.6 * odom_rmse
    assert opt_rmse < 0.3
    # 최적화가 제약 오차(chi2)를 크게 감소
    assert chi2_after < chi2_before
