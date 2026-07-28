"""수술용 환자-영상 정합(patient-to-image registration)을 ICP로 푸는 실험.

영상유도수술(image-guided surgery)의 핵심 전처리는 **환자-영상 정합**이다. 수술 전
CT/MR로 만든 3D 해부 모델(영상 좌표계)과, 수술 중 로봇/광학 트래커가 환자 표면을
프로브로 디지타이징한 점군(환자·트래커 좌표계)을 하나의 좌표계로 묶어야, 영상에서
계획한 표적(종양·삽입경로)을 실제 수술 도구 좌표로 옮겨 유도할 수 있다. 두 좌표계를
잇는 미지의 강체변환 SE(3)(회전+병진)을 복원하는 것이 정합이며, 그 표준 도구가
**ICP(Iterative Closest Point)**다.

이 실험은 exp 21/29의 ICP 자산을 수술 정합으로 재구성한다. 3D 점-대-평면 ICP
(exp 29의 k-NN PCA 법선 + se(3) 접공간 선형화)를 그대로 쓰되, 시나리오를 바꾼다:

  - **수술 전 모델(dst)** : 합성 해부 표면(뼈/장기처럼 굽은 돔형 타원체 캡 + 미세 융기)의
    조밀 점군. 곡률이 세 축을 고루 덮어 6-DOF 정합이 잘 조건화된다.
  - **수술 중 점군(src)** : 같은 표면을 프로브로 찍은 점들 — 단, 미지의 SE(3)로 옮겨져
    있고 현실적 결함(측정 잡음·부분 커버리지(표면 일부만 터치)·소수 outlier(오독))을
    가진다.
  - **정합** : 조대(coarse) 초기정렬(무게중심 정렬)로 수렴 basin에 넣은 뒤 점-대-평면
    ICP로 SE(3)를 정밀 복원. 대응거리 게이트로 outlier를 배제한다.

**FRE vs TRE (임상 정확도 지표)**
  - FRE(Fiducial/surface Registration Error) : 정합에 쓴 표면 점들의 잔차 RMS. 정합이
    표면에서 얼마나 잘 맞았는지를 보지만, 정작 중요한 표적(종양)에서의 정확도는 아니다.
  - TRE(Target Registration Error) : 정합에 **쓰지 않은** 독립 표적(종양 위치·도구 팁
    목표)에서, 복원 변환과 진짜 변환이 얼마나 어긋나는지. **TRE가 임상의 실제 지표**다 —
    낮은 FRE가 낮은 TRE를 보장하지 않으며, 표적이 정합된 표면에서 멀수록(외삽) TRE가
    커진다. 이 실험은 표면에서 떨어진 심부 표적을 두어 그 성질을 드러낸다.

**수렴 basin/초기화 caveat**
  ICP는 국소 최적화라 초기 자세가 나쁘면 엉뚱한 국소최소로 빠진다(오정합). 임상에서
  이는 실제 위험이며, 보통 랜드마크 기반 조대정합이나 술자의 수동 정렬로 basin을
  확보한다. 이 실험은 좋은 초기값(무게중심 정렬)과 큰 회전오차 초기값을 비교해, 후자가
  낮은 표면잔차에도 불구하고 TRE가 폭발함(그럴듯하게 맞았지만 틀린 정합)을 보인다.

한계: 합성 해부(실제 환자·장기 변형 없음), 강체 가정(연조직 변형 미포함), 표면 곡률이
빈약한 부위(평평한 뼈)는 접선 미끄러짐으로 조건화가 나빠짐. TRE는 표적 위치 의존.

    python scripts/41_surgical_registration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.se3 import se3_inv, se3_log, so3_exp  # noqa: E402

# --------------------------------------------------------------------------- #
# 시나리오 상수 (단위: mm)
# --------------------------------------------------------------------------- #
ELL = (40.0, 32.0, 46.0)     # 타원체 캡 반경 (A,B,C) — 뼈/장기 돔
THETA_MAX = np.deg2rad(72.0)  # 캡 극각 상한
NOISE_MM = 0.5               # 수술 중 프로브 측정 잡음 σ
COVERAGE_PHI = np.deg2rad(255.0)  # 수술 중 터치되는 방위 범위(부분 커버리지)
N_INTRAOP = 240              # 수술 중 디지타이징 점 수
N_OUTLIERS = 6               # 소수 outlier(나쁜 프로브 오독)


# --------------------------------------------------------------------------- #
# SE(3) 유틸
# --------------------------------------------------------------------------- #
def apply_T(T, pts):
    """(N,3) 점군에 4x4 SE(3) 적용."""
    return pts @ T[:3, :3].T + T[:3, 3]


def pose_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_T(rotvec, t):
    """축-각 회전벡터 + 병진 → SE(3)."""
    return pose_T(so3_exp(np.asarray(rotvec, float)), np.asarray(t, float))


# --------------------------------------------------------------------------- #
# 합성 해부 표면 · 표적
# --------------------------------------------------------------------------- #
def bumpy_cap(theta, phi):
    """타원체 캡 표면점(융기 포함) — 뼈/장기처럼 알아볼 수 있고 비퇴화."""
    A, B, C = ELL
    st, ct = np.sin(theta), np.cos(theta)
    # 미세 융기: 정합을 헷갈리게 하지 않을 만큼 작지만 표면을 특징적으로 만듦
    bump = 1.0 + 0.06 * np.sin(3.0 * phi) * st + 0.05 * np.cos(2.0 * theta)
    x = A * st * np.cos(phi) * bump
    y = B * st * np.sin(phi) * bump
    z = C * ct * bump
    return np.stack([x, y, z], axis=-1)


def build_ct_model(n_theta=46, n_phi=90):
    """수술 전 CT 표면 모델 = 조밀 점군(영상 좌표계). (N,3) 반환."""
    th = np.linspace(0.02, THETA_MAX, n_theta)
    ph = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(th, ph)
    return bumpy_cap(TH.ravel(), PH.ravel())


def sample_intraop(rng, T_true, noise=NOISE_MM, coverage_phi=COVERAGE_PHI,
                   n=N_INTRAOP, n_outliers=N_OUTLIERS, phi0=None):
    """수술 중 디지타이징 점군(트래커 좌표계) 생성.

    같은 표면의 방위 [phi0, phi0+coverage_phi] 구획만 터치(부분 커버리지),
    미지의 SE(3) T_true로 옮긴 뒤 잡음·outlier 추가. (P, T_true) 반환."""
    if phi0 is None:
        phi0 = rng.uniform(0, 2 * np.pi)
    theta = np.sqrt(rng.uniform(0.02 ** 2, THETA_MAX ** 2, n))  # 면적 균등 근사
    phi = phi0 + rng.uniform(0.0, coverage_phi, n)
    surf = bumpy_cap(theta, phi)                       # CT(영상) 좌표계 표면점
    P = apply_T(T_true, surf)                          # → 트래커 좌표계
    P = P + rng.normal(0.0, noise, P.shape)            # 측정 잡음
    if n_outliers > 0:                                 # 나쁜 프로브 오독
        span = P.max(0) - P.min(0)
        out = P.mean(0) + rng.uniform(-1.0, 1.0, (n_outliers, 3)) * span * 0.9
        P = np.vstack([P, out])
    return P


def target_points():
    """정합에 쓰지 않는 독립 임상 표적(영상 좌표계): 심부 종양 + 도구 팁 목표.
    표면에서 떨어져 있어 TRE의 외삽 성질을 드러낸다."""
    return np.array([
        [0.0, 0.0, 8.0],      # 돔 심부 종양(표면 깊이 아래)
        [18.0, -10.0, 22.0],  # 도구 팁 목표 1
        [-15.0, 14.0, 18.0],  # 도구 팁 목표 2
        [5.0, 20.0, 30.0],    # 표면 근처 표적
    ])


# --------------------------------------------------------------------------- #
# 점-대-평면 ICP (exp 29 재사용: k-NN PCA 법선 + se(3) 선형화)
# --------------------------------------------------------------------------- #
def estimate_normals(pts, k=12):
    """국소 k-NN PCA 법선(공분산 최소고유벡터). (N,3)."""
    n = len(pts)
    if n < k + 1:
        return np.tile([0.0, 0.0, 1.0], (n, 1))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k)
    nb = pts[idx]
    c = nb - nb.mean(axis=1, keepdims=True)
    C = np.einsum("nki,nkj->nij", c, c) / k
    _, V = np.linalg.eigh(C)
    return V[:, :, 0]


def centroid_init(src, dst):
    """조대 초기정렬: 무게중심 정렬(회전=I). ICP 수렴 basin 확보용."""
    return pose_T(np.eye(3), dst.mean(0) - src.mean(0))


def point_to_plane_icp(src, dst, dst_normals=None, init=None,
                       max_iter=60, tol=1e-6, max_corr_dist=8.0):
    """src를 dst에 정렬하는 SE(3) 추정. 반환 (T, rms_plane, n_iter, inlier_mask).

    각 반복: 현재 T로 옮긴 src의 최근접 대응 → 점-대-평면 잔차 n·(t−q)를 se(3)
    증분 [ω, ν]로 선형화(자코비안 [q×n, n]), 정규방정식 풀어 좌곱 retraction.
    max_corr_dist 게이트가 대응거리 초과(=outlier)를 배제한다."""
    T = np.eye(4) if init is None else init.copy()
    tree = cKDTree(dst)
    if dst_normals is None:
        dst_normals = estimate_normals(dst)
    prev = np.inf
    it = 0
    mask = np.ones(len(src), bool)
    for it in range(1, max_iter + 1):
        q = apply_T(T, src)
        dist, idx = tree.query(q)
        m = dist < max_corr_dist
        if m.sum() < 12:
            break
        mask = m
        qm, tm, nm = q[m], dst[idx[m]], dst_normals[idx[m]]
        J = np.hstack([np.cross(qm, nm), nm])          # (M,6) [ω, ν]
        r = np.einsum("ij,ij->i", nm, tm - qm)
        A = J.T @ J
        b = J.T @ r
        delta = np.linalg.solve(A + 1e-9 * np.eye(6), b)
        T = pose_T(so3_exp(delta[:3]), delta[3:]) @ T
        err = float(np.sqrt(np.mean(r ** 2)))
        if abs(prev - err) < tol:
            prev = err
            break
        prev = err
    return T, prev, it, mask


# --------------------------------------------------------------------------- #
# 정합 파이프라인 + 임상 지표
# --------------------------------------------------------------------------- #
def register(model, model_normals, intraop, init=None, max_corr_dist=8.0):
    """수술 중 점군 → CT 모델 정합. 반환 (T_reg, surface_rmse, inlier_mask).

    T_reg: 트래커→영상 좌표 변환. surface_rmse(FRE): inlier 점의 표면 최근접 잔차 RMS."""
    if init is None:
        init = centroid_init(intraop, model)
    T_reg, _, _, mask = point_to_plane_icp(
        intraop, model, dst_normals=model_normals, init=init,
        max_corr_dist=max_corr_dist)
    tree = cKDTree(model)
    aligned = apply_T(T_reg, intraop)
    dist, _ = tree.query(aligned)
    surface_rmse = float(np.sqrt(np.mean(dist[mask] ** 2)))  # FRE
    return T_reg, surface_rmse, mask


def tre(T_reg, T_true, targets):
    """TRE: 표적을 진짜 변환으로 트래커 좌표에 옮긴 뒤 복원 변환으로 되돌렸을 때의 오차.
    완전 정합이면 T_reg∘T_true = I. (per-target 거리 배열) 반환."""
    comp = T_reg @ T_true                       # 이상적으로 항등
    mapped = apply_T(comp, targets)
    return np.linalg.norm(mapped - targets, axis=1)


# --------------------------------------------------------------------------- #
# 강건성 스윕
# --------------------------------------------------------------------------- #
def sweep_noise(model, model_normals, targets, levels, seed=7):
    rmse, tre_mean = [], []
    for s in levels:
        errs_r, errs_t = [], []
        for rep in range(4):
            rng = np.random.default_rng(seed + 100 * rep)
            T_true = make_T([0.10, -0.14, 0.22], [55.0, -38.0, 72.0])
            P = sample_intraop(rng, T_true, noise=s, phi0=0.3)
            T_reg, sr, _ = register(model, model_normals, P)
            errs_r.append(sr)
            errs_t.append(tre(T_reg, T_true, targets).mean())
        rmse.append(np.mean(errs_r))
        tre_mean.append(np.mean(errs_t))
    return np.array(rmse), np.array(tre_mean)


def sweep_coverage(model, model_normals, targets, fracs, seed=11):
    tre_mean = []
    for f in fracs:
        errs = []
        for rep in range(4):
            rng = np.random.default_rng(seed + 100 * rep)
            T_true = make_T([0.10, -0.14, 0.22], [55.0, -38.0, 72.0])
            P = sample_intraop(rng, T_true, coverage_phi=2 * np.pi * f, phi0=0.3)
            T_reg, _, _ = register(model, model_normals, P)
            errs.append(tre(T_reg, T_true, targets).mean())
        tre_mean.append(np.mean(errs))
    return np.array(tre_mean)


# --------------------------------------------------------------------------- #
def main(seed=3, plot=True):
    model = build_ct_model()
    model_normals = estimate_normals(model)
    targets = target_points()

    # --- 대표 정합: 현실적 결함 포함, 무게중심 조대정렬 → 점-대-평면 ICP ---
    rng = np.random.default_rng(seed)
    T_true = make_T([0.10, -0.14, 0.22], [55.0, -38.0, 72.0])  # 미지의 SE(3)
    intraop = sample_intraop(rng, T_true, phi0=0.3)

    T_reg, surface_rmse, mask = register(model, model_normals, intraop)
    tre_vals = tre(T_reg, T_true, targets)
    tre_mean, tre_max = float(tre_vals.mean()), float(tre_vals.max())

    # 복원한 SE(3)가 진짜 변환의 역과 얼마나 일치하나(정합 정확도 자체)
    resid = se3_log(T_reg @ T_true)
    trans_err = float(np.linalg.norm(resid[:3]))
    rot_err_deg = float(np.rad2deg(np.linalg.norm(resid[3:])))

    # --- 수렴 basin caveat: 큰 회전오차 초기값 → 오정합 ---
    # 융기의 sin(3φ) 근사대칭 때문에 ~143° 회전 초기값은 표면잔차(FRE)는 낮으나
    # 완전히 틀린 국소최소로 수렴한다(그럴듯하지만 오정합) → TRE로만 드러남.
    bad_init = make_T([0, 0, 2.5], [0, 0, 0]) @ centroid_init(intraop, model)
    T_bad, rmse_bad, _ = register(model, model_normals, intraop, init=bad_init)
    tre_bad = float(tre(T_bad, T_true, targets).mean())

    # --- outlier 강건성: 대응거리 게이트 유무 비교 ---
    T_gate, rmse_gate, _ = register(model, model_normals, intraop, max_corr_dist=6.0)
    tre_gate = float(tre(T_gate, T_true, targets).mean())
    T_nogate, rmse_nogate, _ = register(
        model, model_normals, intraop, max_corr_dist=1e9)
    tre_nogate = float(tre(T_nogate, T_true, targets).mean())

    # --- 강건성 스윕 ---
    noise_levels = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.8])
    rmse_curve, tre_curve = sweep_noise(model, model_normals, targets, noise_levels)
    cov_fracs = np.array([0.15, 0.25, 0.4, 0.55, 0.7, 0.9])
    tre_cov = sweep_coverage(model, model_normals, targets, cov_fracs)

    print("=== 41. 수술 환자-영상 정합 (patient-to-image via ICP) ===")
    print(f"CT 모델점 {len(model)}, 수술중 점 {len(intraop)}"
          f"(inlier {int(mask.sum())}, outlier {N_OUTLIERS}), "
          f"커버리지 {np.rad2deg(COVERAGE_PHI):.0f}°, 잡음 σ={NOISE_MM} mm")
    print(f"복원 SE(3) 오차 : 병진 {trans_err:.3f} mm, 회전 {rot_err_deg:.3f}°")
    print(f"표면 RMSE (FRE) : {surface_rmse:.3f} mm")
    print(f"TRE            : mean {tre_mean:.3f} mm, max {tre_max:.3f} mm  "
          f"(표적별 {np.array2string(tre_vals, precision=2)})")
    print(f"수렴 basin      : 좋은 초기값 TRE {tre_mean:.2f} mm  vs  "
          f"나쁜 초기값 TRE {tre_bad:.2f} mm (표면RMSE(FRE) {rmse_bad:.2f} mm — "
          f"낮아도 오정합)")
    print(f"outlier 강건성  : 게이트 TRE {tre_gate:.2f} mm  vs  "
          f"게이트無 TRE {tre_nogate:.2f} mm")

    if plot:
        _plot(model, intraop, T_reg, T_true, targets, tre_vals, mask,
              surface_rmse, tre_mean,
              noise_levels, rmse_curve, tre_curve, cov_fracs, tre_cov,
              tre_mean, tre_bad, tre_gate, tre_nogate)

    return surface_rmse, tre_mean, tre_max


# --------------------------------------------------------------------------- #
def _plot(model, intraop, T_reg, T_true, targets, tre_vals, mask,
          surface_rmse, tre_mean, noise_levels, rmse_curve, tre_curve,
          cov_fracs, tre_cov, tre_good, tre_bad, tre_gate, tre_nogate):
    fig = plt.figure(figsize=(16, 9))

    def style3d(ax):
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
        ax.view_init(elev=20, azim=-62)

    ms = model[::3]
    # (0,0) BEFORE — 두 좌표계가 분리돼 있음
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.scatter(ms[:, 0], ms[:, 1], ms[:, 2], c="0.7", s=3, alpha=0.5,
               label="preop CT surface")
    ax.scatter(intraop[:, 0], intraop[:, 1], intraop[:, 2], c="crimson", s=10,
               label="intraop points (tracker frame)")
    ax.set_title("BEFORE: patient vs image frames", fontsize=10)
    ax.legend(fontsize=7, loc="upper left"); style3d(ax)

    # (0,1) AFTER — 정합된 점군 + 표적/TRE
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    aligned = apply_T(T_reg, intraop)
    ax.scatter(ms[:, 0], ms[:, 1], ms[:, 2], c="0.7", s=3, alpha=0.5,
               label="preop CT surface")
    ax.scatter(aligned[mask, 0], aligned[mask, 1], aligned[mask, 2],
               c="seagreen", s=10, label="registered intraop pts")
    if (~mask).any():
        ax.scatter(aligned[~mask, 0], aligned[~mask, 1], aligned[~mask, 2],
                   c="orange", marker="x", s=30, label="rejected outliers")
    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c="blue",
               marker="*", s=90, label="targets (TRE)")
    ax.set_title(f"AFTER: registered  (surface RMSE {surface_rmse:.2f} mm, "
                 f"TRE {tre_mean:.2f} mm)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left"); style3d(ax)

    # (0,2) 수렴 basin caveat
    ax = fig.add_subplot(2, 3, 3)
    ax.bar(["good init\n(centroid)", "bad init\n(large rot)"],
           [tre_good, tre_bad], color=["seagreen", "crimson"])
    ax.set_ylabel("mean TRE [mm]")
    ax.set_title("Convergence-basin caveat\n(bad init → wrong local min)",
                 fontsize=10)
    for i, v in enumerate([tre_good, tre_bad]):
        ax.text(i, v, f" {v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # (1,0) 잡음 스윕
    ax = fig.add_subplot(2, 3, 4)
    ax.plot(noise_levels, rmse_curve, "o-", color="0.4", label="surface RMSE (FRE)")
    ax.plot(noise_levels, tre_curve, "s-", color="blue", label="mean TRE")
    ax.set_xlabel("measurement noise σ [mm]"); ax.set_ylabel("error [mm]")
    ax.set_title("Robustness vs measurement noise", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,1) 커버리지 스윕
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(cov_fracs * 100, tre_cov, "^-", color="purple")
    ax.set_xlabel("intraop surface coverage [%]"); ax.set_ylabel("mean TRE [mm]")
    ax.set_title("Robustness vs surface coverage\n(less touch → weaker "
                 "conditioning)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # (1,2) outlier 게이트 강건성
    ax = fig.add_subplot(2, 3, 6)
    ax.bar(["dist-gate\n(reject)", "no gate"],
           [tre_gate, tre_nogate], color=["seagreen", "crimson"])
    ax.set_ylabel("mean TRE [mm]")
    ax.set_title(f"Outlier robustness ({N_OUTLIERS} bad reads)", fontsize=10)
    for i, v in enumerate([tre_gate, tre_nogate]):
        ax.text(i, v, f" {v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("41. Surgical registration: patient-to-image (intraop→preop CT) "
                 "rigid alignment via point-to-plane ICP", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "41_surgical_registration.png", dpi=130)
    plt.close(fig)
    print("\n[plot] outputs/41_surgical_registration.png, "
          "assets/41_surgical_registration.png")


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 강체 가정: 실제 연조직·장기는 변형되어 강체 정합만으로 부족(변형 정합/생체역학
#     모델 필요). 뼈처럼 강체에 가까운 구조가 이 방법에 가장 적합.
#   - 조건화: 곡률이 세 축을 덮는 부위(돔·모서리)는 6-DOF가 잘 관측되나, 평평한 부위만
#     터치하면 접선 미끄러짐으로 해가 병약. 커버리지 스윕이 이 성질을 보인다.
#   - FRE↛TRE: 표면잔차(FRE)가 낮아도 표적이 정합 영역에서 멀면(외삽) TRE가 커진다.
#     basin caveat 실험은 낮은 FRE의 오정합조차 가능함을 보여준다 → 임상에서 TRE로 검증.
#   - 초기화: ICP는 국소법이라 랜드마크 조대정합/술자 정렬로 basin 확보가 필수.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
