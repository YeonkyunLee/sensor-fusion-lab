"""영상유도 도구 유도 캡스톤: 정합 → 계획 → 기구학 → 동역학 제어의 오차 예산.

exp 39(기구학)·40(동역학/제어)·41(환자-영상 정합)은 각각 한 조각씩만 다뤘다. 실제
영상유도 수술 로봇은 그 조각들이 **직렬로 연결된 사슬**이며, 환자에게 전달되는 것은
사슬 끝의 단 하나의 수치 — **도구 팁이 계획 표적에서 실제로 얼마나 벗어났는가**다.
이 실험은 네 단계를 하나의 파이프라인으로 잇고, 최종 오차를 단계별로 분해한다.

    [수술 전] CT 슬라이스 영상좌표계에서 표적(종양)·금지구조물(혈관)·진입경로 계획
        ↓  (환자가 미지의 SE(2) 자세로 수술대에 놓임)
    [정합]   로봇이 표면 일부를 프로브로 디지타이징 → 점-대-법선 ICP로 robot→image 복원
        ↓
    [계획]   영상좌표 계획을 정합으로 로봇좌표로 옮김 + 금지구조물 여유 검사
        ↓
    [기구학] 카테시안 경로 → DLS IK로 관절궤적 (exp 39)
        ↓
    [제어]   2링크 팔 동역학을 PD / 계산토크로 추종 (exp 40)
        ↓
    [결과]   실제 도구 팁 vs 진짜 표적 = 엔드투엔드 표적오차

--- 왜 오차 '예산'인가 ---
사슬의 각 단계가 오차를 보태므로, 어느 항이 지배적인지를 모르면 엉뚱한 곳을 최적화한다.
네 조건을 같은 지표(엔드투엔드 표적오차)로 비교한다.

  (A) 정합 없음 + 계산토크   : 환자가 공칭 자세에 있다고 가정 → 자세 불일치가 그대로 오차
  (B) 정합 + PD 제어         : 좌표는 맞지만 서보가 처짐/지연으로 못 따라감
  (C) 정합 + 계산토크        : 실제 임상 구성
  (D) 완전정합(오라클)+계산토크: 정합오차 0일 때의 바닥 = 기구학/서보 잔차

(C)와 (D)의 차이가 **정합이 차지하는 몫**, (C)와 (B)의 차이가 **서보가 차지하는 몫**이다.

--- 안전: 불확실도를 아는 계획 (exp 9의 계승) ---
정합은 점추정만 주는 게 아니라 **공분산**도 준다. ICP 정규방정식의 A = JᵀJ에서
Cov(ξ) ≈ σ²A⁻¹ (ξ = se(2) 증분 [ω, νx, νy])를 얻고, 표적점 야코비안으로 전파하면
표적 위치의 불확실도 σ_target 이 나온다. 표면 커버리지가 나쁘면(접선 미끄러짐) 이
공분산이 즉시 커진다 — 즉 **정합이 못 믿을 상황임을 정합 자신이 알려준다.**

몬테카를로로 두 계획 규칙을 비교한다(커버리지·잡음·환자자세 무작위):
  - naive           : 점추정만 믿고 항상 집행 → 나쁜 정합에서 혈관 침범
  - uncertainty-aware: 필요여유 = r_vessel + k·σ_target. 못 지키면 **중단(재프로브)**

    python scripts/42_image_guided_targeting.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
kin = import_module("39_manipulator_kinematics")     # FK · J · DLS IK
dyn = import_module("40_manipulator_dynamics")       # M/C/g · 제어기 · 순동역학

# --------------------------------------------------------------------------- #
# 시나리오 상수 (SI: 길이 m, 각 rad, 시간 s / 결과 출력은 mm)
# --------------------------------------------------------------------------- #
L_ARM = np.array([0.35, 0.30])          # 2링크 팔 링크길이 [m] (리치 0.65 m)
ARM = dyn.ArmParams(m1=3.0, m2=2.0, l1=L_ARM[0], l2=L_ARM[1],
                    lc1=L_ARM[0] / 2, lc2=L_ARM[1] / 2,
                    I1=3.0 * L_ARM[0] ** 2 / 12, I2=2.0 * L_ARM[1] ** 2 / 12)
# 제어기가 '아는' 모델은 실제와 3% 어긋난다(도구 장착·캘리브레이션 오차). 이 불일치가
# 계산토크의 오차 바닥을 만든다 — 완벽한 모델을 가정하면 바닥이 0이라 예산 비교가 무의미.
MASS_ERR = 1.03
ARM_CTRL = dyn.ArmParams(m1=3.0 * MASS_ERR, m2=2.0 * MASS_ERR,
                         l1=L_ARM[0], l2=L_ARM[1],
                         lc1=L_ARM[0] / 2, lc2=L_ARM[1] / 2,
                         I1=3.0 * MASS_ERR * L_ARM[0] ** 2 / 12,
                         I2=2.0 * MASS_ERR * L_ARM[1] ** 2 / 12)

ELL = (0.055, 0.040)                    # 해부 단면(타원) 반경 [m]
TUMOR_IMG = np.array([0.006, -0.010])   # 표적(종양), 영상좌표계
VESSEL_IMG = np.array([-0.004, 0.008])  # 금지구조물(혈관) 중심, 영상좌표계
VESSEL_R = 0.004                        # 혈관 반경 [m] (4 mm)
# 수술 노출(exposure)로 진입 가능한 표면 구획. 임상에서 진입점은 자유롭지 않으며,
# 이 창 안에서 가능한 최선의 경로도 혈관 옆을 좁게 지난다 = 마진이 중요한 상황.
ACCESS_WINDOW = (np.deg2rad(106.0), np.deg2rad(134.0))
MISS_TOL = 0.002                        # 표적 도달 허용오차 [m] (2 mm) — 초과 시 실패
DISAGREE_TOL = 0.001                    # 다중초기값 해 불일치 허용치 [m] (1 mm)
RIVAL_RATIO = 1.15                      # '데이터로 구분 못 하는 경쟁 해'로 볼 잔차 비율

PROBE_NOISE = 5e-4                      # 프로브 측정 잡음 σ [m] (0.5 mm)
N_PROBE = 200                           # 디지타이징 점 수
N_OUTLIERS = 4                          # 나쁜 프로브 오독
COVERAGE_MAIN = np.deg2rad(230.0)       # 메인 시나리오 커버리지

# 환자 공칭 자세(수술 전 계획이 가정한 배치) vs 실제 자세
T_NOM_ANGLE, T_NOM_XY = np.deg2rad(4.0), np.array([0.42, 0.055])
T_TRUE_ANGLE, T_TRUE_XY = np.deg2rad(11.0), np.array([0.431, 0.043])

KP, KD = 400.0, 40.0                    # 세 제어기 공통 이득
DT = 0.002                              # 제어/적분 주기 [s]
T_APPROACH, T_INSERT, T_HOLD = 1.6, 1.6, 0.4   # 접근·삽입·정지유지 [s]
K_SIGMA = 3.0                           # 안전 마진 계수 (k·σ)


# --------------------------------------------------------------------------- #
# SE(2) 유틸
# --------------------------------------------------------------------------- #
def se2(theta, t):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(3)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 2] = t
    return T


def se2_inv(T):
    R = T[:2, :2]
    Ti = np.eye(3)
    Ti[:2, :2] = R.T
    Ti[:2, 2] = -R.T @ T[:2, 2]
    return Ti


def apply_se2(T, pts):
    """(N,2) 또는 (2,) 점에 SE(2) 적용."""
    p = np.atleast_2d(np.asarray(pts, float))
    out = p @ T[:2, :2].T + T[:2, 2]
    return out[0] if np.ndim(pts) == 1 else out


def perp(p):
    """회전 생성자: d/dω [R(ω)p]|_0 = [-p_y, p_x]."""
    return np.stack([-p[..., 1], p[..., 0]], axis=-1)


# --------------------------------------------------------------------------- #
# 해부 단면(영상좌표계): 융기가 있는 닫힌 곡선 + 법선
# --------------------------------------------------------------------------- #
def anatomy_curve(phi):
    """타원 + 미세 융기. 곡률이 방향마다 달라 2-DOF 회전·병진이 관측 가능."""
    A, B = ELL
    bump = 1.0 + 0.07 * np.sin(3.0 * phi) + 0.04 * np.cos(2.0 * phi)
    return np.stack([A * np.cos(phi) * bump, B * np.sin(phi) * bump], axis=-1)


def build_model(n=720):
    """수술 전 CT 모델 = 조밀 경계 점군 + 외향 법선(접선을 90° 회전)."""
    phi = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    pts = anatomy_curve(phi)
    tang = np.gradient(pts, axis=0)                    # 닫힌 곡선의 수치 접선
    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    outward = np.einsum("ij,ij->i", nrm, pts) < 0      # 바깥을 향하도록 부호 정리
    nrm[outward] *= -1
    return pts, nrm


def sample_probe(rng, T_true, coverage, n=N_PROBE, noise=PROBE_NOISE,
                 n_out=N_OUTLIERS, phi0=None):
    """수술 중 프로브 디지타이징(로봇/트래커 좌표계): 부분 커버리지 + 잡음 + outlier."""
    if phi0 is None:
        phi0 = rng.uniform(0, 2 * np.pi)
    phi = phi0 + rng.uniform(0.0, coverage, n)
    P = apply_se2(T_true, anatomy_curve(phi)) + rng.normal(0, noise, (n, 2))
    if n_out > 0:
        span = P.max(0) - P.min(0)
        P = np.vstack([P, P.mean(0) + rng.uniform(-1, 1, (n_out, 2)) * span * 0.8])
    return P


# --------------------------------------------------------------------------- #
# 점-대-법선 ICP (SE(2)) — exp 41의 3D 점-대-평면을 평면 슬라이스로
# --------------------------------------------------------------------------- #
def point_to_normal_icp(src, model, normals, init=None, max_iter=60,
                        tol=1e-9, max_corr=0.012):
    """src(로봇좌표 프로브점) → model(영상좌표) 정렬. 반환 (T, rms, A, mask).

    잔차 r_i = n_i·(m_i − p_i), 좌곱 증분 ξ=[ω,νx,νy]에 대한 야코비안
    J_i = [n_i·perp(p_i), n_ix, n_iy]. A=JᵀJ 는 정보행렬(공분산 산출에 재사용).
    max_corr 게이트가 outlier 대응을 배제한다."""
    T = np.eye(3) if init is None else init.copy()
    tree = cKDTree(model)
    prev, A = np.inf, np.eye(3)
    mask = np.ones(len(src), bool)
    for _ in range(max_iter):
        p = apply_se2(T, src)
        dist, idx = tree.query(p)
        m = dist < max_corr
        if m.sum() < 12:
            break
        mask = m
        pm, mm, nm = p[m], model[idx[m]], normals[idx[m]]
        J = np.column_stack([np.einsum("ij,ij->i", nm, perp(pm)), nm])
        r = np.einsum("ij,ij->i", nm, mm - pm)
        A = J.T @ J
        xi = np.linalg.solve(A + 1e-12 * np.eye(3), J.T @ r)
        T = se2(xi[0], xi[1:]) @ T
        rms = float(np.sqrt(np.mean(r ** 2)))
        if abs(prev - rms) < tol:
            prev = rms
            break
        prev = rms
    return T, prev, A, mask


def coarse_init(src, model):
    """조대 초기정렬(무게중심). ICP는 국소법이라 수렴 basin 확보가 필수."""
    return se2(0.0, model.mean(0) - src.mean(0))


def _init_candidates(src, model, angles=(0.0, np.pi / 2, np.pi, 3 * np.pi / 2)):
    """무게중심 정렬 + 90° 간격 회전 = 다중 초기값. 서로 다른 basin을 탐색한다."""
    c_s, c_m = src.mean(0), model.mean(0)
    out = []
    for a in angles:
        R = se2(a, np.zeros(2))[:2, :2]
        out.append(se2(a, c_m - R @ c_s))
    return out


def _icp_two_stage(probe, model, normals, init):
    """조대→정밀: 넓은 대응게이트로 basin에 들어간 뒤 게이트를 좁혀 outlier를 떨군다."""
    T0, _, _, _ = point_to_normal_icp(probe, model, normals, init=init, max_corr=0.012)
    return point_to_normal_icp(probe, model, normals, init=T0, max_corr=0.003)


def register(probe, model, normals, verify_target=TUMOR_IMG, rival_ratio=RIVAL_RATIO):
    """정합 + 두 종류의 신뢰도 지표. 반환 (T_reg: robot→image, fre, Cov_xi, disagree).

    - Cov(ξ) ≈ σ² A⁻¹ (σ = 잔차 RMS, 측정잡음이 바닥). **조건화**가 나쁠 때 커진다.
    - disagree: 여러 초기값에서 수렴한 해들 중 잔차가 비슷한(=똑같이 그럴듯한) 것들이
      표적을 서로 얼마나 다른 곳으로 보내는가. **잘못된 basin**을 잡는 지표로,
      공분산이 못 보는 '작은 잔차로 확신에 차서 틀린' 정합을 드러낸다.

    ICP가 아예 실패하면 '전혀 못 믿음'을 뜻하는 거대 공분산을 돌려준다."""
    huge = np.eye(3) * 1e6
    # 주 해(primary)는 무게중심 초기화 — 임상 워크플로(술자 조대정렬)에 해당한다.
    # 나머지 초기값은 '해를 고르는' 데 쓰지 않는다: 부분 커버리지에서는 표면을 따라
    # 미끄러진 오정합이 더 낮은 잔차를 낼 수 있어, 최소잔차 선택이 오히려 해롭다.
    T_reg, rms, A, _ = _icp_two_stage(probe, model, normals, coarse_init(probe, model))
    if not np.isfinite(rms) or not np.all(np.isfinite(A)):
        return T_reg, np.inf, huge, np.inf

    # 검증 전용: 다른 초기값에서 출발해 '잔차가 비슷한'(=데이터로 구분 못 하는) 해가
    # 표적을 다른 곳으로 보내면, 이 정합은 basin을 잘못 잡았을 수 있다.
    best_p = apply_se2(se2_inv(T_reg), verify_target)
    disagree = 0.0
    for init in _init_candidates(probe, model)[1:]:
        T_r, rms_r, A_r, _ = _icp_two_stage(probe, model, normals, init)
        if not np.isfinite(rms_r) or rms_r > rival_ratio * max(rms, PROBE_NOISE):
            continue
        d = float(np.linalg.norm(apply_se2(se2_inv(T_r), verify_target) - best_p))
        disagree = max(disagree, d)

    sigma = max(rms, PROBE_NOISE)
    try:
        Cov = sigma ** 2 * np.linalg.inv(A + 1e-12 * np.eye(3))
    except np.linalg.LinAlgError:
        return T_reg, rms, huge, disagree
    if not np.all(np.isfinite(Cov)):
        return T_reg, rms, huge, disagree
    return T_reg, rms, Cov, disagree


def target_sigma(Cov, p_img):
    """표적점의 위치 불확실도(최대 주축 1σ, m). δp = [perp(p), I] ξ."""
    Jp = np.column_stack([perp(np.asarray(p_img, float)), np.eye(2)])
    C = Jp @ Cov @ Jp.T
    return float(np.sqrt(max(np.linalg.eigvalsh(C)[-1], 0.0)))


# --------------------------------------------------------------------------- #
# 계획: 진입점 선택 + 직선 삽입경로 + 금지구조물 여유
# --------------------------------------------------------------------------- #
def segment_clearance(a, b, c, n=120):
    """선분 a→b 와 점 c 사이 최소거리 (샘플링; 경로 여유 계산용)."""
    s = np.linspace(0, 1, n)[:, None]
    pts = a + s * (b - a)
    return float(np.min(np.linalg.norm(pts - c, axis=1)))


def plan_entry(window=ACCESS_WINDOW, target=TUMOR_IMG, vessel=VESSEL_IMG, n=61):
    """수술 노출 창 안에서 혈관 여유가 가장 큰 진입점을 고른다.

    반환 (진입점, 여유[m]). 창이 좁으면 '최선의 경로'조차 여유가 작다 — 이때
    정합오차 몇 mm가 곧바로 침범이 되므로 마진 규칙이 결정적이 된다."""
    phis = np.linspace(window[0], window[1], n)
    best = None
    for phi in phis:
        entry = anatomy_curve(np.array(phi))
        clr = segment_clearance(entry, target, vessel) - VESSEL_R
        if best is None or clr > best[1]:
            best = (entry, float(clr))
    return best


# --------------------------------------------------------------------------- #
# 카테시안 경로 → 관절궤적 (exp 39의 DLS IK)
# --------------------------------------------------------------------------- #
def smoothstep(u):
    """5차 시간스케일링(속도·가속도 0에서 시작·종료) — 매끄러운 관절궤적."""
    return 6 * u ** 5 - 15 * u ** 4 + 10 * u ** 3


def cartesian_path(entry_rob, target_rob, standoff=0.035):
    """접근점(진입점에서 표적 반대방향으로 standoff) → 진입점 → 표적, 시간 스케일링."""
    d = entry_rob - target_rob
    d = d / (np.linalg.norm(d) + 1e-12)
    approach = entry_rob + standoff * d

    n_app = int(round(T_APPROACH / DT))
    n_ins = int(round(T_INSERT / DT))
    n_hold = int(round(T_HOLD / DT))
    u1 = smoothstep(np.linspace(0, 1, n_app)[:, None])
    u2 = smoothstep(np.linspace(0, 1, n_ins)[:, None])
    seg1 = approach + u1 * (entry_rob - approach)      # 자유공간 접근
    seg2 = entry_rob + u2 * (target_rob - entry_rob)   # 삽입
    seg3 = np.repeat(target_rob[None, :], n_hold, axis=0)
    path = np.vstack([seg1, seg2, seg3])
    t = np.arange(len(path)) * DT
    return t, path, (n_app, n_ins, n_hold)


def path_to_joints(path, q0=None):
    """경로점마다 DLS IK(warm start). 반환 (q_des(N,2), 최대 IK 잔차[m])."""
    q = np.array([0.9, -1.4]) if q0 is None else np.array(q0, float)
    Q = np.zeros((len(path), 2))
    worst = 0.0
    for k, p in enumerate(path):
        q, _, _, err = kin.ik_dls(p, q, L=L_ARM, lam=1e-3, max_iters=60,
                                  tol=1e-9, step_cap=0.15)
        Q[k] = q
        worst = max(worst, err)
    return Q, worst


def joint_derivatives(Q, dt=DT):
    """중심차분으로 q̇_d, q̈_d (계산토크 피드포워드용)."""
    Qd = np.gradient(Q, dt, axis=0)
    Qdd = np.gradient(Qd, dt, axis=0)
    return Qd, Qdd


# --------------------------------------------------------------------------- #
# 실행: 관절궤적을 실제 팔로 추종 (exp 40의 동역학·제어기)
# --------------------------------------------------------------------------- #
def track(Q_des, Qd_des, Qdd_des, controller, p_true=ARM, p_ctrl=None):
    """RK4 고정스텝 추종. 반환 (q(N,2), 팁위치(N,2))."""
    if p_ctrl is None:
        p_ctrl = p_true
    q = Q_des[0].copy()
    qd = Qd_des[0].copy()
    Q = np.zeros_like(Q_des)
    TIP = np.zeros_like(Q_des)
    for k in range(len(Q_des)):
        e = Q_des[k] - q
        edot = Qd_des[k] - qd
        tau = controller(q, qd, Qd_des[k], Qdd_des[k], qd, e, edot, KP, KD, p_ctrl)
        Q[k] = q
        TIP[k] = kin.fk(q, L=L_ARM)[:2]
        if k == len(Q_des) - 1:
            break

        def deriv(s):
            return np.concatenate([s[2:], dyn.forward_dynamics(s[:2], s[2:], tau, p_true)])

        s = np.concatenate([q, qd])
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * DT * k1)
        k3 = deriv(s + 0.5 * DT * k2)
        k4 = deriv(s + DT * k3)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q, qd = s[:2], s[2:]
    return Q, TIP


def run_condition(entry_img, target_img, T_map, T_true, controller, p_ctrl=ARM):
    """한 조건 실행. T_map = 시스템이 '믿는' image→robot 변환, p_ctrl = 제어기의 모델.

    반환 dict: 최종 팁위치, 엔드투엔드 표적오차, 서보오차, 정합유발 표적오차."""
    entry_rob = apply_se2(T_map, entry_img)
    target_cmd = apply_se2(T_map, target_img)          # 시스템이 겨냥하는 점
    target_true = apply_se2(T_true, target_img)        # 실제 표적 위치

    t, path, phases = cartesian_path(entry_rob, target_cmd)
    Q_des, ik_worst = path_to_joints(path)
    Qd_des, Qdd_des = joint_derivatives(Q_des)
    Q, TIP = track(Q_des, Qd_des, Qdd_des, controller, p_true=ARM, p_ctrl=p_ctrl)

    tip_final = TIP[-1]
    tip_err = np.linalg.norm(TIP - path, axis=1)
    n_app, n_ins, _ = phases
    return dict(t=t, path=path, tip=TIP, q=Q, phases=phases,
                tip_final=tip_final,
                total=float(np.linalg.norm(tip_final - target_true)),
                servo=float(np.linalg.norm(tip_final - target_cmd)),
                reg=float(np.linalg.norm(target_cmd - target_true)),
                ik_worst=ik_worst,
                tip_err=tip_err,
                # 삽입 구간의 최대 경로이탈 — 혈관 옆을 지날 때 실제로 위험한 양
                insert_dev=float(np.max(tip_err[n_app:n_app + n_ins])))


# --------------------------------------------------------------------------- #
# 몬테카를로: 불확실도-인지 계획이 금지구조물 침범을 막는가
# --------------------------------------------------------------------------- #
def safety_mc(model, normals, n_trials=200, seed=11, k_sigma=K_SIGMA,
              use_disagree=True, disagree_tol=DISAGREE_TOL):
    """커버리지·자세·잡음을 무작위화. naive(항상 집행) vs aware(k·σ 마진, 미달시 중단).

    **unsafe = 혈관 침범 OR 표적 미달(TRE > MISS_TOL)**. 침범만 세면 정합이 크게
    틀어져 경로가 해부구조 밖으로 나간 '엉뚱한 곳을 찌른' 케이스가 안전으로 잡히는
    함정이 있어, 전달 실패까지 포함해 판정한다. aware가 중단한 시도는 '재프로브'
    (커버리지를 넓혀 다시 디지타이징)로 회복되는지도 센다."""
    entry_img, nominal_clear = plan_entry()

    res = dict(naive_unsafe=0, naive_viol=0, naive_miss=0, false_alarm=0,
               aware_unsafe=0, aborted=0, executed=0, recovered=0, caught=0,
               tre=[], sigma=[], clear=[], gated=[], nominal_clear=nominal_clear)
    for i in range(n_trials):
        # 시행별 독립 난수: 재프로브는 게이트가 중단할 때만 실행되므로, 하나의 스트림을
        # 공유하면 게이트 설정이 '이후 시행의 환자'까지 바꿔 naive/aware 비교가 오염된다.
        rng = np.random.default_rng([seed, i])
        rng_re = np.random.default_rng([seed, i, 99])
        cov = rng.uniform(np.deg2rad(70), np.deg2rad(260))     # 커버리지 품질 다양
        T_true = se2(rng.uniform(-0.20, 0.20),
                     T_NOM_XY + rng.uniform(-0.015, 0.015, 2))
        probe = sample_probe(rng, T_true, cov, noise=PROBE_NOISE * rng.uniform(0.6, 1.6))
        T_reg, _, Cov, disagree = register(probe, model, normals)
        T_map = se2_inv(T_reg)                                  # image→robot

        # 집행 경로를 '환자 기준'으로 되돌려 혈관과의 실제 여유를 계산
        back = se2_inv(T_true) @ T_map
        e_pat, t_pat = apply_se2(back, entry_img), apply_se2(back, TUMOR_IMG)
        actual_clear = segment_clearance(e_pat, t_pat, VESSEL_IMG) - VESSEL_R
        tre = float(np.linalg.norm(apply_se2(back, TUMOR_IMG) - TUMOR_IMG))

        viol = actual_clear < 0                     # 혈관 침범
        miss = tre > MISS_TOL                       # 표적 전달 실패
        unsafe = viol or miss

        sig = target_sigma(Cov, TUMOR_IMG)
        # 두 갈래 신뢰도 게이트: (조건화) k·σ 마진 + (basin) 다중초기값 해 불일치
        abort_sigma = nominal_clear - k_sigma * sig < 0
        abort_disagree = use_disagree and disagree > disagree_tol
        abort = abort_sigma or abort_disagree
        res["by_sigma"] = res.get("by_sigma", 0) + int(abort_sigma)
        res["by_disagree"] = res.get("by_disagree", 0) + int(abort_disagree and not abort_sigma)
        res.setdefault("disagree", []).append(disagree)
        res["tre"].append(tre)
        res["sigma"].append(sig)
        res["clear"].append(actual_clear)
        res["gated"].append(abort)
        res.setdefault("unsafe", []).append(unsafe)

        res["naive_viol"] += int(viol)
        res["naive_miss"] += int(miss)
        res["naive_unsafe"] += int(unsafe)
        res["caught"] += int(unsafe and abort)      # 검출력(sensitivity)
        res["false_alarm"] += int((not unsafe) and abort)   # 멀쩡한 계획을 막은 대가

        if abort:
            res["aborted"] += 1
            probe2 = sample_probe(rng_re, T_true, np.deg2rad(330),
                                  noise=PROBE_NOISE)            # 재프로브(커버리지 확대)
            T_reg2, _, Cov2, dis2 = register(probe2, model, normals)
            sig2 = target_sigma(Cov2, TUMOR_IMG)
            back2 = se2_inv(T_true) @ se2_inv(T_reg2)
            e2, t2 = apply_se2(back2, entry_img), apply_se2(back2, TUMOR_IMG)
            clear2 = segment_clearance(e2, t2, VESSEL_IMG) - VESSEL_R
            tre2 = float(np.linalg.norm(t2 - TUMOR_IMG))
            passes = (nominal_clear - k_sigma * sig2 >= 0) and \
                (not use_disagree or dis2 <= disagree_tol)
            # 회복 = 재프로브 결과가 게이트를 통과하고, 실제로도 안전하게 집행됨
            if passes and clear2 >= 0 and tre2 <= MISS_TOL:
                res["recovered"] += 1
        else:
            res["executed"] += 1
            res["aware_unsafe"] += int(unsafe)
    for k in ("tre", "sigma", "clear", "gated", "unsafe", "disagree"):
        res[k] = np.array(res[k])
    res["n_trials"] = n_trials
    return res


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(3)
    model, normals = build_model()

    T_true = se2(T_TRUE_ANGLE, T_TRUE_XY)      # image→robot (진짜)
    T_nom = se2(T_NOM_ANGLE, T_NOM_XY)         # image→robot (수술 전 가정)

    # ---- (1) 정합 ----
    probe = sample_probe(rng, T_true, COVERAGE_MAIN, phi0=np.deg2rad(35))
    T_reg, fre, Cov, disagree = register(probe, model, normals)
    T_map = se2_inv(T_reg)                     # image→robot (정합 추정)
    sigma_t = target_sigma(Cov, TUMOR_IMG)
    tre_reg = float(np.linalg.norm(apply_se2(T_map, TUMOR_IMG)
                                   - apply_se2(T_true, TUMOR_IMG)))
    place_err = float(np.linalg.norm(apply_se2(T_nom, TUMOR_IMG)
                                     - apply_se2(T_true, TUMOR_IMG)))

    # ---- (2) 계획: 노출 창 안에서 혈관 여유가 가장 큰 진입점 ----
    entry_img, nominal_clear = plan_entry()

    # ---- (3~4) 네 조건 실행: 정합/제어 조합 ----
    ct, pd = dyn.ctrl_computed_torque, dyn.ctrl_pd
    conds = {
        "A no registration + computed torque":       (T_nom, ct, ARM),
        "B registration + PD":                       (T_map, pd, ARM),
        "C registration + CT (3% payload error)":    (T_map, ct, ARM_CTRL),
        "D registration + CT (calibrated)":          (T_map, ct, ARM),
        "E oracle registration + CT (floor)":        (T_true, ct, ARM),
    }
    out = {name: run_condition(entry_img, TUMOR_IMG, T_map_i, T_true, ctrl, p_ctrl_i)
           for name, (T_map_i, ctrl, p_ctrl_i) in conds.items()}

    # ---- (5) 안전 몬테카를로 + 게이트 절제(어느 신호가 실제로 일하는가) ----
    mc = safety_mc(model, normals)
    mc_sig = safety_mc(model, normals, use_disagree=False)

    # ---- 출력 ----
    print("=== 42. 영상유도 도구 유도 캡스톤 (정합 → 계획 → IK → 동역학 제어) ===")
    print(f"팔 리치 {L_ARM.sum()*1e3:.0f} mm, 해부 단면 {ELL[0]*2e3:.0f}×{ELL[1]*2e3:.0f} mm, "
          f"프로브 {len(probe)}점(커버리지 {np.rad2deg(COVERAGE_MAIN):.0f}°, "
          f"σ={PROBE_NOISE*1e3:.1f} mm, outlier {N_OUTLIERS})")
    print(f"[정합] FRE {fre*1e3:.3f} mm, 표적 TRE {tre_reg*1e3:.3f} mm, "
          f"예측 불확실도 σ_target {sigma_t*1e3:.3f} mm, 다중초기값 불일치 "
          f"{disagree*1e3:.3f} mm  (공칭배치 가정 시 표적오차 {place_err*1e3:.1f} mm)")
    print(f"[계획] 진입점 {np.round(entry_img*1e3, 1).tolist()} mm(영상좌표), "
          f"혈관 여유 {nominal_clear*1e3:.1f} mm")
    print("-" * 78)
    print(f"{'condition':42s} {'total[mm]':>10s} {'reg[mm]':>9s} {'servo[mm]':>10s}")
    for name, r in out.items():
        print(f"{name:42s} {r['total']*1e3:10.3f} {r['reg']*1e3:9.3f} {r['servo']*1e3:10.3f}")
    print("-" * 78)
    cA = out["A no registration + computed torque"]
    cB = out["B registration + PD"]
    cC = out["C registration + CT (3% payload error)"]
    cD = out["D registration + CT (calibrated)"]
    cE = out["E oracle registration + CT (floor)"]
    print(f"정합의 기여        : {cA['total']/cD['total']:.0f}배 (A→D)")
    print(f"모델기반 제어 기여  : {cB['total']/cD['total']:.0f}배 (B→D)")
    print(f"[예산 순위는 캘리브레이션 상태에 따라 뒤집힌다]")
    print(f"  보정된 팔(D) : 총 {cD['total']*1e6:.0f} µm = 정합 {cD['reg']*1e6:.0f} µm "
          f"+ 서보 {cD['servo']*1e6:.3g} µm → 사실상 전부 정합 몫 "
          "= 다음 투자는 서보가 아니라 트래커/커버리지")
    print(f"  3% 페이로드 오차(C) : 총 {cC['total']*1e3:.3f} mm, 서보 {cC['servo']*1e3:.3f} mm "
          f"→ 정합오차의 {cC['servo']/cD['reg']:.0f}배 = 이 상태에선 캘리브레이션이 1순위")
    print(f"오차 바닥(E, 완전정합+보정): 정착오차 {cE['total']*1e6:.2g} µm(수치한계) / "
          f"삽입구간 최대 경로이탈 {cE['insert_dev']*1e6:.0f} µm ← 사슬이 만들 수 있는 최선")
    print(f"IK 최대 잔차       : {cD['ik_worst']*1e6:.2f} µm (무시가능)")
    print("-" * 78)
    nv = 100 * mc["naive_unsafe"] / mc["n_trials"]
    av = 100 * mc["aware_unsafe"] / max(mc["executed"], 1)
    print(f"[안전 MC {mc['n_trials']}회] 노출 창 내 최선 경로의 계획 여유 "
          f"{mc['nominal_clear']*1e3:.1f} mm, TRE 중앙값 {np.median(mc['tre'])*1e3:.2f} mm / "
          f"최악 {mc['tre'].max()*1e3:.0f} mm (커버리지 70~260° 무작위)")
    print(f"  naive(항상 집행)         : unsafe {nv:.1f}% ({mc['naive_unsafe']}/{mc['n_trials']}) "
          f"= 혈관침범 {mc['naive_viol']} + 표적미달(>{MISS_TOL*1e3:.0f}mm) {mc['naive_miss']} (중복 포함)")
    print(f"  reliability-gated        : 집행 {mc['executed']}회 중 unsafe {av:.1f}% "
          f"({mc['aware_unsafe']}), 중단 {mc['aborted']}회 → 재프로브로 {mc['recovered']}회 회복")
    n_safe = mc["n_trials"] - mc["naive_unsafe"]

    def gate_row(tag, r):
        ns = r["n_trials"] - r["naive_unsafe"]
        return (f"  {tag:26s} 검출 {100*r['caught']/max(r['naive_unsafe'],1):5.1f}% | "
                f"오경보 {100*r['false_alarm']/max(ns,1):4.1f}% | "
                f"집행분 unsafe {100*r['aware_unsafe']/max(r['executed'],1):4.1f}% "
                f"({r['aware_unsafe']}/{r['executed']})")

    print("  [게이트 절제] 어떤 신호가 실제로 위험을 잡는가")
    print(gate_row("k·σ (조건화)만", mc_sig))
    print(gate_row("+ 다중초기값 일치성", mc))
    print(f"  → 공분산은 '잘못된 basin에 확신에 차서 수렴한' 정합을 못 본다. 일치성 검사가 "
          f"검출을 {100*mc_sig['caught']/max(mc_sig['naive_unsafe'],1):.0f}%→"
          f"{100*mc['caught']/max(mc['naive_unsafe'],1):.0f}%로 올리고 집행분 위험을 절반으로. "
          f"대가는 오경보 {100*mc['false_alarm']/max(n_safe,1):.1f}%")

    # ---- 그림 ----
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    # (1) 수술 장면(로봇 좌표계)
    ax = axes[0, 0]
    body_true = apply_se2(T_true, model)
    ax.plot(*body_true.T, color="0.5", lw=1.2, label="anatomy (true pose)")
    ax.plot(*apply_se2(T_nom, model).T, color="0.8", lw=1.0, ls="--",
            label="anatomy (assumed pose)")
    ax.scatter(*probe.T, s=5, color="tab:blue", alpha=0.6, label="probed points")
    r = out["D registration + CT (calibrated)"]
    ax.plot(*r["path"].T, color="tab:green", lw=1.6, label="planned path (registered)")
    q_end = r["q"][-1]
    arm_pts = kin.fk_points(q_end, L=L_ARM)
    ax.plot(*arm_pts.T, "-o", color="tab:orange", lw=3, ms=5, label="arm at target")
    v_rob = apply_se2(T_true, VESSEL_IMG)
    ax.add_patch(plt.Circle(v_rob, VESSEL_R, color="crimson", alpha=0.35))
    ax.scatter(*apply_se2(T_true, TUMOR_IMG), marker="*", s=140, color="k",
               zorder=5, label="target (true)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title("Scene: arm, probed surface, registered plan", fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(fontsize=7, loc="upper left")

    # (2) 정합 전/후 (영상 좌표계)
    ax = axes[0, 1]
    ax.plot(*model.T, color="0.5", lw=1.2, label="pre-op model (CT)")
    ax.scatter(*apply_se2(coarse_init(probe, model), probe).T, s=5, color="tab:red",
               alpha=0.5, label="probe: coarse init")
    ax.scatter(*apply_se2(T_reg, probe).T, s=5, color="tab:blue", alpha=0.7,
               label="probe: after ICP")
    ax.scatter(*TUMOR_IMG, marker="*", s=140, color="k", zorder=5)
    ax.add_patch(plt.Circle(VESSEL_IMG, VESSEL_R, color="crimson", alpha=0.3))
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"Registration: FRE {fre*1e3:.2f} mm → TRE {tre_reg*1e3:.2f} mm",
                 fontsize=10)
    ax.set_xlabel("x [m]"); ax.legend(fontsize=7, loc="upper left")

    # (3) 오차 예산
    ax = axes[0, 2]
    names = list(out.keys())
    totals = [out[n]["total"] * 1e3 for n in names]
    regs = [out[n]["reg"] * 1e3 for n in names]
    servos = [out[n]["servo"] * 1e3 for n in names]
    xs = np.arange(len(names))
    ax.bar(xs - 0.22, regs, 0.2, label="registration", color="tab:purple")
    ax.bar(xs, servos, 0.2, label="servo (tracking)", color="tab:orange")
    ax.bar(xs + 0.22, totals, 0.2, label="end-to-end", color="tab:green")
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([n[0] for n in names])
    ax.set_ylabel("target error [mm], log scale")
    ax.set_title("Error budget: who owns the millimeters? (A–E)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=7)
    for x, v in zip(xs, totals):
        ax.text(x + 0.22, v, f" {v:.2f}", ha="center", va="bottom", fontsize=7)

    # (4) 팁 추종오차 시간이력 (PD vs 계산토크)
    ax = axes[1, 0]
    rb = out["B registration + PD"]
    rc = out["C registration + CT (3% payload error)"]
    rd = out["D registration + CT (calibrated)"]
    ax.plot(rb["t"], rb["tip_err"] * 1e3, color="tab:red", lw=1.2, label="PD")
    ax.plot(rc["t"], rc["tip_err"] * 1e3, color="tab:orange", lw=1.2,
            label="computed torque (3% payload err)")
    ax.plot(rd["t"], rd["tip_err"] * 1e3, color="tab:green", lw=1.2,
            label="computed torque (calibrated)")
    n_app, n_ins, _ = rd["phases"]
    ax.axvspan(n_app * DT, (n_app + n_ins) * DT, color="0.9", label="insertion")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("t [s]"); ax.set_ylabel("tip path error [mm]")
    ax.set_title("Servo error along the trajectory", fontsize=10); ax.legend(fontsize=8)

    # (5) 정합 불확실도 vs 실제 표적오차 (게이트가 무엇을 잡아내는가)
    ax = axes[1, 1]
    safe_m = ~mc["unsafe"]
    sig_mm = np.clip(mc["sigma"] * 1e3, 1e-3, 1e2)     # 발산 사례는 축을 위해 클리핑
    tre_mm = np.clip(mc["tre"] * 1e3, 1e-3, 1e4)
    ax.scatter(sig_mm[safe_m], tre_mm[safe_m], s=14, color="tab:blue", alpha=0.6, label="safe")
    ax.scatter(sig_mm[~safe_m], tre_mm[~safe_m], s=18, color="crimson", alpha=0.8, label="unsafe")
    ax.axvline(mc["nominal_clear"] * 1e3 / K_SIGMA, color="seagreen", ls="--",
               label=f"abort gate (clearance/{K_SIGMA:.0f})")
    ax.axhline(MISS_TOL * 1e3, color="0.4", ls=":", label="miss tolerance")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predicted σ_target [mm] (clipped at 100)")
    ax.set_ylabel("actual target error [mm]")
    ax.set_title("σ catches ill-conditioning — confident wrong fits slip through\n"
                 "(red dots left of the gate)", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, loc="upper left")

    # (6) 안전 결과
    ax = axes[1, 2]
    vals = [nv, 100 * mc_sig["aware_unsafe"] / max(mc_sig["executed"], 1), av]
    ax.bar(["naive\n(always execute)", f"{K_SIGMA:.0f}σ gate\n(conditioning)",
            "+ multi-start\nconsistency"], vals,
           color=["crimson", "tab:orange", "seagreen"])
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("unsafe executions [%]  (violation or miss)")
    ax.set_title(f"Safety: {mc['aborted']}/{mc['n_trials']} aborted "
                 f"({mc['recovered']} recovered by re-probing)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("42. Image-guided tool targeting capstone — registration → planning → IK → "
                 "computed-torque control, and the error budget", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "42_image_guided_targeting.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/42_image_guided_targeting.png, assets/42_image_guided_targeting.png")

    return out, mc, dict(fre=fre, tre=tre_reg, sigma=sigma_t, place_err=place_err,
                         disagree=disagree, mc_sigma_only=mc_sig)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 평면 슬라이스 가정: 실제는 6-DOF 정합 + 다관절 팔. 여기서는 exp 39/40의 평면 팔과
#     결합하려고 단면(SE(2))으로 축소했다. 사슬의 구조와 오차 예산 논리는 동일하다.
#   - 강체 정합: 연조직 변형·호흡 운동은 다루지 않는다(exp 41의 한계와 동일).
#   - 조직 상호작용 없음: 삽입 시 바늘 휨·조직 반력은 모델에 없다. 실제로는 이 항이
#     서보 몫을 키워 예산의 균형이 달라질 수 있다.
#   - σ_target 은 ICP 선형화 기반 근사(등방 잡음·정확한 대응 가정). 절대값보다
#     '나쁜 조건화를 조기에 드러내는 지표'로서의 쓰임이 핵심이다.
#   - **공분산이 못 보는 실패**: 잔차가 작은데 basin을 잘못 잡은 정합(σ 0.1~0.4 mm인데
#     TRE 16~39 mm)은 k·σ 게이트를 그대로 통과한다. 다중초기값 불일치 검사를 더한 이유이며,
#     그래도 남는 통과분이 있다(집행분 중 unsafe 잔량). 확신에 찬 오정합은 단일 지표로
#     못 막는다 — 임상에서 독립 랜드마크 검증을 별도로 두는 이유와 같다.
#   - 게이트의 대가는 중단률·오경보율이다. 임상적으로 중단 = 표면을 더 찍고 다시 정합
#     (회복률로 측정). RIVAL_RATIO(경쟁해로 인정할 잔차 비율)가 이 트레이드오프를 정한다:
#     1.05→오경보 1%, 1.15→4%, 1.3→21%, 1.5→41% (검출력은 큰 변화 없음) → 1.15 선택.
#   - 다중초기값을 '해 선택'에 쓰면 오히려 나빠진다(부분 커버리지에서는 미끄러진 오정합이
#     더 낮은 잔차를 냄). 검증 신호로만 쓴다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
