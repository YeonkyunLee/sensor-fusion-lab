"""6-DOF 영상유도 도구 유도 — 공간 UR5 팔 + 실 스캔 정합으로 사슬을 완성한다.

exp 42 는 같은 사슬(정합→계획→IK→제어)을 **평면(SE(2)) 슬라이스 + 2링크 팔**로 축소해
다뤘다. 축소는 서사를 단순하게 만들지만 두 가지를 숨긴다.

  1) **자세(orientation)** — 바늘·도구는 위치만이 아니라 *축 방향*을 맞춰야 한다.
     6-DOF 팔에서 IK 는 3(위치)+3(자세) 오차를 동시에 줄여야 하고, 특이점도
     평면의 '팔을 편 자세'가 아니라 **손목 정렬(q5→0)** 로 나타난다.
  2) **도구 몸통(shaft)** — 3D 에서 도구는 점이 아니라 선분이다. 팁 궤적만 검사하면
     통과인데 **몸통이 혈관을 관통**하는 계획이 존재한다. 평면 실험은 이 실패를
     구조적으로 볼 수 없다.

이 실험은 그 둘을 정면으로 다룬다. 팔은 exp 43 에서 쓴 UR5 를 **6축 전체**로 올리고
(공개 DH·질량·무게중심, `sensor_fusion.ur5`), 환자 표면은 exp 44 의 **실 레이저 스캔**
(Stanford Bunny)을 그대로 쓴다. 정합기·신뢰도 게이트도 exp 44 의 것을 재사용한다.

    [실 스캔 팬텀] 미지의 SE(3) 로 배치
        ↓ 프로빙 → 3D 점-대-평면 ICP (exp 44)
    [계획] 표적·진입점·삽입축 → 로봇 좌표로. 도구 z축을 삽입축에 정렬
        ↓ 여유자유도(축 둘레 roll)는 조작성 최대화에 쓴다
    [IK]  6-DOF 감쇠최소자승 (위치+자세)
        ↓
    [제어] UR5 6축 동역학(M·C·g)으로 PD vs 계산토크 추종
        ↓
    [평가] 위치·각도 오차 예산 + 손목 특이점 여유 + shaft 안전성

    python scripts/45_image_guided_6dof.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sensor_fusion import ur5  # noqa: E402
from sensor_fusion.se3 import so3_exp, so3_log  # noqa: E402

real = import_module("44_registration_real_scans")   # 실 스캔 로딩·정합 자산

TOOL_LEN = 0.15                 # 바늘 길이 [m] (플랜지 → 팁, 도구 z축)
TOOL = np.eye(4)
TOOL[2, 3] = TOOL_LEN

DT = 0.004
KP, KD = 400.0, 40.0
T_APPROACH, T_INSERT, T_HOLD = 1.2, 1.2, 0.4
STANDOFF = 0.05                 # 진입점 밖 접근 거리 [m]
VESSEL_R = 0.008                # 금지구조물 반경 [m]
MISS_TOL = 3e-3


# --------------------------------------------------------------------------- #
# 계획 기하: 실 스캔 팬텀 안의 표적 · 혈관 · 진입점
# --------------------------------------------------------------------------- #
def frame_from_axis(z_dir, roll=0.0):
    """z축이 z_dir 인 회전행렬. roll 은 그 축 둘레 자유도(도구 회전 = 과제 무관)."""
    z = np.asarray(z_dir, float)
    z /= np.linalg.norm(z)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return R @ so3_exp(np.array([0.0, 0.0, roll]))


def rotation_between(a, b):
    """a 를 b 로 보내는 최소 회전(로드리게스). 팬텀을 팔이 편한 방향으로 놓는 데 쓴다."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-12:
        return np.eye(3) if c > 0 else so3_exp(np.pi * np.array([1.0, 0.0, 0.0]))
    return so3_exp(v / s * np.arctan2(s, c))


def plan_in_phantom(model, seed=0):
    """팬텀(스캔) 좌표계에서 표적·혈관·진입점을 고른다.

    진입점은 표면점 중 '표적까지의 직선이 혈관 옆을 좁게 지나는' 것을 고른다 —
    여유가 크면 안전 논의가 무의미해지므로 임상적으로 빡빡한 통로를 만든다."""
    rng = np.random.default_rng(seed)
    c = model.mean(0)
    target = c + np.array([0.0, -0.01, 0.005])            # 내부 심부 표적
    vessel = target + np.array([0.012, 0.010, 0.004])      # 표적 근처 금지구조물

    best = None
    for idx in rng.choice(len(model), 400, replace=False):
        entry = model[idx]
        d = target - entry
        L = np.linalg.norm(d)
        if L < 0.05:
            continue
        u = d / L
        s = np.clip((vessel - entry) @ u, 0, L)
        clear = float(np.linalg.norm(entry + s * u - vessel) - VESSEL_R)
        if clear < 0.002:                                  # 통과 불가(이미 관통)
            continue
        if best is None or clear < best[2]:                # 가장 빡빡한 통로 선택
            best = (entry, u, clear)
    return target, vessel, best[0], best[1], best[2]


def shaft_clearance(tip, axis, vessel, length=TOOL_LEN):
    """도구 **몸통**(팁에서 뒤로 length 만큼)과 혈관 중심 사이 최소거리 − 반경.

    팁만 검사하면 놓치는 위반을 잡는다 — 3D 에서만 존재하는 실패 모드."""
    s = np.linspace(0.0, length, 60)[:, None]
    pts = tip[None, :] - s * axis[None, :]
    return float(np.min(np.linalg.norm(pts - vessel, axis=1)) - VESSEL_R)


# --------------------------------------------------------------------------- #
# 경로 → 관절궤적 (6-DOF IK, roll 자유도는 조작성 최대화에 사용)
# --------------------------------------------------------------------------- #
def smoothstep(u):
    return 6 * u ** 5 - 15 * u ** 4 + 10 * u ** 3


def pose_path(entry, axis, target):
    """접근 → 진입 → 표적. 자세는 도구 z축을 삽입축에 정렬해 유지."""
    approach = entry - STANDOFF * axis
    n_app, n_ins, n_hold = (int(round(t / DT)) for t in (T_APPROACH, T_INSERT, T_HOLD))
    u1 = smoothstep(np.linspace(0, 1, n_app))[:, None]
    u2 = smoothstep(np.linspace(0, 1, n_ins))[:, None]
    pts = np.vstack([approach + u1 * (entry - approach),
                     entry + u2 * (target - entry),
                     np.repeat(target[None, :], n_hold, axis=0)])
    return pts, (n_app, n_ins, n_hold)


def choose_roll(entry, axis, q_seed, rolls=12):
    """축 둘레 roll 은 과제에 무관한 자유도 — 조작성이 가장 큰 값을 고른다.

    exp 39 의 '영공간 2차목표'가 6-DOF 에서 나타나는 형태다."""
    best = None
    for roll in np.linspace(0, 2 * np.pi, rolls, endpoint=False):
        T = np.eye(4)
        T[:3, :3] = frame_from_axis(axis, roll)
        T[:3, 3] = entry - STANDOFF * axis
        q, _, ep, er = ur5.ik_dls(T, q_seed, tool=TOOL, lam=0.02, max_iters=200)
        if ep > 1e-4 or er > 1e-3:
            continue
        w = ur5.manipulability(q, tool=TOOL)
        if best is None or w > best[0]:
            best = (w, roll, q)
    return best


def path_to_joints(pts, R_des, q0):
    """경로점마다 6-DOF IK(warm start). 반환 (Q, 최악 위치오차, 최악 자세오차, 조작성)."""
    q = np.array(q0, float)
    Q = np.zeros((len(pts), 6))
    worst_p = worst_r = 0.0
    w_traj = np.zeros(len(pts))
    T = np.eye(4)
    T[:3, :3] = R_des
    for k, p in enumerate(pts):
        T[:3, 3] = p
        q, _, ep, er = ur5.ik_dls(T, q, tool=TOOL, lam=5e-3, max_iters=60,
                                  tol=1e-9, step_cap=0.1)
        Q[k] = q
        worst_p, worst_r = max(worst_p, ep), max(worst_r, er)
        w_traj[k] = ur5.manipulability(q, tool=TOOL)
    return Q, worst_p, worst_r, w_traj


# --------------------------------------------------------------------------- #
# 제어: UR5 6축 동역학 추종
# --------------------------------------------------------------------------- #
def track(Q_des, Qd_des, Qdd_des, mode="ct"):
    """RK4 로 실제 팔을 적분.

    mode: 'ct'  계산토크 τ = M(q)(q̈_d + Kp e + Kd ė) + C q̇ + g
          'pdg' PD + 중력보상 (관성·커플링 미보상)
          'pd'  순수 PD — 6축 UR5 는 중력만으로도 붙잡지 못해 **발산**한다
    발산하면 그 시점까지의 궤적을 NaN 으로 채워 반환한다(정직하게 표에 표시).

    **이득의 단위가 다르다**: 계산토크는 오차 동역학이 단위질량이라 Kp/Kd 가 가속도
    단위지만, PD 계열은 토크를 직접 내므로 관절 관성에 비례해 스케일해야 한다. UR5 는
    관절별 유효관성이 2.4 ~ 1e-4 kg·m² 로 4자릿수 차이라, 균일 이득을 쓰면 작은 손목
    관절이 ω=2000 rad/s 로 진동해 수치적으로도 물리적으로도 발산한다. 여기서는 세
    제어기 모두 같은 목표 대역(ω=20 rad/s, ζ=1)을 갖도록 관성으로 스케일한다."""
    m_ref = np.clip(np.diag(ur5.mass_matrix(Q_des[0])), 1e-4, None)
    kp_j, kd_j = KP * m_ref, KD * m_ref
    q, qd = Q_des[0].copy(), Qd_des[0].copy()
    Q = np.full((len(Q_des), 6), np.nan)
    TIP = np.full((len(Q_des), 3), np.nan)
    AXIS = np.full((len(Q_des), 3), np.nan)
    for k in range(len(Q_des)):
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qd))
                and np.max(np.abs(qd)) < 1e3):
            break                                    # 발산 — 여기서 중단
        e, edot = Q_des[k] - q, Qd_des[k] - qd
        if mode == "ct":
            # 계산토크는 오차 동역학을 단위질량으로 만들므로 이득이 가속도 단위다
            tau = (ur5.mass_matrix(q) @ (Qdd_des[k] + KP * e + KD * edot)
                   + ur5.rnea(q, qd, np.zeros(6)))
        elif mode == "pdg":
            tau = kp_j * e + kd_j * edot + ur5.rnea(q, np.zeros(6), np.zeros(6))
        else:
            tau = kp_j * e + kd_j * edot
        T = ur5.fk(q, tool=TOOL)
        Q[k], TIP[k], AXIS[k] = q, T[:3, 3], T[:3, 2]
        if k == len(Q_des) - 1:
            break

        def deriv(s):
            return np.concatenate([s[6:], ur5.forward_dynamics_fast(s[:6], s[6:], tau)])

        s = np.concatenate([q, qd])
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * DT * k1)
        k3 = deriv(s + 0.5 * DT * k2)
        k4 = deriv(s + DT * k3)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q, qd = s[:6], s[6:]
    return Q, TIP, AXIS


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(n_safety=60):
    # ---- 실 스캔 팬텀 준비 (exp 44 자산) ----
    real.fetch()
    scans = real.load_bunny()
    full0, full45 = scans["bun000"], scans["bun045"]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(full0), real.MODEL_SUBSAMPLE, replace=False)
    model = full0[sel]
    normals = real.reg.estimate_normals(model, k=12)
    tree = cKDTree(model)

    print("=== 45. 6-DOF 영상유도 도구 유도 (공간 UR5 + 실 스캔 팬텀) ===")
    print(f"팔: UR5 6축 공개 DH/질량 (리치 {abs(ur5.DH_A).sum()+ur5.DH_D[0]:.3f} m), "
          f"도구 바늘 {TOOL_LEN*1e3:.0f} mm")
    print(f"팬텀: Stanford Bunny 실 스캔 {len(full0)}점 (모델 {len(model)}점)")

    # ---- 계획 (팬텀 좌표계) ----
    target_p, vessel_p, entry_p, axis_p, clear_plan = plan_in_phantom(model)
    print(f"계획: 삽입 길이 {np.linalg.norm(target_p-entry_p)*1e3:.0f} mm, "
          f"혈관(반경 {VESSEL_R*1e3:.0f} mm) 여유 {clear_plan*1e3:.1f} mm")

    # ---- 팬텀을 로봇 작업공간에 배치 (표적이 도달 가능한 자세에 오도록) ----
    # 배치 회전은 임의로 두지 않는다: 삽입축이 기준자세의 도구 축과 정렬되도록 놓아야
    # 팔이 특이점에서 먼 자세로 과제를 수행한다(자세 구속은 조작성을 깎는다 — 임의
    # 배치에서는 w 가 0.006 까지 떨어져 손목 특이점에 붙는다).
    q_ref = np.array([0.6, -1.15, 1.25, -1.05, -1.55, 0.0])
    T_ref = ur5.fk(q_ref, tool=TOOL)
    R_nom = rotation_between(axis_p, T_ref[:3, 2])
    T_nominal = np.eye(4)                                   # 계획이 가정한 배치
    T_nominal[:3, :3] = R_nom
    T_nominal[:3, 3] = T_ref[:3, 3] - R_nom @ target_p

    # 실제 배치는 공칭에서 조금 어긋난다(환자 세팅 오차) — 이게 정합이 풀 문제다
    dR = so3_exp(rng.normal(0, np.deg2rad(5.0), 3))
    T_err = np.eye(4)
    T_err[:3, :3] = dR
    T_err[:3, 3] = rng.normal(0, 0.012, 3)
    T_place = T_err @ T_nominal                             # image→robot (진짜)

    # ---- 정합: 수술실에서는 조대 정렬(공칭 배치)이 주어지고 ICP 가 다듬는다 ----
    probe = real.make_probe(rng, full45, T_place)
    T_reg = real._icp2(probe, model, normals, np.linalg.inv(T_nominal))
    A, fre, inlier = real.information_matrix(T_reg, probe, tree, model, normals)
    sig_t = real.target_sigma(max(fre, real.PROBE_NOISE) ** 2
                              * np.linalg.inv(A + 1e-12 * np.eye(6)), target_p)
    disagree = float("nan")                                 # 조대정렬이 있으면 무의미
    T_map = np.linalg.inv(T_reg)                            # image→robot (추정)
    tre_vec = real.reg.apply_T(T_map, target_p[None, :])[0] \
        - real.reg.apply_T(T_place, target_p[None, :])[0]
    tre = float(np.linalg.norm(tre_vec))
    rot_err = np.rad2deg(np.linalg.norm(so3_log(T_map[:3, :3] @ T_place[:3, :3].T)))
    setup_err = np.linalg.norm(real.reg.apply_T(T_nominal, target_p[None, :])[0]
                               - real.reg.apply_T(T_place, target_p[None, :])[0])
    print(f"세팅 오차(공칭 배치 가정 시): {setup_err*1e3:.1f} mm")
    print(f"정합: FRE {fre*1e3:.3f} mm | 표적 TRE {tre*1e3:.3f} mm | 회전오차 "
          f"{rot_err:.3f}° | σ {sig_t*1e3:.3f} mm | 겹침 {inlier*100:.0f}%")

    # ---- 계획을 로봇 좌표로 (추정 정합 사용) ----
    def to_robot(T, p):
        return real.reg.apply_T(T, np.atleast_2d(p))[0]

    entry_r = to_robot(T_map, entry_p)
    target_r = to_robot(T_map, target_p)
    axis_r = T_map[:3, :3] @ axis_p
    axis_r /= np.linalg.norm(axis_r)
    target_true_r = to_robot(T_place, target_p)
    vessel_true_r = to_robot(T_place, vessel_p)

    # ---- roll 자유도로 조작성 최대화 (6-DOF 여유자유도) ----
    best = choose_roll(entry_r, axis_r, q_ref)
    w_best, roll_best, q_start = best
    R_des = frame_from_axis(axis_r, roll_best)
    print(f"자세 선택: 축 둘레 roll {np.rad2deg(roll_best):.0f}° 에서 조작성 "
          f"w={w_best:.4f} (12개 후보 중 최대)")

    # ---- IK ----
    pts, phases = pose_path(entry_r, axis_r, target_r)
    Q_des, ik_p, ik_r, w_traj = path_to_joints(pts, R_des, q_start)
    Qd_des = np.gradient(Q_des, DT, axis=0)
    Qdd_des = np.gradient(Qd_des, DT, axis=0)
    print(f"IK: 최악 위치잔차 {ik_p*1e6:.2f} µm, 자세잔차 "
          f"{np.rad2deg(ik_r)*3600:.2f} arcsec | 조작성 최소 {w_traj.min():.4f} "
          f"(손목 특이점 w→0 에서 떨어져 있음)")

    # ---- 제어: PD vs 계산토크 ----
    out = {}
    for mode in ("pd", "pdg", "ct"):
        Q, TIP, AXIS = track(Q_des, Qd_des, Qdd_des, mode=mode)
        diverged = not np.all(np.isfinite(TIP))
        tip_err = np.linalg.norm(TIP - pts, axis=1)
        ang_err = np.array([np.rad2deg(np.arccos(np.clip(a @ R_des[:, 2], -1, 1)))
                            for a in AXIS])
        out[mode] = dict(Q=Q, TIP=TIP, AXIS=AXIS, tip_err=tip_err, ang_err=ang_err,
                         diverged=diverged,
                         final_pos=float(np.linalg.norm(TIP[-1] - target_true_r)),
                         servo_pos=float(np.linalg.norm(TIP[-1] - target_r)),
                         final_ang=float(ang_err[-1]))

    print("-" * 78)
    print(f"{'condition':32s} {'위치오차[mm]':>13s} {'정합몫':>8s} {'서보몫':>9s} "
          f"{'축각도[°]':>10s}")
    for mode, tag in (("pd", "PD only"), ("pdg", "PD + 중력보상"),
                      ("ct", "계산토크 (UR5 6축 M·C·g)")):
        r = out[mode]
        if r["diverged"]:
            print(f"{tag:32s} {'발산':>13s} {tre*1e3:8.3f} {'—':>9s} {'—':>10s}")
        else:
            print(f"{tag:32s} {r['final_pos']*1e3:13.3f} {tre*1e3:8.3f} "
                  f"{r['servo_pos']*1e3:9.3f} {r['final_ang']:10.3f}")
    print("  (총 오차는 정합·서보 오차가 벡터로 상쇄될 수 있어 역전될 수 있다 — "
          "제어기 비교는 '서보 몫' 열로 읽어야 한다)")
    print(f"→ 순수 PD 는 중력에 눌려 {out['pd']['servo_pos']*1e3:.1f} mm 처지고 도구 축이 "
          f"{out['pd']['final_ang']:.1f}° 틀어진다 — 6축에서는 자세까지 무너진다는 점이 "
          "평면 실험과 다르다(관절별 관성이 4자릿수 차이라 이득 스케일링도 필수).")
    print(f"→ 중력보상 대비 계산토크가 서보 몫을 "
          f"{out['pdg']['servo_pos']/max(out['ct']['servo_pos'],1e-12):.0f}배 줄이고, "
          f"남은 오차는 정합 지배(정합 {tre*1e3:.3f} mm vs 서보 "
          f"{out['ct']['servo_pos']*1e3:.3f} mm) — exp 42 결론이 공간에서 재현된다")

    # ---- 6-DOF 에서만 보이는 실패: 팁은 통과, shaft 는 관통 ----
    print("-" * 78)
    print("[shaft 안전성] 도구는 점이 아니라 선분이다 — 축 오차를 키우며 두 검사를 비교")
    angles = np.linspace(0.0, 15.0, n_safety)
    tip_cl, shaft_cl = [], []
    rot_axis = np.cross(axis_r, vessel_true_r - target_true_r)
    if np.linalg.norm(rot_axis) < 1e-9:
        rot_axis = np.cross(axis_r, np.array([0.0, 0.0, 1.0]))
    rot_axis /= np.linalg.norm(rot_axis)
    for a_deg in angles:
        # 혈관을 향해 기우는 쪽이 최악 — 부호를 둘 다 보고 나쁜 쪽을 취한다
        worst = min(
            shaft_clearance(target_true_r,
                            (so3_exp(sgn * rot_axis * np.deg2rad(a_deg)) @ axis_r),
                            vessel_true_r)
            for sgn in (+1.0, -1.0))
        tip_cl.append(float(np.linalg.norm(target_true_r - vessel_true_r) - VESSEL_R))
        shaft_cl.append(worst)
    tip_cl, shaft_cl = np.array(tip_cl), np.array(shaft_cl)
    bad = np.where(shaft_cl < 0)[0]
    if len(bad) and bad[0] > 0:                              # 격자 사이를 선형 보간
        i = bad[0]
        f = shaft_cl[i - 1] / (shaft_cl[i - 1] - shaft_cl[i])
        a_crit = float(angles[i - 1] + f * (angles[i] - angles[i - 1]))
    else:
        a_crit = float(angles[0]) if len(bad) else float("inf")
    slope = -float(np.polyfit(angles, shaft_cl, 1)[0])       # mm/° 로 감소하는 기울기
    print(f"  팁 여유는 축 오차와 무관하게 {tip_cl[0]*1e3:.1f} mm — 팁 검사는 각도에 대해 "
          "**아무 정보도 주지 않는다**")
    print(f"  몸통 여유는 0° 에서 {shaft_cl[0]*1e3:.2f} mm, 기울기 {slope*1e3:.2f} mm/° "
          f"→ **{a_crit:.1f}° 에서 관통**(그 순간에도 팁 검사는 통과)")
    print(f"  이번 정합의 회전오차 {rot_err:.3f}° 이므로 실제 마진은 "
          f"{a_crit/max(rot_err,1e-9):.0f}배로 안전하다. 요점은 이 통로가 위험하다는 게 "
          "아니라, **점 도구를 가정한 평면 실험에서는 이 임계 자체가 존재하지 않는다**는 것")

    # ---- 그림 ----
    fig = plt.figure(figsize=(16.5, 9))

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ph = real.reg.apply_T(T_place, model[::12])
    ax.scatter(ph[:, 0], ph[:, 1], ph[:, 2], s=1, color="0.7", label="phantom (real scan)")
    arm = np.vstack([np.zeros(3)] + [T[:3, 3] for T in ur5.fk_all(out["ct"]["Q"][-1])])
    ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], "-o", color="tab:orange", lw=2, ms=3,
            label="UR5 at target")
    tip_f = out["ct"]["TIP"][-1]
    ax.plot(*np.stack([arm[-1], tip_f]).T, color="tab:red", lw=2, label="tool shaft")
    ax.scatter(*target_true_r, marker="*", s=90, color="k", label="target")
    ax.set_title("6-DOF scene: UR5 + real-scan phantom", fontsize=10)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    ax = fig.add_subplot(2, 3, 2)
    aligned = real.reg.apply_T(T_reg, probe)
    ax.scatter(model[:, 0], model[:, 2], s=1, color="0.7", label="pre-op model")
    ax.scatter(aligned[:, 0], aligned[:, 2], s=4, color="tab:blue", alpha=0.6,
               label="probe after ICP")
    ax.scatter(target_p[0], target_p[2], marker="*", s=120, color="k")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"Registration (real scans): TRE {tre*1e3:.2f} mm, {rot_err:.2f}°",
                 fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]"); ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 3)
    labels = ["PD +\ngravity comp.", "computed\ntorque"]
    xs = np.arange(2)
    ax.bar(xs - 0.2, [tre * 1e3] * 2, 0.35, label="registration", color="tab:purple")
    ax.bar(xs + 0.2, [out["pdg"]["servo_pos"] * 1e3, out["ct"]["servo_pos"] * 1e3], 0.35,
           label="servo", color="tab:orange")
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("error share [mm], log")
    ax.set_title("6-DOF error budget", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 4)
    t = np.arange(len(pts)) * DT
    ax.semilogy(t, np.maximum(out["pdg"]["tip_err"], 1e-9) * 1e3, color="crimson",
                label="PD + gravity comp.")
    ax.semilogy(t, np.maximum(out["ct"]["tip_err"], 1e-9) * 1e3, color="tab:green",
                label="computed torque")
    n_app, n_ins, _ = phases
    ax.axvspan(n_app * DT, (n_app + n_ins) * DT, color="0.9", label="insertion")
    ax.set_xlabel("t [s]"); ax.set_ylabel("tip error [mm]")
    ax.set_title("Tracking (position)", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, w_traj, color="tab:blue")
    ax.axhline(0, color="crimson", ls="--", lw=1, label="wrist singularity (w=0)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("manipulability w")
    ax.set_title(f"Manipulability along path (roll chosen: {np.rad2deg(roll_best):.0f}°)",
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(angles, tip_cl * 1e3, color="tab:blue", label="tip-only check")
    ax.plot(angles, shaft_cl * 1e3, color="crimson", label="tool shaft")
    ax.axhline(0, color="0.4", ls="--", lw=1, label="vessel surface")
    if np.isfinite(a_crit):
        ax.axvline(a_crit, color="crimson", ls=":", lw=1)
        ax.text(a_crit, 1, f" cuts at {a_crit:.1f}°", fontsize=8, color="crimson")
    ax.axvline(rot_err, color="seagreen", ls=":", lw=1)
    ax.text(rot_err, tip_cl[0] * 1e3 * 0.6, f" registration {rot_err:.2f}°",
            fontsize=7, color="seagreen")
    ax.set_xlabel("tool-axis error [deg]"); ax.set_ylabel("clearance [mm]")
    ax.set_title("A point tool cannot see this failure", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle("45. 6-DOF image-guided targeting — spatial UR5 (published specs) + "
                 "real-scan phantom", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "45_image_guided_6dof.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/45_image_guided_6dof.png, assets/45_image_guided_6dof.png")

    return dict(tre=tre, rot_err=rot_err, ik_p=ik_p, ik_r=ik_r,
                w_min=float(w_traj.min()), out=out, a_crit=a_crit,
                clear_plan=clear_plan, setup_err=float(setup_err),
                shaft_clear0=float(shaft_cl[0]), tip_clear=float(tip_cl[0]))


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 팬텀은 해부 구조가 아니라 Bunny 실 스캔이다(exp 44 와 같은 한계). 검증하는 것은
#     '수술'이 아니라 실측 기하 위에서의 6-DOF 사슬이다.
#   - 관성 텐서는 균일 원기둥 근사(공개표의 값이 출처마다 달라 명시 채택). 결론(모델기반
#     제어의 이득, 정합 지배)은 이 근사에 민감하지 않다.
#   - 조직 반력·바늘 휨 없음. 삽입 중 shaft 가 휘면 여유가 더 줄어든다.
#   - roll 최적화는 이산 12개 후보의 완전탐색이다. 실제로는 관절한계·충돌까지 넣은
#     연속 최적화가 필요하다.
#   - shaft 안전성 실험은 자세 오차를 직접 주입해 만든 것으로, 정합오차→자세오차의
#     전파를 완전히 모델링하지는 않았다(방향성은 같다).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
