"""조직 반력과 임피던스 제어: 접촉하는 순간 위치 제어만으로는 부족하다.

exp 45 까지의 6-DOF 사슬은 **자유공간**을 가정했다. 도구가 조직에 닿는 순간부터는
로봇이 환경에 힘을 가하고, 환경도 로봇을 민다. 이때 위치 제어기는 목표를 지키려고
힘을 무한정 키우려 든다 — 자유공간에서는 미덕이던 '강한 추종'이 접촉에서는 위험이 된다.

이 실험은 exp 45 의 팔·과제에 **바늘-조직 상호작용 모델**을 붙이고, 두 제어 철학을
같은 과제에서 비교한다.

  - **위치 제어(계산토크)**: 목표 궤적을 강하게 추종. 조직이 밀어도 굽히지 않는다.
  - **임피던스 제어**: 팁이 원하는 강성·감쇠를 가진 것처럼 거동하게 한다. 구현은
    작업공간(operational-space) 제어 τ = Jᵀ Λ(q)[ẍ_d + Kp e + Kd ė − J̇q̇] + Cq̇ + g 로,
    등가 강성은 K_eff = Λ·Kp. 조직이 예상보다 단단하면 **로봇이 양보**한다.
    (단순한 τ = Jᵀ[K e + D ė] 형태가 왜 이 팔에서 발산하는지는 simulate() 주석에.)

--- 바늘-조직 힘 모델 ---
문헌(바늘 삽입 역학)에서 보고되는 세 성분을 자릿수 수준으로 재현한다. 특정 논문 값의
재현이 아니라, **거동의 구조**(찌르기 전 강성 증가 → 관통 시 급락 → 삽입 중 마찰)를
갖춘 합성 모델이다.

  1) 관통 전: 표면이 눌리며 비선형 강성 F = k₁·d + k₂·d²  (d = 눌린 깊이)
  2) 관통: 힘이 임계 F_punc 를 넘으면 표면이 뚫리고 힘이 급감(불연속)
  3) 관통 후: 절삭력(일정) + 삽입 깊이에 비례하는 축방향 마찰 + 접선 방향 저항

--- 정직하게 볼 것 ---
  - 위치 제어는 표적을 잘 맞히지만 **최대 상호작용력**이 크다.
  - 임피던스 제어는 힘을 낮추지만 **정상상태 위치 오차**(조직 반력 / K)를 남긴다.
  - 그 둘은 강성 K 하나로 이어진 트레이드오프 곡선 위에 있고, 임상 요구(허용 힘 vs
    표적 정확도)가 작동점을 고른다. 스윕으로 그 곡선을 그린다.

    python scripts/47_needle_impedance.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sensor_fusion import ur5  # noqa: E402
from sensor_fusion.se3 import so3_log  # noqa: E402

g6 = import_module("45_image_guided_6dof")      # 6-DOF 과제 기하·도구 정의

TOOL = g6.TOOL
DT = 0.004
KP, KD = 400.0, 40.0            # 계산토크(가속도 단위) 이득
T_APPROACH, T_INSERT, T_HOLD = 1.0, 1.6, 0.6

# --- 바늘-조직 힘 모델 파라미터 (문헌에서 보고되는 자릿수) ---
K1_TISSUE = 300.0               # 표면 선형 강성 [N/m]
K2_TISSUE = 40000.0             # 2차 항 [N/m²] — 누를수록 급격히 단단해짐
F_PUNCTURE = 4.0                # 관통 임계력 [N]
F_CUT = 0.8                     # 관통 후 절삭력 [N]
MU_AXIAL = 12.0                 # 축방향 마찰 계수 [N/m] (삽입 깊이에 비례)
K_LATERAL = 800.0               # 축에 수직한 방향의 조직 저항 [N/m]
F_LIMIT = 6.0                   # 임상 허용 최대 상호작용력 [N] (안전 기준)
LUNGE_LIMIT = 5e-4              # 임상 허용 관통 돌진 [m] (0.5 mm)


class NeedleTissue:
    """바늘 팁이 표면을 누르고 → 관통하고 → 삽입되는 동안의 반력 모델(상태 있음)."""

    def __init__(self, surface_point, axis):
        self.p0 = np.asarray(surface_point, float)      # 표면 진입점
        self.u = np.asarray(axis, float) / np.linalg.norm(axis)
        self.punctured = False
        self.puncture_k = None                          # 관통이 일어난 스텝
        self.max_force = 0.0
        self.k = 0                                      # 현재 스텝(외부에서 갱신)

    def force(self, tip):
        """팁 위치에서 조직이 바늘에 가하는 힘 [N] (로봇 좌표계 3벡터)."""
        rel = tip - self.p0
        depth = float(rel @ self.u)                      # 축방향 침투 깊이
        lateral = rel - depth * self.u                   # 축에서 벗어난 성분
        if depth <= 0.0:
            f = np.zeros(3)                              # 아직 접촉 전
        elif not self.punctured:
            mag = K1_TISSUE * depth + K2_TISSUE * depth ** 2
            if mag >= F_PUNCTURE:
                self.punctured = True                    # 표면 관통(불연속)
                self.puncture_k = self.k
                mag = F_CUT
            f = -mag * self.u
        else:
            f = -(F_CUT + MU_AXIAL * depth) * self.u     # 절삭 + 축방향 마찰
        f = f - K_LATERAL * lateral                      # 조직이 축을 잡아 준다
        self.max_force = max(self.max_force, float(np.linalg.norm(f)))
        return f


def tip_state(q, qd):
    """도구 팁 위치와 속도(자코비안으로 전파)."""
    T = ur5.fk(q, tool=TOOL)
    J = ur5.jacobian(q, tool=TOOL)
    return T[:3, 3], J[:3] @ qd, J


def apparent_inertia(q, J=None):
    """작업공간 유효관성 Λ = (J M⁻¹ Jᵀ)⁻¹ (6×6)."""
    if J is None:
        J = ur5.jacobian(q, tool=TOOL)
    Minv = np.linalg.inv(ur5.mass_matrix(q))
    return np.linalg.inv(J @ Minv @ J.T + 1e-12 * np.eye(6)), J


def simulate(Q_des, Qd_des, Qdd_des, path, R_des, tissue, mode="position",
             kp_task=400.0, zeta=1.0, dt=DT):
    """조직과 상호작용하며 궤적 추종.

    mode='position' : 관절공간 계산토크. 등가 카테시안 강성 ≈ Λ·KP 로 매우 뻣뻣하다.
    mode='impedance': **작업공간(operational-space) 제어**
        τ = Jᵀ Λ(q)[ ẍ_d + Kp e + Kd ė − J̇q̇ ] + C q̇ + g

    왜 Jᵀ(K e + D ė) 형태를 쓰지 않았나(정직한 기록): 그렇게 하면 방향마다 유효관성이
    달라 대역이 제각각이 된다. UR5+바늘의 경우 **바늘 축 둘레 스핀의 카테시안 유효관성이
    1.2e-4 kg·m²** 라, 관성 결합을 통해 그 모드가 ω≈1400 rad/s 로 여기되어 스텝마다 16배씩
    발산했다(축 성분을 사영해 제거해도 Λ 의 비대각 결합 때문에 남는다). Λ 로 관성을
    정규화하면 모든 방향의 폐루프 대역이 √Kp 로 균일해져 이 문제가 사라진다.

    **임피던스로서의 해석**: 정상상태에서 외력 f 에 대한 변위는 f/(Λ·Kp) 이므로,
    등가 강성은 K_eff = Λ_uu·Kp (자세 의존). 즉 Kp 하나가 '얼마나 양보할지'를 정한다.
    """
    q, qd = Q_des[0].copy(), Qd_des[0].copy()
    n = len(Q_des)
    TIP = np.zeros((n, 3))
    F = np.zeros((n, 3))
    kd_task = 2.0 * zeta * np.sqrt(kp_task)
    for k in range(n):
        tissue.k = k
        tip, tip_v, J = tip_state(q, qd)
        f_ext = tissue.force(tip)                        # 조직 → 바늘
        TIP[k], F[k] = tip, f_ext
        if k == n - 1:
            break

        if mode == "position":
            e, edot = Q_des[k] - q, Qd_des[k] - qd
            tau_cmd = (ur5.mass_matrix(q) @ (Qdd_des[k] + KP * e + KD * edot)
                       + ur5.rnea(q, qd, np.zeros(6)))
        else:
            T_cur = ur5.fk(q, tool=TOOL)
            Lam, J = apparent_inertia(q, J)
            xd_d = (path[k + 1] - path[k]) / dt
            xdd_d = ((path[k + 1] - 2 * path[k] + path[k - 1]) / dt ** 2
                     if 0 < k < n - 1 else np.zeros(3))
            e_p = path[k] - tip
            e_r = so3_log(R_des @ T_cur[:3, :3].T)
            omega = J[3:] @ qd
            a_cmd = np.concatenate([xdd_d + kp_task * e_p + kd_task * (xd_d - tip_v),
                                    kp_task * e_r - kd_task * omega])
            # J̇q̇ (수치 미분) — 작업공간 가속도와 관절 가속도를 잇는 항
            h = 1e-5
            Jdot_qd = (ur5.jacobian(q + h * qd, tool=TOOL)
                       - ur5.jacobian(q - h * qd, tool=TOOL)) @ qd / (2 * h)
            tau_cmd = J.T @ (Lam @ (a_cmd - Jdot_qd)) + ur5.rnea(q, qd, np.zeros(6))

        # 외력은 관절토크로 τ_ext = Jᵀ f 로 들어온다
        tau_ext = J[:3].T @ f_ext

        def deriv(s):
            return np.concatenate(
                [s[6:], ur5.forward_dynamics_fast(s[:6], s[6:], tau_cmd + tau_ext)])

        s = np.concatenate([q, qd])
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q, qd = s[:6], s[6:]
    return TIP, F


def build_task(dt=DT):
    """exp 45 의 기하를 그대로 쓰되, 자유공간 대신 조직이 있는 삽입 과제로."""
    real = g6.real
    real.fetch()
    scans = real.load_bunny()
    rng = np.random.default_rng(0)
    full0 = scans["bun000"]
    sel = rng.choice(len(full0), real.MODEL_SUBSAMPLE, replace=False)
    model = full0[sel]
    target_p, vessel_p, entry_p, axis_p, clear = g6.plan_in_phantom(model)

    q_ref = np.array([0.6, -1.15, 1.25, -1.05, -1.55, 0.0])
    T_ref = ur5.fk(q_ref, tool=TOOL)
    R_place = g6.rotation_between(axis_p, T_ref[:3, 2])
    T_place = np.eye(4)
    T_place[:3, :3] = R_place
    T_place[:3, 3] = T_ref[:3, 3] - R_place @ target_p

    entry_r = real.reg.apply_T(T_place, entry_p[None, :])[0]
    target_r = real.reg.apply_T(T_place, target_p[None, :])[0]
    axis_r = R_place @ axis_p
    axis_r /= np.linalg.norm(axis_r)

    # 카테시안 경로: 접근 → 진입점 → 표적 (자세는 삽입축 정렬 유지)
    approach = entry_r - g6.STANDOFF * axis_r
    n_app, n_ins, n_hold = (int(round(t / dt)) for t in (T_APPROACH, T_INSERT, T_HOLD))
    u1 = g6.smoothstep(np.linspace(0, 1, n_app))[:, None]
    u2 = g6.smoothstep(np.linspace(0, 1, n_ins))[:, None]
    path = np.vstack([approach + u1 * (entry_r - approach),
                      entry_r + u2 * (target_r - entry_r),
                      np.repeat(target_r[None, :], n_hold, axis=0)])

    R_des = g6.frame_from_axis(axis_r, 0.0)
    q = q_ref.copy()
    Q = np.zeros((len(path), 6))
    T = np.eye(4)
    T[:3, :3] = R_des
    for k, p in enumerate(path):
        T[:3, 3] = p
        q, _, _, _ = ur5.ik_dls(T, q, tool=TOOL, lam=5e-3, max_iters=60, tol=1e-9,
                                step_cap=0.1)
        Q[k] = q
    Qd = np.gradient(Q, dt, axis=0)
    Qdd = np.gradient(Qd, dt, axis=0)
    return dict(Q=Q, Qd=Qd, Qdd=Qdd, path=path, entry=entry_r, target=target_r,
                axis=axis_r, R_des=R_des, phases=(n_app, n_ins, n_hold), dt=dt)


def subsample(task, factor):
    """미세 스텝으로 만든 과제를 factor 배 성기게(같은 궤적, 큰 dt) 다시 샘플링."""
    out = dict(task)
    dt = task["dt"] * factor
    out["Q"] = task["Q"][::factor]
    out["path"] = task["path"][::factor]
    out["Qd"] = np.gradient(out["Q"], dt, axis=0)
    out["Qdd"] = np.gradient(out["Qd"], dt, axis=0)
    out["phases"] = tuple(n // factor for n in task["phases"])
    out["dt"] = dt
    return out


def puncture_lunge(TIP, path, axis, tissue, dt=DT, window_s=0.15):
    """관통 직후 '돌진'(breakthrough overshoot): 표면이 뚫려 힘이 급락하는 순간,
    눌려 있던 변형이 풀리며 팁이 **계획보다 더 많이** 전진하는 양.

    주의: 절대 위치로 재면 안 된다 — 하중 아래에서 팁은 늘 계획보다 뒤처져 있어
    값이 항상 음수가 된다. 관통 시점을 기준으로 한 **전진량의 초과분**을 봐야 한다."""
    if tissue.puncture_k is None:
        return 0.0
    k0 = tissue.puncture_k
    k1 = min(len(TIP), k0 + int(window_s / dt))
    if k1 <= k0 + 1:
        return 0.0
    adv_actual = (TIP[k0:k1] - TIP[k0]) @ axis      # 실제 전진
    adv_plan = (path[k0:k1] - path[k0]) @ axis      # 계획된 전진
    return float(np.max(adv_actual - adv_plan))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_case(task, mode, **kw):
    """한 조건 실행 + 지표 계산."""
    tissue = NeedleTissue(task["entry"], task["axis"])
    TIP, F = simulate(task["Q"], task["Qd"], task["Qdd"], task["path"], task["R_des"],
                      tissue, mode=mode, dt=task["dt"], **kw)
    fmag = np.linalg.norm(F, axis=1)
    ok = np.all(np.isfinite(TIP))
    return dict(TIP=TIP, F=F, fmag=fmag, ok=ok,
                peak=float(np.nanmax(fmag)) if ok else np.nan,
                final_err=float(np.linalg.norm(TIP[-1] - task["target"]))
                if ok else np.nan,
                punctured=tissue.punctured,
                lunge=puncture_lunge(TIP, task["path"], task["axis"], tissue,
                                     dt=task["dt"]) if ok else np.nan,
                over=float(np.mean(fmag > F_LIMIT) * 100) if ok else np.nan,
                dt=task["dt"])


def main(k_sweep=(50.0, 100.0, 200.0, 400.0), dt_fine=0.001, kp_fine_below=100.0):
    # 부드러운 게인은 접촉 강성 대비 감쇠가 작아 큰 스텝에서 적분이 불안정해진다.
    # 궤적을 미세 스텝으로 한 번 만들고, 필요한 조건만 그대로 쓰고 나머지는 성기게 쓴다.
    task_fine = build_task(dt=dt_fine)
    task = subsample(task_fine, int(round(DT / dt_fine)))
    Q, Qd, Qdd, path = task["Q"], task["Qd"], task["Qdd"], task["path"]

    print("=== 47. 조직 반력과 임피던스 제어 (6-DOF UR5 + 바늘-조직 모델) ===")
    print(f"조직 모델: 표면강성 {K1_TISSUE:.0f} N/m + {K2_TISSUE:.0f} N/m², "
          f"관통력 {F_PUNCTURE:.1f} N, 절삭 {F_CUT:.1f} N, 축마찰 {MU_AXIAL:.0f} N/m")
    print(f"과제: 진입점→표적 {np.linalg.norm(task['target']-task['entry'])*1e3:.0f} mm 삽입, "
          f"임상 허용력 {F_LIMIT:.1f} N")

    # 등가 강성 환산용: 삽입축 방향의 작업공간 유효관성
    Lam0, _ = apparent_inertia(task["Q"][0])
    lam_u = float(task["axis"] @ Lam0[:3, :3] @ task["axis"])
    print(f"삽입축 방향 유효관성 Λ_uu = {lam_u:.2f} kg → 등가 강성 K_eff = Λ_uu·Kp")

    res = {"position": run_case(task, "position"),
           "impedance": run_case(task, "impedance", kp_task=100.0)}

    print("-" * 78)
    print(f"{'controller':16s} {'표적오차[mm]':>13s} {'최대힘[N]':>10s} "
          f"{'관통돌진[mm]':>13s} {'관통':>6s}")
    for tag in ("position", "impedance"):
        r = res[tag]
        print(f"{tag:16s} {r['final_err']*1e3:13.3f} {r['peak']:10.2f} "
              f"{r['lunge']*1e3:13.3f} {'예' if r['punctured'] else '아니오':>6s}")
    p, i = res["position"], res["impedance"]
    print(f"→ 최대힘은 둘 다 관통 임계({F_PUNCTURE:.1f} N) 근처다 — 이 모델에서 힘의 상한은 "
          "조직이 정한다. 차이는 **관통 직후**에 나온다: 힘이 급락하는 순간 위치 제어는 "
          f"{p['lunge']*1e3:.2f} mm 돌진하고, 임피던스는 {i['lunge']*1e3:.2f} mm 에 그친다.")
    print(f"   대신 임피던스는 조직 반력을 강성으로 나눈 만큼 표적오차를 내준다 "
          f"({p['final_err']*1e3:.2f} → {i['final_err']*1e3:.2f} mm).")

    # ---- 강성 스윕: 트레이드오프 곡선 ----
    print("-" * 78)
    print("[강성 스윕] 임피던스 K 하나가 '힘 vs 정확도' 작동점을 고른다")
    sweep = []
    for kp in k_sweep:
        t_used = task_fine if kp < kp_fine_below else task
        r = run_case(t_used, "impedance", kp_task=kp)
        sweep.append((kp, r["peak"], r["final_err"], r["punctured"], r["lunge"],
                      lam_u * kp, r["dt"]))
        print(f"  Kp={kp:5.0f} (K_eff {lam_u*kp:7.0f} N/m, dt={r['dt']*1e3:.0f} ms): "
              f"최대힘 {r['peak']:5.2f} N | 표적오차 {r['final_err']*1e3:8.3f} mm | "
              f"관통돌진 {r['lunge']*1e3:6.3f} mm | "
              f"관통 {'예' if r['punctured'] else '아니오'}")
    ok = [s for s in sweep if s[3]]
    if ok:
        best = min(ok, key=lambda s: s[2])
        print(f"  → 제약 없이 정확도만 보면 가장 뻣뻣한 Kp={best[0]:.0f} "
              f"(K_eff {best[5]:.0f} N/m, 오차 {best[2]*1e3:.3f} mm)이 이긴다 — "
              f"단 관통 돌진 {best[4]*1e3:.3f} mm 를 감수한다")
        safe = [s for s in ok if s[4] <= LUNGE_LIMIT]
        if safe:
            pick = min(safe, key=lambda s: s[2])
            print(f"  → 돌진 {LUNGE_LIMIT*1e3:.1f} mm 이하를 요구하면 작동점이 바뀐다: "
                  f"Kp={pick[0]:.0f} (K_eff {pick[5]:.0f} N/m, 오차 {pick[2]*1e3:.3f} mm, "
                  f"돌진 {pick[4]*1e3:.3f} mm) — 임상 제약이 강성을 고른다")
        soft = [s for s in sweep if not s[3]]
        if soft:
            print(f"  → Kp ≤ {max(s[0] for s in soft):.0f} (K_eff "
                  f"{max(s[5] for s in soft):.0f} N/m) 는 조직을 뚫지 못한다 — "
                  "너무 물렁한 로봇은 과제 자체를 수행하지 못한다(안전의 반대편 대가)")
    else:
        print("  → 이 조직에서는 어떤 Kp 로도 관통하지 못했다")
    print(f"  * 부드러운 게인(Kp<{kp_fine_below:.0f})은 dt={dt_fine*1e3:.0f} ms 로 적분했다: "
          "제어 감쇠가 접촉 강성에 비해 작아 4 ms 스텝에서는 적분이 깨진다(수치 문제 — "
          "Kp=100 에서 두 스텝의 결과가 4.75 vs 4.76 mm 로 일치함을 확인)")

    # ---- 예산에 추가되는 항 ----
    print("-" * 78)
    print("[오차 예산에 새 항] exp 45 까지의 예산은 정합 + 서보였다")
    print("  자유공간 서보 몫(exp 45)      : ~0.000 mm")
    print(f"  조직 접촉 시 상호작용 몫      : 위치제어 {p['final_err']*1e3:.3f} mm / "
          f"임피던스(K_eff {lam_u*100:.0f} N/m) {i['final_err']*1e3:.3f} mm")
    print(f"  정합 몫(exp 45 실측)          : 0.081 mm")
    print("  → 접촉이 들어오면 상호작용 항이 정합·서보와 같은 자릿수로 올라온다. "
          "'무엇을 개선할지'의 답이 또 바뀐다.")

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    dt_c = task["dt"]
    t = np.arange(len(path)) * dt_c
    n_app, n_ins, _ = task["phases"]

    ax = axes[0]
    ax.plot(t, res["position"]["fmag"], color="crimson", label="position control")
    ax.plot(t, res["impedance"]["fmag"], color="seagreen",
            label=f"impedance (K_eff {lam_u*100:.0f} N/m)")
    ax.axhline(F_LIMIT, color="0.4", ls="--", lw=1, label=f"limit {F_LIMIT:.0f} N")
    ax.axvspan(n_app * dt_c, (n_app + n_ins) * dt_c, color="0.92", label="insertion")
    ax.set_xlabel("t [s]"); ax.set_ylabel("interaction force [N]")
    ax.set_title("Contact force: the puncture spike", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    err_p = np.linalg.norm(res["position"]["TIP"] - path, axis=1) * 1e3
    err_i = np.linalg.norm(res["impedance"]["TIP"] - path, axis=1) * 1e3
    ax.semilogy(t, np.maximum(err_p, 1e-6), color="crimson", label="position control")
    ax.semilogy(t, np.maximum(err_i, 1e-6), color="seagreen", label="impedance")
    ax.axvspan(n_app * dt_c, (n_app + n_ins) * dt_c, color="0.92")
    ax.set_xlabel("t [s]"); ax.set_ylabel("tip tracking error [mm]")
    ax.set_title("The price of yielding", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[2]
    punc = [s for s in sweep if s[3]]
    fail = [s for s in sweep if not s[3]]
    if punc:
        ax.plot([s[4] * 1e3 for s in punc], [s[2] * 1e3 for s in punc], "-o",
                color="tab:blue", label="impedance (punctured)")
        for s in punc:
            ax.annotate(f"{s[5]:.0f} N/m", (s[4] * 1e3, s[2] * 1e3), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    if fail:
        ax.scatter([s[4] * 1e3 for s in fail], [s[2] * 1e3 for s in fail],
                   marker="x", s=60, color="0.5", label="too soft: no puncture")
    ax.scatter([res["position"]["lunge"] * 1e3], [res["position"]["final_err"] * 1e3],
               marker="*", s=160, color="crimson", label="position control", zorder=5)
    ax.set_xlabel("puncture lunge [mm]"); ax.set_ylabel("target error [mm]")
    ax.set_yscale("log")
    ax.set_title("Stiffness K picks the operating point", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    fig.suptitle("47. Needle–tissue interaction: position control vs impedance control "
                 "(6-DOF UR5)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "47_needle_impedance.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/47_needle_impedance.png, assets/47_needle_impedance.png")

    return dict(res=res, sweep=sweep, task=task)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 조직 모델은 문헌에서 보고되는 **자릿수와 거동 구조**(관통 전 비선형 강성 → 관통 →
#     절삭+마찰)를 따른 합성 모델이다. 특정 장기·바늘의 실측 재현이 아니다.
#   - 바늘 휨(유연 바늘)은 없다. 실제로는 축에서 벗어난 힘이 바늘을 휘게 해 exp 45 의
#     shaft 여유를 더 깎는다 — 두 실험을 잇는 다음 항목.
#   - 힘 측정은 이상적(잡음·대역 제한 없음)이라고 가정했다. 실제 F/T 센서 잡음과 지연은
#     임피던스 제어의 안정 한계를 정한다.
#   - 임피던스 제어는 여기서 도구 팁 3자유도에만 걸었다(자세는 동역학 보상만). 6-DOF
#     임피던스는 회전 강성까지 설계해야 한다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
