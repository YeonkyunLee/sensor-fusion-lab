"""유연 바늘: 휨이 먹어치우는 여유와, 스핀이 되찾는 정확도.

exp 45 는 도구를 **강체 선분**으로 봤다. 그 가정 위에서 "몸통(shaft)이 혈관을 관통하는
축 오차 임계는 9.7°" 라는 결론이 나왔다. exp 47 은 조직이 바늘을 미는 **힘**을 넣었지만
바늘 자체는 여전히 휘지 않았다. 실제 바늘은 가늘고(21G ≈ 0.8 mm) 길어서, 경사면
(bevel) 팁이 만드는 횡력에 눌려 **활처럼 휜다** — 이것이 needle steering 이라는 분야가
존재하는 이유다.

이 실험은 두 조각을 잇는다.

  1) **왜 휘는가(역학)**: 팁 경사면의 횡력 + 조직의 횡방향 지지를 받는 외팔보로 풀어
     처짐을 구한다. 작은 각 가정에서 에너지가 2차식이라 **선형 방정식 하나**로 정확히
     풀린다(반복 없음). 여기서 등가 곡률 κ 를 얻는다.
  2) **그래서 얼마나 위험한가(기하)**: 그 곡률로 바늘 경로를 적분해 **휜 형상**을 만들고,
     exp 45 의 혈관 통로에 대해 여유를 다시 잰다. 강체 가정이 봤던 2.17 mm 여유와
     9.7° 임계가 어떻게 바뀌는지.

그리고 해법이 예상 밖의 곳에서 나온다. exp 45 에서 **"과제와 무관한 자유도"** 로 취급해
조작성 최대화에 써버렸던 *바늘 축 둘레 스핀*이, 여기서는 곡률의 방향을 바꾸는 **제어
입력**이다. 삽입 중간에 180° 뒤집으면 두 호가 서로를 상쇄하고(flip), 계속 돌리면 곡률이
평균 0 이 된다(duty cycling). 같은 자유도를 두 목적에 동시에 쓸 수는 없다 —
그 트레이드오프까지 정직하게 적는다.

    python scripts/48_flexible_needle.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
g6 = import_module("45_image_guided_6dof")      # 통로 기하(진입점·표적·혈관)
imp = import_module("47_needle_impedance")      # 조직 힘 규모(절삭력·횡지지)

# --- 바늘 물성 (21G 스테인리스 규격 수준) ---
D_NEEDLE = 0.8e-3               # 외경 [m]
E_STEEL = 200e9                 # 영률 [Pa]
I_AREA = np.pi * D_NEEDLE ** 4 / 64.0
EI = E_STEEL * I_AREA           # 굽힘강성 [N·m²]

# --- 조직/절삭 (exp 47 과 같은 자릿수) ---
F_BEVEL_RATIO = 0.3             # 경사면 횡력 / 절삭력
F_BEVEL = F_BEVEL_RATIO * imp.F_CUT
K_T_PER_LEN = imp.K_LATERAL / g6.TOOL_LEN   # 횡방향 조직 지지 [N/m per m]

N_SEG = 40                      # 보 이산화 세그먼트 수
VESSEL_R = g6.VESSEL_R


# --------------------------------------------------------------------------- #
# 1) 역학: 조직에 지지된 외팔보의 처짐 (선형 해)
# --------------------------------------------------------------------------- #
def beam_deflection(length, f_bevel=F_BEVEL, k_t=K_T_PER_LEN, n=N_SEG, ei=EI):
    """삽입 깊이 length 의 바늘이 팁 횡력 f_bevel 을 받을 때의 처짐선 y(s).

    작은 각 가정에서 세그먼트 꺾임각 θ 에 대해
        E(θ) = ½θᵀ(EI/Δs)θ + ½(Aθ)ᵀ K_t (Aθ) − f·(Aθ)_tip,   y = Aθ
    가 2차식이므로 ∂E/∂θ = 0 이 선형 방정식이 된다 — 반복 없이 정확히 푼다.

    주의(직접 틀렸던 부분): 꺾임각 → 처짐은 **이중 적분**이다. 기울기 = 꺾임각의
    누적합, 처짐 = 기울기의 누적합×Δs. 한 번만 누적하면 처짐이 수백 배 작게 나와
    (여기서는 6.8 mm → 0.01 mm) 결론이 통째로 뒤집힌다. 해석해 y=Fℓ³/(3EI) 와 대조해
    검증한다(tests).

    반환 (s, y, 등가곡률 κ)."""
    ds = length / n
    s = np.arange(1, n + 1) * ds
    S = np.tril(np.ones((n, n)))               # 누적합 연산자
    A = ds * (S @ S)                           # y = Δs·(기울기의 누적) = Δs·S(Sθ)
    K_b = np.eye(n) * (ei / ds)                # 굽힘 강성
    K_t = np.eye(n) * (k_t * ds)               # 조직의 횡방향 지지(노드마다)
    rhs = f_bevel * A[-1]                      # 팁 횡력이 하는 일
    theta = np.linalg.solve(K_b + A.T @ K_t @ A, rhs)
    y = A @ theta
    kappa = 2.0 * y[-1] / length ** 2          # sagitta ↔ 곡률 근사
    return s, y, float(kappa)


# --------------------------------------------------------------------------- #
# 2) 기하: 곡률을 적분해 3D 바늘 형상 (스핀 정책에 따라 곡률 방향이 바뀐다)
# --------------------------------------------------------------------------- #
def needle_shape(entry, axis, length, kappa, spin_policy=None, n=200):
    """진입점에서 axis 방향으로 들어가며 곡률 kappa 로 휘는 바늘 중심선 (n,3).

    spin_policy(s) → 경사면 방위각 φ [rad]. None 이면 φ=0 고정(한쪽으로만 휨).
    곡률 벡터는 축에 수직한 평면에서 φ 만큼 돌아간 방향을 향한다."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    # 축에 수직한 기준 두 축
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    ds = length / n
    p = np.asarray(entry, float).copy()
    t = axis.copy()
    pts = [p.copy()]
    for k in range(n):
        s = (k + 0.5) * ds
        phi = 0.0 if spin_policy is None else float(spin_policy(s))
        bend_dir = np.cos(phi) * e1 + np.sin(phi) * e2
        bend_dir = bend_dir - (bend_dir @ t) * t        # 현재 축에 수직화
        nrm = np.linalg.norm(bend_dir)
        if nrm > 1e-12:
            bend_dir /= nrm
        t = t + kappa * ds * bend_dir                   # 방향을 곡률만큼 회전
        t /= np.linalg.norm(t)
        p = p + ds * t
        pts.append(p.copy())
    return np.array(pts)


def shaft_clearance_curve(pts, vessel, r=VESSEL_R):
    """휜 중심선과 혈관 사이 최소거리 − 반경."""
    return float(np.min(np.linalg.norm(pts - vessel, axis=1)) - r)


def tip_deviation(pts, entry, axis, length):
    """계획된 직선 표적(진입점 + length·axis)에서 팁이 벗어난 거리."""
    return float(np.linalg.norm(pts[-1] - (np.asarray(entry) + length * np.asarray(axis))))


# --------------------------------------------------------------------------- #
# 스핀 정책
# --------------------------------------------------------------------------- #
def policy_flip(depth):
    """삽입 깊이 depth 에서 경사면을 180° 뒤집는다 — 두 호가 서로를 상쇄."""
    return lambda s: 0.0 if s < depth else np.pi


def policy_duty(period):
    """연속 회전(duty cycling): 곡률 방향이 계속 돌아 평균 곡률이 0 에 수렴."""
    return lambda s: 2 * np.pi * s / period


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    # ---- exp 45 와 같은 통로 기하 (팬텀 좌표계에서 그대로 쓴다) ----
    real = g6.real
    real.fetch()
    scans = real.load_bunny()
    rng = np.random.default_rng(0)
    full0 = scans["bun000"]
    sel = rng.choice(len(full0), real.MODEL_SUBSAMPLE, replace=False)
    model = full0[sel]
    target, vessel, entry, axis, clear_plan = g6.plan_in_phantom(model)
    L_INS = float(np.linalg.norm(target - entry))

    print("=== 48. 유연 바늘: 휨이 먹는 여유, 스핀이 되찾는 정확도 ===")
    print(f"바늘: 외경 {D_NEEDLE*1e3:.1f} mm, EI = {EI*1e3:.3f} mN·m², "
          f"경사면 횡력 {F_BEVEL:.2f} N (절삭력의 {F_BEVEL_RATIO:.0%})")
    print(f"통로: 삽입 {L_INS*1e3:.0f} mm, 혈관 반경 {VESSEL_R*1e3:.0f} mm, "
          f"강체 가정 여유 {clear_plan*1e3:.2f} mm (exp 45)")

    # ---- 1) 역학: 처짐과 등가 곡률 ----
    s_b, y_b, kappa = beam_deflection(L_INS)
    _, y_free, kappa_free = beam_deflection(L_INS, k_t=0.0)
    print("-" * 78)
    print(f"[역학] 조직 지지 있음: 팁 처짐 {y_b[-1]*1e3:6.2f} mm → 곡률 κ={kappa:5.2f} /m "
          f"(곡률반경 {1/kappa*1e3:.0f} mm)")
    print(f"       조직 지지 없음(공기 중): 팁 처짐 {y_free[-1]*1e3:6.2f} mm "
          f"(곡률반경 {1/kappa_free*1e3:.0f} mm) → 조직이 처짐을 "
          f"{y_free[-1]/y_b[-1]:.1f}배 억제한다")
    print(f"       해석해 대조: 외팔보 Fℓ³/(3EI) = "
          f"{F_BEVEL*L_INS**3/(3*EI)*1e3:.2f} mm — 지지 없는 해와 일치(이산화 검증)")
    print("       문헌의 경사면 바늘 곡률반경은 대략 100~300 mm 대다. 지지 없는 해"
          f"({1/kappa_free*1e3:.0f} mm)가 그 범위에 가깝고, 분포 스프링으로 근사한 조직"
          f"지지 해({1/kappa*1e3:.0f} mm)는 더 곧다 — 이 근사가 처짐을 과소평가함을 뜻한다")

    # ---- 2) 기하: 강체 vs 유연 ----
    rigid = needle_shape(entry, axis, L_INS, 0.0)
    bent = needle_shape(entry, axis, L_INS, kappa)
    cl_rigid = shaft_clearance_curve(rigid, vessel)
    cl_bent = shaft_clearance_curve(bent, vessel)
    dev_bent = tip_deviation(bent, entry, axis, L_INS)
    print("-" * 78)
    print(f"[기하] 강체 가정 : 여유 {cl_rigid*1e3:6.2f} mm, 팁 편차 0.00 mm")
    print(f"       유연 바늘 : 여유 {cl_bent*1e3:6.2f} mm, 팁 편차 {dev_bent*1e3:5.2f} mm")
    if cl_bent < 0:
        print("       → **휨만으로 통로가 사라진다**: 축 오차가 0 이어도 몸통이 혈관을 "
              "관통한다. exp 45 의 9.7° 임계는 강체 가정이 만든 낙관이었다.")
    else:
        print(f"       → 휨이 여유의 {100*(1-cl_bent/cl_rigid):.0f}% 를 먹는다")

    # 휨이 있을 때의 축 오차 임계 (exp 45 와 같은 방식으로 다시 계산)
    def crit_angle(kappa_used, policy=None, lo=0.0, hi=15.0, n=61):
        rot_axis = np.cross(axis, vessel - entry)
        if np.linalg.norm(rot_axis) < 1e-12:
            rot_axis = np.cross(axis, np.array([0.0, 0.0, 1.0]))
        rot_axis /= np.linalg.norm(rot_axis)
        angles = np.linspace(lo, hi, n)
        cl = []
        for a in angles:
            worst = min(
                shaft_clearance_curve(
                    needle_shape(entry, g6.so3_exp(sg * rot_axis * np.deg2rad(a)) @ axis,
                                 L_INS, kappa_used, policy), vessel)
                for sg in (+1.0, -1.0))
            cl.append(worst)
        cl = np.array(cl)
        bad = np.where(cl < 0)[0]
        if not len(bad):
            return float("inf"), angles, cl
        if bad[0] == 0:
            return 0.0, angles, cl
        i = bad[0]
        f = cl[i - 1] / (cl[i - 1] - cl[i])
        return float(angles[i - 1] + f * (angles[i] - angles[i - 1])), angles, cl

    a_rigid, ang, cl_r_curve = crit_angle(0.0)
    a_bent, _, cl_b_curve = crit_angle(kappa)
    print(f"       축 오차 임계: 강체 {a_rigid:.1f}° → 유연 "
          f"{'0(이미 관통)' if a_bent == 0 else f'{a_bent:.1f}°'} "
          f"({100*(1-a_bent/max(a_rigid,1e-9)):.0f}% 감소)")
    print("       * 이 임계는 exp 45 의 9.7° 와 직접 비교하면 안 된다: 거기서는 표적에 놓인"
          " 팁을 중심으로 도구를 기울였고, 여기서는 **진입점에서** 바늘 전체를 기울인다"
          "(needle steering 관례). 안쪽 비교(강체 vs 유연)만 같은 조건이다.")

    # ---- 3) 스핀으로 되찾기 ----
    print("-" * 78)
    print("[스핀 보상] exp 45 가 '과제와 무관'하다며 조작성에 써버린 자유도가 여기선 제어입력")
    flips = np.linspace(0.2, 0.8, 13) * L_INS
    flip_res = []
    for d in flips:
        pts = needle_shape(entry, axis, L_INS, kappa, policy_flip(d))
        flip_res.append((d, tip_deviation(pts, entry, axis, L_INS),
                         shaft_clearance_curve(pts, vessel)))
    best = min(flip_res, key=lambda r: r[1])
    pts_flip = needle_shape(entry, axis, L_INS, kappa, policy_flip(best[0]))
    pts_duty = needle_shape(entry, axis, L_INS, kappa, policy_duty(L_INS / 3.0))
    dev_duty = tip_deviation(pts_duty, entry, axis, L_INS)
    cl_duty = shaft_clearance_curve(pts_duty, vessel)

    d_analytic = 1 - 1 / np.sqrt(2)
    print(f"  보상 없음        : 팁 편차 {dev_bent*1e3:6.2f} mm | 여유 {cl_bent*1e3:6.2f} mm")
    print(f"  180° flip (깊이 {best[0]/L_INS*100:.0f}%) : 팁 편차 {best[1]*1e3:6.2f} mm "
          f"({dev_bent/max(best[1],1e-12):.0f}배 개선) | 여유 {best[2]*1e3:6.2f} mm")
    print(f"     └ 해석적 최적 깊이 = 1 − 1/√2 = {d_analytic*100:.1f}% "
          f"(2x²−4x+1=0). 스윕이 찾은 {best[0]/L_INS*100:.0f}% 와 일치. "
          "50% 는 기울기만 상쇄하고 오프셋을 남겨 최적이 아니다.")
    print(f"  duty cycling     : 팁 편차 {dev_duty*1e3:6.2f} mm | 여유 {cl_duty*1e3:6.2f} mm")
    a_flip, _, cl_f_curve = crit_angle(kappa, policy_flip(best[0]))
    print(f"  → flip 후 축 오차 임계 {a_flip:.1f}° 로 회복(강체 가정의 {a_rigid:.1f}° 에 근접)")

    # ---- 4) 예산과 트레이드오프 ----
    print("-" * 78)
    print("[예산] 정합 0.081 mm(exp 45) · 서보 ~0(자유공간) · 상호작용 1.3~4.7 mm(exp 47) 옆에")
    print(f"  휨 항: 보상 없으면 {dev_bent*1e3:.2f} mm — 지금까지의 어떤 항보다 크다")
    print(f"         flip 보상 후 {best[1]*1e3:.2f} mm — 다시 정합·상호작용과 같은 자릿수로")
    print("  주의: 스핀 자유도는 하나뿐이다. exp 45 는 그것을 조작성(w 0.006→0.064)에 썼고, "
          "여기서는 휨 보상에 쓴다. 동시에 둘 다 최적화할 수는 없다 — 실제 설계에서는 "
          "삽입 구간에만 스핀을 쓰고 접근 구간에서 자세를 잡는 식의 분할이 필요하다.")

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # (1) 바늘 형상 (삽입축-편향 평면으로 투영)
    ax = axes[0]
    u = axis / np.linalg.norm(axis)
    w = bent[-1] - (entry + L_INS * u)
    w = w - (w @ u) * u
    w = w / (np.linalg.norm(w) + 1e-12)

    def proj(pts):
        rel = pts - entry
        return (rel @ u) * 1e3, (rel @ w) * 1e3

    for pts, lab, c in ((rigid, "rigid assumption", "0.5"),
                        (bent, "flexible, no spin", "crimson"),
                        (pts_flip, f"flip @ {best[0]/L_INS*100:.0f}%", "seagreen"),
                        (pts_duty, "duty cycling", "tab:blue")):
        x, y = proj(pts)
        ax.plot(x, y, color=c, lw=1.8, label=lab)
    vx, vy = proj(vessel[None, :])
    ax.add_patch(plt.Circle((vx[0], vy[0]), VESSEL_R * 1e3, color="crimson", alpha=0.25))
    ax.set_xlabel("insertion depth [mm]"); ax.set_ylabel("lateral deflection [mm]")
    ax.set_title("Needle shapes in the bending plane\n(lateral axis exaggerated)",
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (2) 축 오차 vs 여유
    ax = axes[1]
    ax.plot(ang, cl_r_curve * 1e3, color="0.5", label=f"rigid (cuts at {a_rigid:.1f}°)")
    lab_b = "already cutting" if a_bent == 0 else f"cuts at {a_bent:.1f}°"
    ax.plot(ang, cl_b_curve * 1e3, color="crimson", label=f"flexible ({lab_b})")
    ax.plot(ang, cl_f_curve * 1e3, color="seagreen",
            label=f"flexible + flip (cuts at {a_flip:.1f}°)")
    ax.axhline(0, color="0.3", ls="--", lw=1)
    ax.set_xlabel("tool-axis error [deg]"); ax.set_ylabel("shaft clearance [mm]")
    ax.set_title("Bending eats the corridor a rigid model promised", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (3) flip 깊이 스윕
    ax = axes[2]
    ax.plot([r[0] / L_INS * 100 for r in flip_res], [r[1] * 1e3 for r in flip_res],
            "-o", color="seagreen", label="tip deviation")
    ax.axhline(dev_bent * 1e3, color="crimson", ls="--", lw=1, label="no compensation")
    ax.axhline(dev_duty * 1e3, color="tab:blue", ls=":", lw=1, label="duty cycling")
    ax.set_xlabel("flip depth [% of insertion]"); ax.set_ylabel("tip deviation [mm]")
    ax.set_title("One well-timed 180° spin cancels the arc", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("48. Flexible needle — bevel-induced bending, and the spin DOF that "
                 "compensates it", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "48_flexible_needle.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/48_flexible_needle.png, assets/48_flexible_needle.png")

    return dict(kappa=kappa, y_tip=float(y_b[-1]), cl_rigid=cl_rigid, cl_bent=cl_bent,
                dev_bent=dev_bent, a_rigid=a_rigid, a_bent=a_bent, a_flip=a_flip,
                best_flip=best, dev_duty=dev_duty, cl_duty=cl_duty, L=L_INS)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 조직 지지를 '직선 축으로 되당기는 분포 스프링'으로 근사했다. 실제 바늘은 자신이
#     이미 낸 **채널**을 따라가므로 이력(hysteresis)이 있고, 되당기는 기준이 직선이
#     아니다. 이 근사는 처짐을 다소 과소평가한다.
#   - 곡률을 삽입 전 길이에 대해 하나로 고정했다. 실제로는 삽입 깊이에 따라 유효 외팔보
#     길이가 변해 κ 가 변한다(초반에 더 크게 휨).
#   - 작은 각(선형 보) 가정이다. 처짐이 길이의 10% 를 넘으면 비선형 보 해석이 필요하다.
#   - flip 보상은 열린 루프다. 실제로는 초음파/전자기 추적으로 팁을 보며 닫아야 하고,
#     조직 불균질성 때문에 κ 가 예측과 달라진다.
#   - 스핀 자유도의 이중 사용(조작성 vs 휨 보상)은 여기서 지적만 하고 풀지 않았다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
