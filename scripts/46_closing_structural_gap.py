"""구조 간극 닫기: 루프가 "남았다"고 알려준 물리를 모델에 넣으면 어디까지 내려가나.

exp 43 의 sim-to-real 루프는 두 갈래로 끝났다.
  - **파라미터 간극**(도구 페이로드 + 점성·쿨롱 마찰): 회귀 한 번으로 45.9 → 0.003 mm.
  - **구조 간극**(스트라이벡 스틱션): 회귀자에 대응 열이 아예 없어 **0.207 mm 에서 정체**.
    같은 데이터를 아무리 더 모아도 내려가지 않았다.

그때의 결론은 "다음 수는 재식별이 아니라 모델 구조 확장"이었다. 이 실험은 그 말을
실행하고, **정말 닫히는지 · 얼마나 닫히는지 · 무엇이 대가인지**를 잰다.

--- 확장의 방법: 비선형 파라미터를 밖으로 빼기 ---
스트라이벡 마찰은
    f(q̇) = fv·q̇ + [fc + (fs − fc)·exp(−(q̇/vs)²)]·sign(q̇)
로, fs 는 선형이지만 **vs(스트라이벡 속도)는 지수 안에 들어가 비선형**이다. 그래서
회귀를 그대로 쓸 수 없다. 표준 처리는 비선형 파라미터를 **바깥에서 격자 탐색**하고,
안쪽은 선형 최소자승으로 푸는 것이다(separable least squares).

    for vs in 격자:  π̂(vs) = argmin ‖Y(vs)·π − τ‖  →  잔차가 최소인 vs 채택

확장 회귀자 π = [a, b, d, G1, G2, fv₁, fv₂, fc₁, fc₂, **fs₁, fs₂**] (11개).

--- 정직하게 물을 것 ---
  1) 구조를 맞게 늘리면 정체가 풀리는가? (예상: 예)
  2) vs 를 틀리게 잡으면 어떻게 되는가? (부분적으로만 닫힌다 — 격자 탐색이 필요한 이유)
  3) 늘 늘리는 게 답인가? **구조 간극이 없는 팔에 확장 모델을 쓰면** 잉여 파라미터가
     잡음을 학습해 오히려 나빠지는가? (모델 선택의 대가)

    python scripts/46_closing_structural_gap.py
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
s2r = import_module("43_sim_to_real_arm")       # 플랜트·루프·식별 자산 재사용

VS_GRID = np.array([0.02, 0.035, 0.05, 0.07, 0.10, 0.15])   # 스트라이벡 속도 후보 [rad/s]
N_ITERS = 3


# --------------------------------------------------------------------------- #
# 확장 모델: 회귀자에 스트라이벡 열 2개 추가 (vs 는 밖에서 고정)
# --------------------------------------------------------------------------- #
def regressor_ext(q, qd, qdd, vs):
    """exp 43 의 9열 회귀자 + 관절별 스트라이벡 열 2개 = (2, 11)."""
    Y9 = s2r.regressor(q, qd, qdd)
    Y = np.zeros((2, 11))
    Y[:, :9] = Y9
    for j in range(2):
        # 스트라이벡 항: exp(−(q̇/vs)²)·tanh(q̇/ε) — 저속에서만 켜지는 추가 마찰
        Y[j, 9 + j] = np.exp(-(qd[j] / vs) ** 2) * np.tanh(qd[j] / 0.01)
    return Y


def friction_ext(pi, qd, vs):
    """확장 파라미터로 계산한 마찰(제어기 피드포워드용)."""
    fv, fc, fs = pi[5:7], pi[7:9], pi[9:11]
    return (fv * qd + fc * np.tanh(qd / 0.01)
            + fs * np.exp(-(qd / vs) ** 2) * np.tanh(qd / 0.01))


def inverse_dynamics_ext(pi, q, qd, qdd, vs):
    return (s2r.M_of(pi, q) @ qdd + s2r.Cqd_of(pi, q, qd) + s2r.g_of(pi, q)
            + friction_ext(pi, qd, vs))


def computed_torque_ext(pi_hat, q, qd, q_d, qd_d, qdd_d, vs):
    e, edot = q_d - q, qd_d - qd
    return (s2r.M_of(pi_hat, q) @ (qdd_d + s2r.KP * e + s2r.KD * edot)
            + s2r.Cqd_of(pi_hat, q, qd) + s2r.g_of(pi_hat, q)
            + friction_ext(pi_hat, qd, vs))


def identify_ext(logs, vs, decimate=5):
    """확장 회귀자로 최소자승. 반환 (π̂(11), cond, 토크 잔차 RMS)."""
    rows, rhs = [], []
    for q, qd, qdd, tau in logs:
        for k in range(0, len(q), decimate):
            rows.append(regressor_ext(q[k], qd[k], qdd[k], vs))
            rhs.append(tau[k])
    A = np.vstack(rows)
    y = np.concatenate(rhs)
    pi_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.sqrt(np.mean((A @ pi_hat - y) ** 2)))
    return pi_hat, float(np.linalg.cond(A.T @ A)), resid


def excitation_lowspeed(t_end=8.0, seed=0):
    """**저속** 여기 궤적: 스틱션은 |q̇| < vs 에서만 켜지므로 그 구간에 오래 머물러야 한다.

    exp 43 의 다중사인 여기는 관성·코리올리를 잘 흔들지만 속도가 0.5~2 rad/s 라
    스트라이벡 항이 거의 켜지지 않는다 — 모델을 늘려도 **데이터에 그 물리가 없으면**
    식별되지 않는다. 여기서는 진폭을 줄이고 주파수를 낮춰 최대 속도를 ~0.1 rad/s 로
    두고, 부호 반전을 여러 번 만든다(쿨롱·스틱션 구분에 필요)."""
    rng = np.random.default_rng(1000 + seed)
    t = np.arange(0, t_end, s2r.DT)
    Q = np.zeros((len(t), 2))
    Qd = np.zeros_like(Q)
    Qdd = np.zeros_like(Q)
    for j in range(2):
        Q[:, j] = [0.6, -1.1][j]
        for w, amp in [(0.55 + 0.12 * j, 0.13), (1.1 + 0.2 * j, 0.04)]:
            ph = rng.uniform(0, 6.28)
            Q[:, j] += amp * np.sin(w * t + ph)
            Qd[:, j] += amp * w * np.cos(w * t + ph)
            Qdd[:, j] += -amp * w ** 2 * np.sin(w * t + ph)
    return Q, Qd, Qdd


def identify_ext_search(logs, grid=VS_GRID):
    """비선형 파라미터 vs 는 격자 탐색, 나머지는 선형 최소자승(separable LS)."""
    best = None
    curve = []
    for vs in grid:
        pi_hat, cond, resid = identify_ext(logs, vs)
        curve.append((float(vs), resid))
        if best is None or resid < best[3]:
            best = (pi_hat, float(vs), cond, resid)
    return best, curve


# --------------------------------------------------------------------------- #
# 확장 모델로 배치 (실기는 항상 스트라이벡 포함)
# --------------------------------------------------------------------------- #
def rollout_ext(pi_hat, vs, Q_des, Qd_des, Qdd_des, rng=None, log=False,
                stribeck=True):
    """확장 제어기로 실기를 추종. exp 43 의 rollout 과 같은 구조, 제어식만 확장."""
    n = len(Q_des)
    q, qd = Q_des[0].copy(), Qd_des[0].copy()
    Q = np.zeros((n, 2))
    TIP = np.zeros((n, 2))
    TAU = np.zeros((n, 2))
    for k in range(n):
        tau = computed_torque_ext(pi_hat, q, qd, Q_des[k], Qd_des[k], Qdd_des[k], vs)
        Q[k], TAU[k] = q, tau
        TIP[k] = s2r.kin.fk(q, L=s2r.L_ARM)[:2]
        if k == n - 1:
            break

        def deriv(s):
            return np.concatenate(
                [s[2:], s2r.forward_dynamics(s2r.PI_TRUE, s[:2], s[2:], tau, stribeck)])

        st = np.concatenate([q, qd])
        k1 = deriv(st)
        k2 = deriv(st + 0.5 * s2r.DT * k1)
        k3 = deriv(st + 0.5 * s2r.DT * k2)
        k4 = deriv(st + s2r.DT * k3)
        st = st + (s2r.DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q, qd = st[:2], st[2:]

    if not log:
        return Q, TIP, TAU
    rng = np.random.default_rng(0) if rng is None else rng
    Q_meas = Q + rng.normal(0, s2r.ENC_NOISE, Q.shape)
    TAU_meas = TAU + rng.normal(0, s2r.TORQUE_NOISE, TAU.shape)
    return Q, TIP, TAU, s2r.smooth_derivatives(Q_meas, TAU_meas)


def deploy_ext(pi_hat, vs, task, stribeck=True):
    """확장 모델 제어기를 임상 과제에 배치. 반환 (표적오차, 추종잔차)."""
    Q_des, Qd_des, Qdd_des, path, target_rob, _ = task
    _, TIP, _ = rollout_ext(pi_hat, vs, Q_des, Qd_des, Qdd_des, stribeck=stribeck)
    if not np.all(np.isfinite(TIP)):
        return np.inf, np.inf
    tip_err = np.linalg.norm(TIP - path, axis=1)
    return (float(np.linalg.norm(TIP[-1] - target_rob)),
            float(np.sqrt(np.mean(tip_err ** 2))))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    task = s2r.clinical_task()
    fs_true = s2r.FC_TRUE * (s2r.FS_RATIO - 1.0)      # 확장 파라미터의 참값

    print("=== 46. 구조 간극 닫기: 마찰 모델을 늘려 exp 43 의 정체를 푼다 ===")
    print(f"실기의 숨은 물리: 스트라이벡 스틱션 (fs/fc = {s2r.FS_RATIO}, "
          f"vs = {s2r.V_STRIBECK} rad/s) — exp 43 의 9열 모델에는 대응 열이 없었다")

    # ---- 기준선 재현: exp 43 의 9열 모델로 루프를 돌리면 정체한다 ----
    rng = np.random.default_rng(7)
    pi9 = s2r.PI_NOMINAL.copy()
    logs = []
    base_hist = [s2r.deploy(pi9, task, stribeck=True)[0]]
    for it in range(1, N_ITERS + 1):
        Qe, Qde, Qdde = s2r.excitation_trajectory(seed=it)
        *_, meas = s2r.rollout(s2r.PI_TRUE, pi9, Qe, Qde, Qdde, rng=rng, log=True,
                               stribeck=True)
        logs.append(meas)
        pi9, _, res9 = s2r.identify(logs)
        base_hist.append(s2r.deploy(pi9, task, stribeck=True)[0])
    print(f"[기준선] 9열 모델 루프: {base_hist[0]*1e3:.1f} → "
          f"{base_hist[-1]*1e3:.3f} mm 에서 정체 (토크 잔차 {res9:.3f} N·m)")

    # ---- 확장 1: 구조만 늘리고 데이터는 그대로 ----
    (pi11_fast, vs_fast, _, res11_fast), _ = identify_ext_search(logs)
    err_fast, _ = deploy_ext(pi11_fast, vs_fast, task)
    print(f"[확장 1] 구조만 확장(같은 고속 로그) → vs_hat {vs_fast:.3f}, 토크잔차 "
          f"{res11_fast:.3f} N·m, 표적오차 {err_fast*1e3:.3f} mm")
    print(f"         추정 fs {np.round(pi11_fast[9:11], 3).tolist()} vs 참값 "
          f"{np.round(fs_true, 3).tolist()} → **거의 식별되지 않는다**")
    print("         이유: 스틱션은 |q̇| < vs(0.05) 에서만 켜지는데 고속 여기 궤적은 "
          "그 구간을 스쳐 지나간다 — 모델을 늘려도 데이터에 그 물리가 없으면 소용없다")

    # ---- 확장 2: 구조 + 저속 여기 데이터 ----
    logs_all = list(logs)
    for it in range(1, 3):
        Qs, Qds, Qdds = excitation_lowspeed(seed=it)
        *_, meas_s = s2r.rollout(s2r.PI_TRUE, pi9, Qs, Qds, Qdds,
                                 rng=np.random.default_rng(500 + it), log=True,
                                 stribeck=True)
        logs_all.append(meas_s)
    (pi11, vs_hat, cond11, res11), curve = identify_ext_search(logs_all)
    err_ext, resid_ext = deploy_ext(pi11, vs_hat, task)
    frac = np.mean(np.abs(np.concatenate([m[1].ravel() for m in logs_all[len(logs):]]))
                   < s2r.V_STRIBECK)
    print(f"[확장 2] 구조 확장 + **저속 여기** 추가(저속 로그의 {frac*100:.0f}% 샘플이 "
          f"|q̇| < vs) → vs_hat {vs_hat:.3f} (참값 {s2r.V_STRIBECK})")
    print(f"         추정 fs {np.round(pi11[9:11], 3).tolist()} vs 참값 "
          f"{np.round(fs_true, 3).tolist()}, 토크잔차 {res11:.3f} N·m")
    print(f"         표적오차 {base_hist[-1]*1e3:.3f} → **{err_ext*1e3:.3f} mm** "
          f"({base_hist[-1]/max(err_ext,1e-12):.0f}배), 추종잔차 {resid_ext*1e3:.3f} mm")

    # 파라미터 간극만 있던 경우의 바닥(exp 43 루프 A)과 비교
    pi_paramonly = s2r.identify(logs)[0]
    floor_ref = 0.003e-3    # exp 43 루프 A 의 최종 표적오차 [m] (기록값)
    ratio_floor = err_ext / floor_ref
    verdict_floor = ("구조 간극이 사실상 완전히 닫혔다" if ratio_floor < 3
                     else "닫혔지만 근사 한계가 남는다")
    print(f"         구조 간극이 없던 경우의 바닥 {floor_ref*1e3:.3f} mm 대비 "
          f"{ratio_floor:.1f}배 → {verdict_floor}")

    # ---- vs 를 틀리게 잡으면? ----
    print("-" * 78)
    print("[vs 민감도] 비선형 파라미터를 격자로 찾아야 하는 이유")
    sens = []
    for vs in VS_GRID:
        pi_v, _, res_v = identify_ext(logs_all, vs)
        e_v, _ = deploy_ext(pi_v, vs, task)
        sens.append((float(vs), res_v, e_v))
        mark = " ←선택" if abs(vs - vs_hat) < 1e-9 else ""
        print(f"  vs={vs:5.3f}: 토크잔차 {res_v:.3f} N·m | 표적오차 {e_v*1e3:7.3f} mm{mark}")

    # ---- 반대 방향의 정직성: 구조 간극이 없는데 확장 모델을 쓰면? ----
    print("-" * 78)
    print("[모델 선택의 대가] 구조 간극이 **없는** 팔(스틱션 없음)에 두 모델을 각각 적용")
    logs_clean = []
    pi_c = s2r.PI_NOMINAL.copy()
    for it in range(1, N_ITERS + 1):
        Qe, Qde, Qdde = s2r.excitation_trajectory(seed=it)
        *_, meas = s2r.rollout(s2r.PI_TRUE, pi_c, Qe, Qde, Qdde,
                               rng=np.random.default_rng(31 + it), log=True,
                               stribeck=False)
        logs_clean.append(meas)
        pi_c, _, _ = s2r.identify(logs_clean)
    for it in range(1, 3):                       # 확장 모델과 같은 데이터 조건으로
        Qs, Qds, Qdds = excitation_lowspeed(seed=it)
        *_, meas_s = s2r.rollout(s2r.PI_TRUE, pi_c, Qs, Qds, Qdds,
                                 rng=np.random.default_rng(700 + it), log=True,
                                 stribeck=False)
        logs_clean.append(meas_s)
    pi_c = s2r.identify(logs_clean)[0]
    err9_clean = s2r.deploy(pi_c, task, stribeck=False)[0]
    (pi11_c, vs_c, _, _), _ = identify_ext_search(logs_clean)
    err11_clean, _ = deploy_ext(pi11_c, vs_c, task, stribeck=False)
    print(f"  9열(맞는 구조) : 표적오차 {err9_clean*1e6:8.1f} µm")
    print(f"  11열(과잉 구조): 표적오차 {err11_clean*1e6:8.1f} µm "
          f"(추정 fs {np.round(pi11_c[9:11],3).tolist()} ≈ 0 이어야 정상)")
    ratio = err11_clean / max(err9_clean, 1e-12)
    verdict = "잉여 파라미터가 잡음을 학습해 손해" if ratio > 1.5 else \
        "잉여 파라미터가 0 근처로 추정돼 실질 손해 없음"
    print(f"  → {ratio:.1f}배 — {verdict}")

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    its = np.arange(len(base_hist))
    ax.semilogy(its, np.array(base_hist) * 1e3, "-o", color="crimson",
                label="exp 43 model (9 params)")
    ax.axhline(err_fast * 1e3, color="tab:orange", ls="-.",
               label=f"11 params, same data: {err_fast*1e3:.3f} mm")
    ax.axhline(err_ext * 1e3, color="seagreen", ls="--",
               label=f"11 params + low-speed data: {err_ext*1e3:.3f} mm")
    ax.axhline(floor_ref * 1e3, color="0.5", ls=":",
               label=f"no-structural-gap floor: {floor_ref*1e3:.3f} mm")
    ax.set_xlabel("sim-to-real iteration"); ax.set_ylabel("target error [mm]")
    ax.set_title("Extending the model releases the plateau", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8); ax.set_xticks(its)

    ax = axes[1]
    vs_a = np.array([s[0] for s in sens])
    ax.plot(vs_a, [s[1] for s in sens], "-o", color="tab:blue", label="torque residual")
    ax.axvline(s2r.V_STRIBECK, color="0.4", ls=":", label="true vs")
    ax.axvline(vs_hat, color="seagreen", ls="--", label="selected vs")
    ax.set_xlabel("Stribeck velocity vs [rad/s]"); ax.set_ylabel("torque residual [N·m]")
    ax.set_title("Nonlinear parameter: grid search outside, LS inside", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(vs_a, [s[2] * 1e3 for s in sens], "-s", color="tab:orange", alpha=0.7)
    ax2.set_ylabel("target error [mm]", color="tab:orange")

    ax = axes[2]
    names = ["fv₁", "fv₂", "fc₁", "fc₂", "fs₁", "fs₂"]
    true_vals = np.concatenate([s2r.FV_TRUE, s2r.FC_TRUE, fs_true])
    est9 = np.concatenate([pi_paramonly[5:9], [0.0, 0.0]])
    est11 = np.concatenate([pi11[5:9], pi11[9:11]])
    xs = np.arange(len(names))
    ax.bar(xs - 0.26, true_vals, 0.25, label="true", color="0.6")
    ax.bar(xs, est9, 0.25, label="9-param fit", color="crimson")
    ax.bar(xs + 0.26, est11, 0.25, label="11-param fit", color="seagreen")
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("friction parameter")
    ax.set_title("The 9-param fit absorbs stiction into Coulomb terms", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("46. Closing the structural gap — extend the model the loop said was "
                 "missing", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "46_closing_structural_gap.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/46_closing_structural_gap.png, "
          "assets/46_closing_structural_gap.png")

    return dict(base_hist=base_hist, err_ext=err_ext, err_fast=err_fast,
                vs_hat=vs_hat, res9=res9, res11=res11, sens=sens,
                err9_clean=err9_clean, err11_clean=err11_clean, pi11=pi11,
                pi11_fast=pi11_fast, fs_true=fs_true)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 여기서 늘린 구조는 '정답을 알고' 늘린 것이다. 실제로는 무엇이 빠졌는지 모르므로,
#     잔차의 패턴(속도 부호 반전 근처에서 커지는가 등)을 보고 후보를 세워야 한다.
#   - 스트라이벡 모델도 근사다. 실제 스틱션은 이력(hysteresis)·정지시간 의존성이 있어,
#     확장 후에도 파라미터 간극만 있던 경우의 바닥까지는 내려가지 않는다.
#   - 비선형 파라미터(vs)는 격자 탐색이다. 차원이 늘면 이 방식은 곧 한계에 부딪히고,
#     비선형 최소자승·베이즈 추정이 필요해진다.
#   - 모델을 늘리는 데는 대가가 있다(마지막 실험). 구조 간극이 없을 때 잉여 파라미터가
#     무해한지 여부는 여기의 잡음·여기 조건에 의존하며, 일반적 보장이 아니다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
