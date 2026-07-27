"""SimOpt: 시스템 식별로 sim-to-real 루프를 닫는다.

시뮬레이터로 학습한 정책을 실제 로봇에 올리면, 시뮬레이터의 물리 파라미터가
실제와 어긋난 만큼 성능이 무너진다(sim-to-real gap). 이 실험은 그 간극을
'한 번에 튜닝'하는 대신 피드백 루프로 좁힌다: 실제에서 굴려보고(act in real) →
그 로그로 시뮬레이터의 물리 파라미터를 실제에 맞게 보정하고(system ID) →
보정된 시뮬레이터에서 정책을 다시 학습하고 → 반복한다. 시뮬레이터가 실제로
수렴할수록 실제 성능이 함께 올라간다(Chebotar 2019, "Closing the Sim-to-Real Loop").

구성(전부 numpy, 결정론적):
- 플랜트: 밑바닥부터 구현한 카트-폴(도립진자). 숨겨진 '실제' 시스템은 참 물리
  파라미터(폴 질량 mp, 폴 길이 l)를 가지며, 시뮬레이터의 초기 추정치는 이와 크게
  어긋나 있다. 적분기와 모델 구조는 동일 — 오직 파라미터만 틀렸다(정직한 설정).
- 정책: 선형 상태피드백 F = -K·s. 현재 시뮬레이터 안에서 CEM(교차엔트로피법)으로
  학습(numpy만, torch/gym 없음).
- SimOpt 루프(K회 반복):
    1) 현재 시뮬레이터에서 CEM으로 정책 학습.
    2) 실제 시스템에 배치 → 균형유지 시간(성능) 측정 + 궤적 기록.
       기록 시 결정론적 탐침힘(probe)을 더해 파라미터가 관측 가능하도록 여기(勵起)한다.
    3) 시스템 식별: 같은 제어입력 아래 시뮬레이터의 1-스텝 예측이 실제 관측을
       재현하도록 파라미터를 보정(가중 최소자승, scipy.optimize). 실제 로그는
       매 반복 누적되어 데이터가 늘수록 추정이 정밀해진다.
    4) 시뮬레이터를 식별된 파라미터로 갱신하고 반복.

보여주는 것: (a) 시뮬레이터 파라미터 오차 → 0 으로 수렴, (b) 실제 균형유지 성능
상승. 정직한 기준선으로 '나쁜 시뮬레이터에서 한 번만 학습, 루프 없음'은 계속 나쁘다.
정직한 한계: 시스템 식별에는 정보량 있는 여기가 필요하고(관측 안 되는 파라미터는
수렴하지 않음), 관측 잡음 탓에 한 번의 식별은 편차가 있어 데이터 누적으로 좁힌다.

    python scripts/31_simopt_loop.py
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# 플랜트: 카트-폴 (도립진자). 상태 s=[x, xdot, theta, thetadot], theta=0 이 수직 상단.
# 알려진 상수: 카트 질량 mc, 중력 g, 적분 dt. 미지(식별 대상): 폴 질량 mp, 폴 길이 l.
# --------------------------------------------------------------------------
G = 9.81
MC = 1.0
DT = 0.02
T = 200                     # 에피소드 길이(스텝) = 4.0 s
F_MAX = 15.0                # 액추에이터 힘 한계 [N]
FALL = 0.40                 # 균형 실패 판정 각도 [rad] (~23도)
XLIM = 3.0                  # 카트 위치 한계 [m]

TRUE_PARAMS = np.array([0.50, 1.00])    # 실제: (mp, l) — 숨겨진 참값(무겁고 긴 폴)
INIT_GUESS = np.array([0.10, 0.30])     # 시뮬레이터 초기 추정: 크게 어긋남

IC_THETAS = np.array([0.08, -0.12, 0.15])   # 결정론적 초기각 집합 [rad]
K_LOOP = 5                                    # SimOpt 루프 반복 수


def _deriv(s, F, p):
    """카트-폴 연속시간 동역학. s:(...,4), F:(...), p=(mp,l). 균일봉 4/3 관성계수."""
    mp, l = p
    x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    st, ct = np.sin(th), np.cos(th)
    total = MC + mp
    temp = (F + mp * l * thd ** 2 * st) / total
    thdd = (G * st - ct * temp) / (l * (4.0 / 3.0 - mp * ct ** 2 / total))
    xdd = temp - mp * l * thdd * ct / total
    ds = np.stack([xd, xdd, thd, thdd], axis=-1)
    return ds


def rk4_step(s, F, p, dt=DT):
    k1 = _deriv(s, F, p)
    k2 = _deriv(s + 0.5 * dt * k1, F, p)
    k3 = _deriv(s + 0.5 * dt * k2, F, p)
    k4 = _deriv(s + dt * k3, F, p)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# --------------------------------------------------------------------------
# 정책 평가 (벡터화): 여러 K를 여러 초기조건에서 동시에 굴려 보상 합을 구한다.
# 보상 = cos(theta) - 위치벌점 - 제어벌점. cos 는 수직에서 1, 넘어지면 음수 → 매끈.
# --------------------------------------------------------------------------
def eval_policies(Kb, p, ic_thetas=IC_THETAS, horizon=T):
    M = Kb.shape[0]
    nic = len(ic_thetas)
    s0 = np.zeros((M, nic, 4))
    s0[:, :, 2] = ic_thetas[None, :]
    s = s0.reshape(M * nic, 4)
    Krep = np.repeat(Kb, nic, axis=0)          # (M*nic, 4)
    reward = np.zeros(M * nic)
    for _ in range(horizon):
        F = np.clip(-(s * Krep).sum(axis=1), -F_MAX, F_MAX)
        s = rk4_step(s, F, p)
        s[:, 0] = np.clip(s[:, 0], -XLIM * 2, XLIM * 2)
        reward += np.cos(s[:, 2]) - 0.02 * s[:, 0] ** 2 - 0.0005 * F ** 2
    return reward.reshape(M, nic).mean(axis=1)


def cem_train(p, seed, pop=48, iters=14, elite_frac=0.2):
    """현재 시뮬레이터(p) 안에서 선형피드백 K 를 CEM 으로 학습."""
    rng = np.random.default_rng(seed)
    mean = np.array([0.0, 0.0, 40.0, 8.0])     # 대략적 초기 평균(안정화 게인 규모)
    sigma = np.array([5.0, 5.0, 30.0, 15.0])
    n_elite = max(2, int(pop * elite_frac))
    best_K, best_r = mean.copy(), -1e18
    for _ in range(iters):
        Kb = mean[None, :] + sigma[None, :] * rng.standard_normal((pop, 4))
        r = eval_policies(Kb, p)
        idx = np.argsort(r)[::-1][:n_elite]
        elites = Kb[idx]
        mean = elites.mean(axis=0)
        sigma = elites.std(axis=0) + 1e-6
        if r[idx[0]] > best_r:
            best_r, best_K = r[idx[0]], Kb[idx[0]].copy()
    return best_K


# --------------------------------------------------------------------------
# 실제 시스템 상호작용
# --------------------------------------------------------------------------
def balancing_time(K, p, ic_thetas=IC_THETAS, horizon=T):
    """실제 성능 지표: 넘어지기 전 균형유지 스텝 수(초기조건 평균, 최대 horizon)."""
    times = []
    for th0 in ic_thetas:
        s = np.array([0.0, 0.0, th0, 0.0])
        t = 0
        for t in range(1, horizon + 1):
            F = float(np.clip(-K @ s, -F_MAX, F_MAX))
            s = rk4_step(s, F, p)
            if abs(s[2]) > FALL or abs(s[0]) > XLIM:
                return_t = t - 1
                break
        else:
            return_t = horizon
        times.append(return_t)
    return float(np.mean(times))


def collect_rollout(K, p, th0, rng, horizon=T, probe_amp=3.5, obs_std=None):
    """실제에서 궤적 로그 수집. 결정론적 탐침힘을 더해 파라미터를 여기(관측 가능화).
    관측 잡음을 더해 정직하게 만든다 → 1회 식별은 편차, 데이터 누적으로 좁힘."""
    if obs_std is None:
        obs_std = np.array([0.010, 0.03, 0.010, 0.03])
    s = np.array([0.0, 0.0, th0, 0.0])
    S_cur, F_log, S_next = [], [], []
    for t in range(horizon):
        probe = probe_amp * (np.sin(2 * np.pi * 0.7 * t * DT)
                             + 0.6 * np.sin(2 * np.pi * 1.9 * t * DT))
        F = float(np.clip(-K @ s + probe, -F_MAX, F_MAX))
        s_next = rk4_step(s, F, p)
        # 관측(잡음 포함)
        obs_cur = s + obs_std * rng.standard_normal(4)
        obs_next = s_next + obs_std * rng.standard_normal(4)
        S_cur.append(obs_cur)
        F_log.append(F)
        S_next.append(obs_next)
        s = s_next
        if abs(s[2]) > 1.2:      # 완전히 넘어가면 로그 종료(비현실 영역 제외)
            break
    return np.array(S_cur), np.array(F_log), np.array(S_next)


# --------------------------------------------------------------------------
# 시스템 식별: 같은 제어입력 아래 시뮬레이터의 1-스텝 예측이 실제 관측 다음상태를
# 재현하도록 (mp, l) 을 보정. 상태차원 스케일 차이는 화이트닝으로 정규화.
# --------------------------------------------------------------------------
def sysid_fit(dataset, p_init, p_true_scale=TRUE_PARAMS):
    S_cur = np.concatenate([d[0] for d in dataset], axis=0)
    F_log = np.concatenate([d[1] for d in dataset], axis=0)
    S_next = np.concatenate([d[2] for d in dataset], axis=0)
    delta = S_next - S_cur
    w = 1.0 / (delta.std(axis=0) + 1e-6)        # 차원별 화이트닝 가중

    def loss(logp):
        p = np.exp(logp)
        pred = rk4_step(S_cur, F_log, p)
        err = (S_next - pred) * w[None, :]
        return float(np.mean(err ** 2))

    res = minimize(loss, np.log(p_init), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 800})
    return np.exp(res.x)


def param_error(p):
    """참값 대비 상대 파라미터 오차(RMS, 정규화)."""
    return float(np.sqrt(np.mean(((p - TRUE_PARAMS) / TRUE_PARAMS) ** 2)))


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)

    # ---- SimOpt 루프 ----
    p_est = INIT_GUESS.copy()
    dataset = []
    param_err_hist, real_perf_hist, p_hist = [], [], []

    # 기준선(루프 없음): 나쁜 시뮬레이터에서 딱 한 번 학습한 정책
    K_noloop = cem_train(INIT_GUESS, seed=100)
    noloop_perf = balancing_time(K_noloop, TRUE_PARAMS)

    K_first = None
    rollout_first = None
    for k in range(K_LOOP):
        # 1) 현재 시뮬레이터에서 정책 학습
        K = cem_train(p_est, seed=100 + k)
        if k == 0:
            K_first = K.copy()
        # 2) 실제 배치 → 성능 측정 + 로그 수집
        perf = balancing_time(K, TRUE_PARAMS)
        param_err_hist.append(param_error(p_est))
        real_perf_hist.append(perf)
        p_hist.append(p_est.copy())
        # 여기용 로그(탐침 포함)를 매 반복 1개씩 수집해 데이터셋에 누적
        # (데이터가 쌓일수록 관측잡음에 대한 식별 편차가 줄어 파라미터오차가 점차 감소)
        th0 = 0.20 if k % 2 == 0 else -0.20
        dataset.append(collect_rollout(K, TRUE_PARAMS, th0, rng))
        if k == 0:
            rollout_first = collect_rollout(K, TRUE_PARAMS, 0.20,
                                            np.random.default_rng(7), probe_amp=3.5)
        # 3) 시스템 식별 → 4) 시뮬레이터 갱신
        p_est = sysid_fit(dataset, p_est)

    # 마지막 갱신된 파라미터로 한 번 더 학습/배치(최종 성능)
    K_final = cem_train(p_est, seed=200)
    final_perf = balancing_time(K_final, TRUE_PARAMS)
    param_err_hist.append(param_error(p_est))
    real_perf_hist.append(final_perf)
    p_hist.append(p_est.copy())

    param_err_hist = np.array(param_err_hist)
    real_perf_hist = np.array(real_perf_hist)
    p_hist = np.array(p_hist)
    iters = np.arange(len(param_err_hist))

    initial_real = real_perf_hist[0]
    final_real = real_perf_hist[-1]
    initial_perr = param_err_hist[0]
    final_perr = param_err_hist[-1]

    print("=== SimOpt: sim-to-real 루프 닫기 (카트-폴 시스템 식별) ===")
    print(f"참 파라미터  (mp,l) = ({TRUE_PARAMS[0]:.3f}, {TRUE_PARAMS[1]:.3f})")
    print(f"초기 추정   (mp,l) = ({INIT_GUESS[0]:.3f}, {INIT_GUESS[1]:.3f})  "
          f"→ 파라미터오차 {initial_perr:.3f}")
    print(f"식별 최종   (mp,l) = ({p_hist[-1,0]:.3f}, {p_hist[-1,1]:.3f})  "
          f"→ 파라미터오차 {final_perr:.3f}  ({100*(1-final_perr/initial_perr):.0f}% 감소)")
    print("-" * 60)
    print(f"{'iter':>4} {'param_err':>10} {'real_balance[steps]':>20}")
    for i in iters:
        print(f"{i:>4} {param_err_hist[i]:>10.3f} {real_perf_hist[i]:>20.1f}")
    print("-" * 60)
    print(f"실제 균형유지  초기 {initial_real:.1f} → 최종 {final_real:.1f} "
          f"(최대 {T}) / 루프없음 기준선 {noloop_perf:.1f} (계속 나쁨)")

    # ---------------- 플롯 ----------------
    fig = plt.figure(figsize=(13.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(iters, param_err_hist, "o-", color="#1f77b4", lw=2)
    ax0.axhline(0, color="gray", ls=":", lw=1)
    ax0.set_title("System ID: sim params → real (error ↓)", fontsize=10)
    ax0.set_xlabel("SimOpt iteration")
    ax0.set_ylabel("normalized param error")
    ax0.set_ylim(bottom=-0.03)
    ax0.grid(alpha=0.25)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(iters, real_perf_hist, "o-", color="#2ca02c", lw=2, label="SimOpt (loop)")
    ax1.axhline(noloop_perf, color="#d9534f", ls="--", lw=1.8,
                label=f"no loop (bad sim): {noloop_perf:.0f}")
    ax1.axhline(T, color="gray", ls=":", lw=1, label=f"max = {T}")
    ax1.set_title("Real-world balancing (↑ as sim converges)", fontsize=10)
    ax1.set_xlabel("SimOpt iteration")
    ax1.set_ylabel("balancing time [steps]")
    ax1.set_ylim(-5, T + 15)
    ax1.legend(fontsize=8, loc="center right")
    ax1.grid(alpha=0.25)

    # 궤적 오버레이: 실제 rollout vs 초기(나쁜)파라미터 예측 vs 식별후 예측
    ax2 = fig.add_subplot(gs[0, 2])
    S_cur, F_log, S_next = rollout_first
    tt = np.arange(len(F_log)) * DT
    # 같은 제어입력으로 초기추정/최종식별 파라미터가 만드는 궤적(오픈루프 재생)
    def replay(p):
        s = S_cur[0].copy()
        out = [s[2]]
        for i in range(len(F_log)):
            s = rk4_step(s, F_log[i], p)
            out.append(s[2])
        return np.array(out[:len(tt)])
    ax2.plot(tt, S_next[:len(tt), 2], color="k", lw=2.2, label="real rollout")
    ax2.plot(tt, replay(INIT_GUESS), color="#d9534f", lw=1.6, ls="--",
             label="sim BEFORE ID (bad params)")
    ax2.plot(tt, replay(p_hist[-1]), color="#1f77b4", lw=1.6, ls="-.",
             label="sim AFTER ID (fitted)")
    ax2.set_title("Sim reproduces real under same controls", fontsize=10)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("pole angle θ [rad]")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(alpha=0.25)

    fig.suptitle("SimOpt — closing the sim-to-real loop by system identification",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    for path in ("outputs/31_simopt_loop.png", "assets/31_simopt_loop.png"):
        fig.savefig(path, dpi=130, bbox_inches="tight")
    print("\n[plot] outputs/31_simopt_loop.png, assets/31_simopt_loop.png")

    return initial_real, final_real, initial_perr, final_perr


if __name__ == "__main__":
    main()
