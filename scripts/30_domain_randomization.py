"""도메인 랜덤화(domain randomization): 시뮬레이터에서 배운 정책을 '현실'로 옮긴다.

로봇 정책을 시뮬레이터에서 학습해 실제 하드웨어에 올리면, 시뮬과 현실의 물리
차이(sim-to-real gap) 때문에 성능이 무너지는 일이 흔하다. 특히 제어기를 '명목(nominal)'
파라미터 하나에만 맞춰 튜닝하면, 시뮬 상에서는 완벽하지만 실제 질량·길이·구동 지연이
조금만 달라져도 발산한다. 이 과적합을 정면으로 다루는 표준 처방이 도메인 랜덤화다.
학습 중 물리 파라미터를 매 에피소드 무작위로 흔들어(uniform 샘플링), 정책이 '하나의
세계'가 아니라 '세계의 가족(family of worlds)' 전체에서 작동하도록 강제한다. 그러면
정책은 어느 한 세계에 맞춘 공격적 튜닝을 포기하는 대신, 파라미터가 이동해도 견디는
보수적이고 강건한 해로 수렴한다.

이 실험은 그 트레이드오프를 숫자로 보인다. 과제는 고전적인 카트-폴 균형(도립진자)이며,
카트-폴 ODE를 직접 구현하고 세미-임플리싯 오일러로 적분한다. 정책은 4-가중치 선형
상태피드백 f = clip(w·state, ±F_max)이고, 미분 불가·시뮬레이터 기반 목적함수에 강건한
교차엔트로피법(CEM, cross-entropy method)으로 학습한다 — 가우시안에서 가중치를 뽑아
균형 보상으로 평가하고 상위 엘리트로 분포를 재적합하는 과정을 반복한다(torch/gym 없이
numpy만 사용). 두 정책을 비교한다:
  (a) 명목 학습(nominal-only): 명목 파라미터에서만 CEM.
  (b) 도메인 랜덤화(DR): 매 평가 에피소드마다 폴 질량·길이·카트 질량·구동 지연을
      무작위로 뽑아 그 가족 위에서 CEM.
평가는 명목값을 이동시킨 '현실' 세계의 격자(구동 지연 × 폴 길이)에서 두 정책의 성공률
(에피소드를 임계시간 이상 세워 유지한 비율)을 측정한다. 격자에는 DR 학습 범위 '바깥'
구간도 포함해 외삽(extrapolation)까지 정직하게 드러낸다.

정직한 결과: 명목 정책은 시뮬-지연 0에서 완벽하지만 구동 지연이 조금만 늘어도 급격히
무너진다(공격적 고이득 튜닝의 대가). DR 정책은 명목에서도 완벽하면서 지연이 커져도
넓은 구간에서 균형을 유지한다 — 다만 학습 범위를 크게 벗어난 극단적 지연에서는 DR도
결국 실패한다. 즉 DR은 약간의 최고성능(peak)을 내주고 강건성(robustness)을 산다.

    python scripts/30_domain_randomization.py
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 물리 상수 / 과제 설정
# --------------------------------------------------------------------------
G = 9.8                 # 중력 [m/s^2]
DT = 0.02               # 적분 스텝 [s]
MAX_STEPS = 200         # 에피소드 최대 길이 (= 4초)
THETA_LIM = 0.35        # 낙하 판정 각도 [rad] (~20도)
X_LIM = 2.4             # 트랙 이탈 판정 [m]
F_MAX = 15.0            # 구동력 포화 [N]
SUCCESS_FRAC = 0.95     # 이 비율 이상 스텝을 버티면 '성공'

# 명목(sim) 세계: (카트질량 M, 폴질량 m, 폴 반길이 l, 구동지연 lat[step], 센서잡음 noise)
NOMINAL = (1.0, 0.1, 0.5, 0, 0.0)

# 도메인 랜덤화 학습 범위 (uniform)
DR_M = (0.8, 1.5)
DR_m = (0.05, 0.4)
DR_l = (0.35, 0.75)
DR_LAT_MAX = 5          # 지연 0..5 스텝을 학습 중 무작위로 경험

# CEM 하이퍼파라미터
CEM_ITERS = 14
CEM_POP = 48
CEM_ELITE = 8
CEM_K = 6               # 후보 하나당 평가 세계 수(평균 → 강건 목적함수)
TRAIN_SEED = 0

# 평가 격자: 구동 지연(행) × 폴 길이(열). DR 학습 범위 바깥(지연 6~9)까지 포함.
LAT_GRID = list(range(0, 10))
LEN_GRID = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
EVAL_EPS = 20
EVAL_SEED = 1234


# --------------------------------------------------------------------------
# 카트-폴 동역학 (Barto/Sutton 도립진자, 세미-임플리싯 오일러)
#   state = [x, x_dot, theta, theta_dot], theta=0 이 수직 상방
# --------------------------------------------------------------------------
def cartpole_step(s, force, M, m, l):
    x, xd, th, thd = s
    st, ct = np.sin(th), np.cos(th)
    tot = M + m
    temp = (force + m * l * thd * thd * st) / tot
    thdd = (G * st - ct * temp) / (l * (4.0 / 3.0 - m * ct * ct / tot))
    xdd = temp - m * l * thdd * ct / tot
    xd = xd + DT * xdd          # 세미-임플리싯: 속도 먼저 갱신
    thd = thd + DT * thdd
    x = x + DT * xd
    th = th + DT * thd
    return np.array([x, xd, th, thd])


def rollout(w, world, rng, max_steps=MAX_STEPS, th0=0.18):
    """한 세계에서 선형정책 w로 1 에피소드. (정형화 보상 R, 생존 스텝수) 반환.

    보상은 매 스텝 '수직에 얼마나 가까운지'를 적분한다: 낙하하면 남은 스텝의 보상을
    잃으므로 생존과 정밀 조절을 동시에 보상한다. 명목에서는 이 정밀 보상이 공격적
    고이득을 부추기고(지연에 취약), DR에서는 지연 세계까지 평균되어 보수적 해로 이끈다.
    """
    M, m, l, lat, noise = world
    s = np.array([0.0, 0.0, rng.uniform(-th0, th0), 0.0]) + rng.uniform(-0.02, 0.02, 4)
    buf = [0.0] * (lat + 1)     # 구동 지연 큐
    R = 0.0
    steps = 0
    for _ in range(max_steps):
        obs = s + rng.normal(0, noise, 4) if noise > 0 else s
        a = float(np.clip(w @ obs, -F_MAX, F_MAX))
        if lat > 0:
            buf.append(a)
            a = buf.pop(0)
        s = cartpole_step(s, a, M, m, l)
        if abs(s[2]) > THETA_LIM or abs(s[0]) > X_LIM:
            break
        R += 1.0 - (s[2] / THETA_LIM) ** 2 - 0.05 * (s[0] / X_LIM) ** 2
        steps += 1
    return R, steps


def episode_return(w, world, rng):
    return rollout(w, world, rng)[0]


# --------------------------------------------------------------------------
# 교차엔트로피법(CEM): 가우시안 샘플 → 엘리트 재적합 반복
# --------------------------------------------------------------------------
def cem_train(sample_worlds, seed=TRAIN_SEED,
              iters=CEM_ITERS, pop=CEM_POP, elite=CEM_ELITE, K=CEM_K):
    rng = np.random.default_rng(seed)
    mean = np.zeros(4)
    std = np.array([3.0, 3.0, 25.0, 5.0])       # theta 성분에 큰 초기 분산
    for _ in range(iters):
        W = rng.normal(mean, std, size=(pop, 4))
        scores = np.empty(pop)
        for i in range(pop):
            worlds = sample_worlds(rng, K)
            scores[i] = np.mean([episode_return(W[i], wd, rng) for wd in worlds])
        idx = np.argsort(scores)[-elite:]
        mean = W[idx].mean(0)
        std = W[idx].std(0) + 1e-3
    return mean


def nominal_worlds(rng, K):
    """명목 학습: 항상 명목 세계."""
    return [NOMINAL] * K


def dr_worlds(rng, K):
    """도메인 랜덤화: 물리 파라미터·구동 지연을 매번 무작위."""
    out = []
    for _ in range(K):
        M = rng.uniform(*DR_M)
        m = rng.uniform(*DR_m)
        l = rng.uniform(*DR_l)
        lat = int(rng.integers(0, DR_LAT_MAX + 1))
        out.append((M, m, l, lat, 0.0))
    return out


# --------------------------------------------------------------------------
# 평가: 고정 세계에서 성공률(임계시간 이상 균형 유지 비율)
# --------------------------------------------------------------------------
def success_rate(w, world, eps=EVAL_EPS, seed=EVAL_SEED):
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(eps):
        if rollout(w, world, rng)[1] >= MAX_STEPS * SUCCESS_FRAC:
            ok += 1
    return ok / eps


def evaluate_grid(w):
    """LAT_GRID × LEN_GRID 위 성공률 행렬 (행=지연, 열=길이)."""
    Z = np.zeros((len(LAT_GRID), len(LEN_GRID)))
    for i, lat in enumerate(LAT_GRID):
        for j, l in enumerate(LEN_GRID):
            Z[i, j] = success_rate(w, (1.0, 0.1, l, lat, 0.0))
    return Z


# --------------------------------------------------------------------------
def main():
    # 1) 두 정책 학습 (동일 시드 → 재현 가능)
    w_nom = cem_train(nominal_worlds)
    w_dr = cem_train(dr_worlds)

    # 2) 명목 세계 성능
    nom_at_nominal = success_rate(w_nom, NOMINAL)
    dr_at_nominal = success_rate(w_dr, NOMINAL)

    # 3) 이동된 '현실' 격자 전체 평가
    Z_nom = evaluate_grid(w_nom)
    Z_dr = evaluate_grid(w_dr)

    # 강건성 지표 = 이동된(지연>=1) 세계들에서의 평균 성공률 (외삽 구간 포함)
    shifted = np.array(LAT_GRID) >= 1
    nom_robust = float(Z_nom[shifted].mean())
    dr_robust = float(Z_dr[shifted].mean())

    print("=== Domain randomization: sim-to-real transfer (cart-pole) ===")
    print(f"정책: 선형 상태피드백 4-가중치, CEM(iters={CEM_ITERS}, pop={CEM_POP}, "
          f"elite={CEM_ELITE}, K={CEM_K})")
    print(f"학습 가중치  nominal-only w = {np.round(w_nom, 2)}")
    print(f"학습 가중치  DR           w = {np.round(w_dr, 2)}")
    print(f"  → theta 이득: nominal {w_nom[2]:.1f} (공격적) vs DR {w_dr[2]:.1f} (보수적/강건)")
    print(f"DR 학습 범위: 지연 0..{DR_LAT_MAX} step, 폴 길이 {DR_l}, 폴 질량 {DR_m}, 카트 질량 {DR_M}")
    print()
    print(f"명목(sim) 세계 성공률   nominal-only {nom_at_nominal:.2f}   |   DR {dr_at_nominal:.2f}")
    print(f"이동된 현실 격자 평균    nominal-only {nom_robust:.3f}   |   DR {dr_robust:.3f}")
    print(f"  → DR 강건성 우위 +{dr_robust - nom_robust:.3f} "
          f"(격자: 지연 {LAT_GRID[0]}..{LAT_GRID[-1]} step × 폴 길이 {LEN_GRID[0]}..{LEN_GRID[-1]} m)")
    # 정직: DR도 학습 범위를 크게 벗어난 극단 지연에서는 실패
    far = np.array(LAT_GRID) >= DR_LAT_MAX + 3
    if far.any():
        print(f"  (정직: 학습 범위 밖 지연 {DR_LAT_MAX+3}+ step 에서는 DR 성공률도 "
              f"{Z_dr[far].mean():.2f} 로 무너짐 — 외삽의 한계)")

    # ---------------- 플롯 ----------------
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.32)

    extent = [LEN_GRID[0] - 0.05, LEN_GRID[-1] + 0.05,
              LAT_GRID[-1] + 0.5, LAT_GRID[0] - 0.5]

    def heat(ax, Z, title):
        im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                       extent=extent, interpolation="nearest")
        ax.set_xlabel("pole half-length l [m]  (nominal 0.5)")
        ax.set_ylabel("control latency [steps]")
        ax.set_yticks(LAT_GRID)
        ax.set_title(title, fontsize=10)
        # DR 학습 지연 범위 경계
        ax.axhline(DR_LAT_MAX + 0.5, color="k", ls="--", lw=1.2, alpha=0.8)
        return im

    ax0 = fig.add_subplot(gs[0, 0])
    heat(ax0, Z_nom, f"nominal-only policy\n(sim {nom_at_nominal:.2f}, shifted-grid {nom_robust:.2f})")
    ax1 = fig.add_subplot(gs[0, 1])
    im = heat(ax1, Z_dr, f"domain-randomized policy\n(sim {dr_at_nominal:.2f}, shifted-grid {dr_robust:.2f})")
    ax1.text(LEN_GRID[0] - 0.02, DR_LAT_MAX + 0.5, " DR train range ",
             va="bottom", ha="left", fontsize=7.5, color="k",
             bbox=dict(fc="white", ec="none", alpha=0.7))
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("success rate", fontsize=8)

    # 세 번째 패널: 폴 길이 평균 성공률 vs 지연 (강건 구간 대비)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(LAT_GRID, Z_nom.mean(1), "o-", color="#d9534f", lw=2, label="nominal-only")
    ax2.plot(LAT_GRID, Z_dr.mean(1), "s-", color="#1f77b4", lw=2, label="domain-randomized")
    ax2.axvspan(-0.5, DR_LAT_MAX + 0.5, color="#1f77b4", alpha=0.08)
    ax2.text(DR_LAT_MAX / 2, 0.04, "DR training\nlatency range", ha="center",
             fontsize=7.5, color="#1f77b4")
    ax2.axvline(DR_LAT_MAX + 0.5, color="k", ls="--", lw=1.2, alpha=0.8)
    ax2.set_xlabel("control latency [steps]")
    ax2.set_ylabel("success rate (avg over pole length)")
    ax2.set_title("DR holds far wider before failing\n(both die deep outside DR range)",
                  fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(LAT_GRID)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle("Domain randomization for sim-to-real: nominal tuning overfits, "
                 "DR trades peak for robustness", fontsize=12, y=1.02)
    for p in ("outputs/30_domain_randomization.png", "assets/30_domain_randomization.png"):
        fig.savefig(p, dpi=130, bbox_inches="tight")
    print("\n[plot] outputs/30_domain_randomization.png, assets/30_domain_randomization.png")

    return nom_at_nominal, dr_at_nominal, nom_robust, dr_robust


if __name__ == "__main__":
    main()
