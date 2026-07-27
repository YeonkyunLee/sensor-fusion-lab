"""합성 데이터와 자동 라벨링(synthetic data & automatic labeling): sim-to-real 인식이 공짜 라벨로 굴러가는 이유.

로봇 인식 모델(물체 위치·자세 추정)을 학습하려면 대개 '이미지 + 정답 라벨' 쌍이 필요하다.
그런데 현실 이미지에 사람이 손으로 위치를 찍어 라벨을 다는 일은 느리고 비싸며 양도 적다.
시뮬레이터는 이 병목을 정면으로 없앤다: 물체를 어디에 놓을지 '내가 정해서' 렌더링하므로
정답 위치(cx, cy)를 이미 알고 있다 — 렌더링과 동시에 완벽한 라벨이 공짜로, 무제한으로
쏟아진다. 이것이 자동 라벨링(automatic labeling)의 핵심이다. 라벨 노이즈도 없고, 손으로
찍을 때 생기는 픽셀 오차도 없다.

문제는 '보기(look)'다. 시뮬레이터를 명목(nominal) 설정 하나로만 렌더링하면 배경·대비·크기·
잡음이 늘 똑같은 '너무 깨끗한' 이미지만 나온다. 그 위에서 학습한 모델은 그 특정 외형에
과적합해, 조명·대비·배경 기울기가 다른 '현실' 이미지에서 위치를 크게 틀린다(sim-to-real
gap). 처방은 30번 실험과 같은 도메인 랜덤화(domain randomization, DR)다: 합성 이미지를
렌더링할 때 배경 밝기/기울기, 물체 대비/크기, 뷰포인트 전단(shear), 잡음을 매번 넓게
무작위로 흔든다. 그러면 정답 라벨(cx, cy)은 그대로 공짜인 채, 모델은 '한 가지 외형'이 아니라
'외형의 가족(family of looks)' 전체에서 위치를 읽도록 강제된다. 라벨이 아니라 외형을
랜덤화하는 것 — 이것이 공짜 합성 라벨을 현실로 '전이(transfer)'시키는 장치다.

이 실험은 그 논리를 숫자로 보인다. 과제는 24x24 그레이스케일 이미지 속 가우시안 블롭의
2D 위치 추정이다. 시뮬레이터가 블롭을 (cx, cy)에 그려 넣으므로 라벨은 정의상 정확하다.
동일한 회귀기(순수 numpy 릿지 회귀, 닫힌 해)를 세 가지 학습셋에 각각 학습시킨다:
  (1) 희소 현실(scarce real): 이동된 '현실' 분포에서 손으로 라벨링했다고 가정한 소수(40장).
  (2) 합성-DR없음(synthetic, no-DR): 명목 외형만으로 렌더링한 다수(800장) — 깨끗하지만 단조.
  (3) 합성+DR(synthetic + domain randomization): 외형을 넓게 랜덤화한 다수(800장).
평가는 셋 다 동일한 '현실' 테스트셋(이동된 분포, 일부는 DR 학습 범위 바깥까지)에서 하고,
지표는 픽셀 단위 위치 RMSE다.

정직한 결과: 합성-DR없음은 명목 외형에 과적합해 현실에서 크게 틀린다. 희소 현실은 분포는
맞지만 표본이 적어(고차원 픽셀 + 40장) 불안정하다. 합성+DR은 무제한 공짜 라벨 + 넓은 외형
커버리지 덕에 현실 테스트에서 가장 낮은 RMSE를 낸다. 다만 정직하게 두 가지 한계를 남긴다:
(a) 소량의 현실 데이터로 합성+DR 위에 미세조정(fine-tune)하면 보통 그보다도 낫다 — 합성이
현실을 대체하는 게 아니라 현실 라벨 예산을 아끼는 것이다. (b) DR도 학습 범위를 크게 벗어난
극단 외형(테스트셋 꼬리)에서는 오차가 커진다 — 랜덤화 범위 밖 외삽은 여전히 위험하다.

    python scripts/34_synthetic_labeling.py
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 과제 설정: 24x24 이미지 속 가우시안 블롭의 2D 위치 (cx, cy) 추정
# --------------------------------------------------------------------------
N = 24                              # 이미지 한 변 [px]
POS_LO, POS_HI = 6.0, 18.0          # 블롭 중심이 놓이는 범위 (테두리 여유 확보)

# 렌더링에 쓰는 좌표 격자 (한 번만 생성)
_YY, _XX = np.mgrid[0:N, 0:N].astype(float)   # _XX[y,x]=x, _YY[y,x]=y

# 명목(nominal, sim) 외형: 배경 밝기/기울기·대비·크기·전단·잡음이 모두 고정된 '깨끗한' 설정
NOM = dict(a=1.0, sigma=2.6, b0=0.5, gx=0.0, gy=0.0, shear=0.0, noise=0.0)

# 도메인 랜덤화(DR) 학습 범위 — 외형을 넓게 흔든다 (라벨 cx,cy 는 건드리지 않음)
DR_RANGES = dict(
    a=(0.45, 1.10),                 # 물체 대비
    sigma=(2.0, 3.4),               # 물체 크기 [px]
    b0=(0.15, 0.85),                # 배경 밝기(DC)
    g=(-0.50, 0.50),                # 배경 기울기(gradient) gx, gy
    shear=(-0.30, 0.30),            # 뷰포인트 전단(블롭 타원화)
    noise=(0.00, 0.06),             # 가산 잡음 std
)

# '현실'(shifted real) 분포 — 명목에서 이동, 일부는 DR 범위 바깥까지(정직한 외삽 노출)
REAL_RANGES = dict(
    a=(0.35, 0.85),                 # 명목(1.0)보다 낮은 대비
    sigma=(2.0, 3.6),
    b0=(0.10, 0.90),
    g=(-0.60, 0.60),                # DR(±0.5)보다 강한 기울기까지
    shear=(-0.35, 0.35),
    noise=(0.02, 0.08),
)

N_SYNTH = 800                       # 합성(무제한 공짜 라벨) 학습 이미지 수
N_SCARCE = 40                       # 희소 현실(비싼 손라벨) 학습 이미지 수
N_TEST = 400                        # 현실 테스트 이미지 수
RIDGE_LAMBDA = 5.0                  # 릿지 정규화 계수

SEED_SYNTH = 0
SEED_SCARCE = 7
SEED_TEST = 1234


# --------------------------------------------------------------------------
# 렌더러 = 시뮬레이터. 블롭 중심을 '내가 정하므로' (cx,cy)가 곧 완벽한 자동 라벨.
#   전단(shear)/크기는 블롭을 타원화할 뿐 중심(=라벨)은 (cx,cy) 그대로 유지된다.
# --------------------------------------------------------------------------
def render(cx, cy, a, sigma, b0, gx, gy, shear, noise, rng):
    dx = _XX - cx
    dy = _YY - cy
    # 전단으로 타원/기울어진 가우시안 (뷰포인트 변화 아날로그) — 중심은 불변
    sx = sigma * (1.0 + shear)
    sy = sigma * (1.0 - shear)
    q = (dx / sx) ** 2 + (dy / sy) ** 2 + 0.6 * shear * (dx * dy) / (sx * sy)
    blob = a * np.exp(-0.5 * q)
    # 배경: DC 밝기 + 선형 기울기
    bg = b0 + gx * (_XX - N / 2.0) / N + gy * (_YY - N / 2.0) / N
    img = bg + blob
    if noise > 0:
        img = img + rng.normal(0.0, noise, size=img.shape)
    return img


def _sample_nuisance(ranges, rng):
    return dict(
        a=rng.uniform(*ranges["a"]),
        sigma=rng.uniform(*ranges["sigma"]),
        b0=rng.uniform(*ranges["b0"]),
        gx=rng.uniform(*ranges["g"]),
        gy=rng.uniform(*ranges["g"]),
        shear=rng.uniform(*ranges["shear"]),
        noise=rng.uniform(*ranges["noise"]),
    )


def make_dataset(n, mode, seed):
    """(X 픽셀행렬 [n, N*N], Y 라벨 [n, 2]) 생성.

    mode: "nominal"(고정 명목 외형), "dr"(DR 범위 랜덤화), "real"(현실 이동분포 랜덤화).
    어느 모드든 라벨 (cx,cy)는 렌더러가 '정해서' 그린 값 → 정의상 정확(자동 라벨링).
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, N * N))
    Y = np.empty((n, 2))
    for i in range(n):
        cx = rng.uniform(POS_LO, POS_HI)
        cy = rng.uniform(POS_LO, POS_HI)
        if mode == "nominal":
            nz = dict(NOM)
        elif mode == "dr":
            nz = _sample_nuisance(DR_RANGES, rng)
        elif mode == "real":
            nz = _sample_nuisance(REAL_RANGES, rng)
        else:
            raise ValueError(mode)
        img = render(cx, cy, rng=rng, **nz)
        X[i] = img.ravel()
        Y[i] = (cx, cy)
    return X, Y


# --------------------------------------------------------------------------
# 회귀기: 순수 numpy 릿지 회귀 (닫힌 해 → 완전 결정론적·안정)
#   입력 = 평탄화 픽셀 + 바이어스, 출력 = (cx, cy)
# --------------------------------------------------------------------------
def ridge_fit(X, Y, lam=RIDGE_LAMBDA):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])       # 바이어스 열
    d = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(d)
    A[-1, -1] -= lam                                    # 바이어스 항은 정규화 제외
    W = np.linalg.solve(A, Xb.T @ Y)
    return W


def ridge_predict(W, X):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ W


def rmse_px(W, X, Y):
    P = ridge_predict(W, X)
    err2 = np.sum((P - Y) ** 2, axis=1)                 # 샘플별 유클리드 오차^2 [px^2]
    return float(np.sqrt(np.mean(err2)))


# --------------------------------------------------------------------------
def main():
    # 1) 세 학습셋 구성 -----------------------------------------------------
    #    합성 두 셋은 라벨이 공짜·무제한(N_SYNTH), 현실 셋은 손라벨이라 희소(N_SCARCE).
    X_scarce, Y_scarce = make_dataset(N_SCARCE, "real", SEED_SCARCE)
    X_noDR, Y_noDR = make_dataset(N_SYNTH, "nominal", SEED_SYNTH)
    X_DR, Y_DR = make_dataset(N_SYNTH, "dr", SEED_SYNTH + 1)

    # 2) 동일 회귀기를 각 셋에 학습 ----------------------------------------
    W_scarce = ridge_fit(X_scarce, Y_scarce)
    W_noDR = ridge_fit(X_noDR, Y_noDR)
    W_DR = ridge_fit(X_DR, Y_DR)

    # 3) 공통 '현실' 테스트셋에서 평가 (일부 표본은 DR 범위 바깥) -----------
    X_test, Y_test = make_dataset(N_TEST, "real", SEED_TEST)
    scarce_rmse = rmse_px(W_scarce, X_test, Y_test)
    noDR_rmse = rmse_px(W_noDR, X_test, Y_test)
    DR_rmse = rmse_px(W_DR, X_test, Y_test)

    # 참고: 합성+DR 위에 희소 현실로 미세조정하면 보통 더 낫다 (합성이 라벨 예산을 아낌)
    X_ft = np.vstack([X_DR, np.repeat(X_scarce, 8, axis=0)])   # 현실 표본 가중 결합
    Y_ft = np.vstack([Y_DR, np.repeat(Y_scarce, 8, axis=0)])
    W_ft = ridge_fit(X_ft, Y_ft)
    ft_rmse = rmse_px(W_ft, X_test, Y_test)

    print("=== Synthetic data & auto-labeling for sim-to-real perception ===")
    print(f"과제: {N}x{N} 이미지 속 가우시안 블롭 2D 위치 추정 (라벨=시뮬레이터가 정한 cx,cy, 공짜)")
    print(f"회귀기: 순수 numpy 릿지 회귀(닫힌 해, lambda={RIDGE_LAMBDA}), 입력=평탄화 {N*N}픽셀 → (cx,cy)")
    print(f"학습셋 크기: scarce-real {N_SCARCE}장(손라벨) | synthetic {N_SYNTH}장(자동라벨, 공짜·무제한)")
    print()
    print("현실 테스트셋 위치 RMSE [px] (낮을수록 좋음):")
    print(f"  (1) scarce real   (현실분포, 40장 손라벨)     : {scarce_rmse:6.3f}")
    print(f"  (2) synthetic no-DR (명목 외형만, 800장)      : {noDR_rmse:6.3f}   <- 깨끗한 외형에 과적합")
    print(f"  (3) synthetic + DR (외형 랜덤화, 800장)       : {DR_rmse:6.3f}   <- 공짜 라벨이 현실로 전이")
    print(f"  → DR이 no-DR 대비 {noDR_rmse - DR_rmse:+.3f}px, scarce-real 대비 {scarce_rmse - DR_rmse:+.3f}px 개선")
    print()
    print(f"  (참고) synthetic+DR 에 현실 40장 미세조정      : {ft_rmse:6.3f}   "
          f"({'DR보다 더 낫다' if ft_rmse < DR_rmse else 'DR과 비슷'} — 현실은 대체가 아니라 절약)")
    print(f"  (정직) 테스트셋 일부는 DR 학습 범위(대비/기울기) 밖 외형 — 그 구간에선 DR 오차도 상승(외삽 한계)")

    # ---------------- 플롯 ----------------
    _make_figure(W_scarce, W_noDR, W_DR,
                 scarce_rmse, noDR_rmse, DR_rmse, ft_rmse)

    return scarce_rmse, noDR_rmse, DR_rmse


# --------------------------------------------------------------------------
def _make_figure(W_scarce, W_noDR, W_DR,
                 scarce_rmse, noDR_rmse, DR_rmse, ft_rmse):
    rng = np.random.default_rng(2024)

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(
        3, 6, height_ratios=[1.0, 1.0, 1.15],
        hspace=0.55, wspace=0.30,
        left=0.05, right=0.97, top=0.90, bottom=0.07)

    # --- Row 1: 합성 학습 이미지 예시 (명목 vs DR 랜덤화), 자동 라벨(빨간 x) 표시 ---
    for k in range(3):
        cx = rng.uniform(POS_LO, POS_HI); cy = rng.uniform(POS_LO, POS_HI)
        img = render(cx, cy, rng=rng, **NOM)
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1.4, origin="upper")
        ax.plot(cx, cy, "x", color="red", ms=9, mew=2)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("synthetic\nno-DR", fontsize=9)
        ax.set_title(f"auto-label\n({cx:.0f},{cy:.0f})", fontsize=7.5)
    for k in range(3):
        cx = rng.uniform(POS_LO, POS_HI); cy = rng.uniform(POS_LO, POS_HI)
        nz = _sample_nuisance(DR_RANGES, rng)
        img = render(cx, cy, rng=rng, **nz)
        ax = fig.add_subplot(gs[0, 3 + k])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1.4, origin="upper")
        ax.plot(cx, cy, "x", color="red", ms=9, mew=2)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("synthetic\n+ DR", fontsize=9)
        ax.set_title(f"auto-label\n({cx:.0f},{cy:.0f})", fontsize=7.5)

    # --- Row 2: '현실' 테스트 이미지 6장, 참값 + 세 모델 예측 ---
    trng = np.random.default_rng(55)
    for k in range(6):
        cx = trng.uniform(POS_LO, POS_HI); cy = trng.uniform(POS_LO, POS_HI)
        nz = _sample_nuisance(REAL_RANGES, trng)
        img = render(cx, cy, rng=trng, **nz)
        x = img.ravel()[None, :]
        p_s = ridge_predict(W_scarce, x)[0]
        p_n = ridge_predict(W_noDR, x)[0]
        p_d = ridge_predict(W_DR, x)[0]
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1.4, origin="upper")
        ax.plot(cx, cy, "o", mfc="none", mec="#2ca02c", ms=13, mew=2.2)
        ax.plot(p_s[0], p_s[1], "s", color="#7f7f7f", ms=6)
        ax.plot(p_n[0], p_n[1], "^", color="#d62728", ms=6)
        ax.plot(p_d[0], p_d[1], "*", color="#1f77b4", ms=9)
        ax.set_xlim(0, N - 1); ax.set_ylim(N - 1, 0)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("'real'\ntest", fontsize=9)
    fig.text(0.5, 0.625,
             "'real' (shifted) test images — ◯ true (green)   ■ scarce-real   ▲ synth no-DR   ★ synth+DR",
             ha="center", fontsize=9)

    # --- Row 3: 막대그래프(테스트 RMSE) + 요약 텍스트 ---
    axb = fig.add_subplot(gs[2, 0:3])
    names = ["scarce\nreal (40)", "synthetic\nno-DR (800)", "synthetic\n+DR (800)"]
    vals = [scarce_rmse, noDR_rmse, DR_rmse]
    colors = ["#7f7f7f", "#d62728", "#1f77b4"]
    bars = axb.bar(names, vals, color=colors)
    axb.axhline(ft_rmse, color="#2ca02c", ls="--", lw=1.6)
    axb.text(-0.42, ft_rmse + 0.08, f"synth+DR & real fine-tune = {ft_rmse:.2f} px",
             color="#2ca02c", va="bottom", ha="left", fontsize=8)
    for b, v in zip(bars, vals):
        axb.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axb.set_ylabel("localization RMSE on 'real' test [px]")
    axb.set_title("Domain-randomized synthetic labels transfer best", fontsize=11)
    axb.set_ylim(0, max(vals) * 1.25)
    axb.grid(axis="y", alpha=0.25)

    axt = fig.add_subplot(gs[2, 3:6]); axt.axis("off")
    lines = [
        "Synthetic data + automatic labeling",
        "",
        "- The simulator PLACES the blob at (cx,cy),",
        "  so the label is free, unlimited, and exact",
        "  (no hand-labeling needed).",
        "",
        "- no-DR : overfits the clean nominal look",
        "          -> large error on 'real'.",
        "- scarce-real : right distribution but only",
        "          40 images -> high-variance / unstable.",
        "- +DR  : randomize the LOOK, keep free labels",
        "          -> synthetic labels transfer to 'real'.",
        "",
        "Honest limits:",
        "  * a little real fine-tuning on top is usually",
        "    best (synthetic saves the label budget,",
        "    it does not replace real data).",
        "  * far outside the DR range, DR fails too",
        "    (extrapolation is still risky).",
    ]
    axt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9.5,
             family="DejaVu Sans", linespacing=1.3)

    fig.suptitle(
        "Synthetic data gives free perfect labels — domain randomization is what makes them transfer to 'real'",
        fontsize=13, y=0.965)

    for p in ("outputs/34_synthetic_labeling.png", "assets/34_synthetic_labeling.png"):
        fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\n[plot] outputs/34_synthetic_labeling.png, assets/34_synthetic_labeling.png")


if __name__ == "__main__":
    main()
