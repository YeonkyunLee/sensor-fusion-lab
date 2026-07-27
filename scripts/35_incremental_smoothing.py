"""증분 스무딩(iSAM 계열): 매 스텝 전체를 다시 풀지 않고 MAP 추정을 갱신.

온라인 SLAM에서 새 pose·측정이 하나 들어올 때마다 **전체 궤적을 처음부터 다시
최적화(full batch)**하는 것은 낭비다. 스텝당 비용이 O(N)으로 커지고, 대부분의 과거
pose는 이번 측정과 무관해 값이 거의 안 바뀐다. 증분 스무딩(iSAM, Kaess et al.)의 핵심
아이디어는 두 가지다:

1. **영향받은 변수만 갱신** — 새 팩터는 그래프의 국소 부분에만 영향을 준다. 오도메트리
   엣지는 방금 추가한 pose 주변만, 루프클로저 엣지는 두 끝점을 잇는 구간만 건드린다.
   그 부분만 다시 인수분해(re-factorize)/재선형화하면 되고 나머지는 그대로 둔다.
2. **요구 시 재선형화(relinearize-on-demand)** — 이미 잘 수렴한 과거 변수는 선형화점을
   유지하고, 오차가 큰(=이번에 크게 움직인) 변수만 다시 선형화한다.

그 결과 탐사(오도메트리) 중에는 스텝당 비용이 사실상 O(1), 루프클로저가 터질 때만
영향 구간에 비례하는 일시적 비용이 든다. 누적 계산량은 매 스텝 배치 재풀이 대비 훨씬
작으면서 정확도는 배치 최적해에 근접한다.

정직한 범위 명시:
- **진짜 iSAM**은 정보행렬의 제곱근 인수 R을 Bayes tree로 유지하고, 새 팩터가 오면
  Givens 회전으로 트리의 영향받은 clique만 국소 재인수분해한다(변수 재정렬 포함).
- 본 실험은 그 Bayes-tree 증분 인수분해를 **적응형 윈도우 Gauss-Newton**으로 근사한다:
  오도메트리 스텝에는 최근 몇 개 pose만, 루프클로저 스텝에는 루프가 잇는 구간 전체를
  변수로 두고 희소 GN(scipy.sparse)으로 재풀이·재선형화한다. "영향 변수만 재선형화"
  라는 iSAM의 본질은 그대로 구현하되, 제곱근 인수의 국소 갱신(Givens/QR) 대신 영향
  clique를 통째로 다시 푸는 방식으로 근사한 것이다. 트레이드오프: 근사 윈도우 밖의
  과거 변수는 이번 스텝에 고정되므로, 매우 먼 과거까지 파급되는 보정은 배치보다 약간
  덜 반영될 수 있다(실측상 무시할 수준).

비교(같은 문제, 같은 시드):
  (1) full batch    : 매 스텝 pose 1..k 전체를 처음부터 GN — 정확하나 스텝당 O(N).
  (2) incremental   : 적응형 윈도우 iSAM 근사 — 배치급 정확도에 경계 있는 스텝당 비용.
  (3) naive warm    : 방금 추가한 pose만 갱신(과거 재선형화 안 함) — 싸지만 루프클로저를
                      과거로 파급 못 해 드리프트.

    python scripts/35_incremental_smoothing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.posegraph import Edge, _error_and_jacobians, t2v, v2t, wrap  # noqa: E402


def rel(a, b):
    """pose a 기준 pose b의 상대 pose (a→b)."""
    return t2v(np.linalg.inv(v2t(a)) @ v2t(b))


def gn_step(X, edges, var_ids, iters=4, tol=1e-5):
    """var_ids 의 pose만 희소 Gauss-Newton으로 최적화(나머지는 고정 상수).

    매 반복마다 활성 변수에서 error·Jacobian을 다시 계산 = 활성 구간 재선형화.
    반환: (실제 수행한 반복 수). X 는 제자리 갱신.

    희소 정보행렬을 scipy.sparse 로 조립하고 spsolve 로 정규방정식을 푼다 — 활성 변수
    수 M 에만 비용이 좌우되므로, 윈도우가 작으면 스텝당 비용이 작게 유지된다.
    """
    vpos = {v: k for k, v in enumerate(var_ids)}
    M = 3 * len(var_ids)
    if M == 0:
        return 0
    done = 0
    for _ in range(iters):
        rows, cols, vals = [], [], []
        b = np.zeros(M)

        def add_block(r0, c0, Blk):
            for a in range(3):
                for c in range(3):
                    rows.append(r0 + a); cols.append(c0 + c); vals.append(Blk[a, c])

        for e in edges:
            iv = e.i in vpos
            jv = e.j in vpos
            if not (iv or jv):
                continue
            err, A, B = _error_and_jacobians(X[e.i], X[e.j], e.z)
            Om = e.omega
            if iv:
                a = 3 * vpos[e.i]
                add_block(a, a, A.T @ Om @ A)
                b[a:a + 3] += A.T @ Om @ err
            if jv:
                c = 3 * vpos[e.j]
                add_block(c, c, B.T @ Om @ B)
                b[c:c + 3] += B.T @ Om @ err
            if iv and jv:
                a, c = 3 * vpos[e.i], 3 * vpos[e.j]
                add_block(a, c, A.T @ Om @ B)
                add_block(c, a, B.T @ Om @ A)
        # 게이지/조건화 정칙항
        for d in range(M):
            rows.append(d); cols.append(d); vals.append(1e-6)
        H = sp.csr_matrix((vals, (rows, cols)), shape=(M, M))
        dx = spla.spsolve(H, -b)
        for v, k in vpos.items():
            X[v] += dx[3 * k:3 * k + 3]
            X[v, 2] = wrap(X[v, 2])
        done += 1
        if np.max(np.abs(dx)) < tol:
            break
    return done


def main():
    rng = np.random.default_rng(0)

    # --- 2바퀴 궤적(참값): 2번째 바퀴가 1번째 바퀴 위치를 재방문 → 루프클로저 ---
    dt = 0.1
    v, R = 6.0, 20.0
    w = v / R
    per_lap = 55
    laps = 3
    n = per_lap * laps

    true = [np.array([0.0, 0.0, 0.0])]
    for _ in range(n):
        p = true[-1]
        true.append(np.array([p[0] + v * dt * np.cos(p[2]),
                              p[1] + v * dt * np.sin(p[2]),
                              wrap(p[2] + w * dt)]))
    true = np.array(true)
    npose = len(true)

    # --- 오도메트리 엣지(잡음 상대이동) + 루프클로저 엣지(2바퀴째 재방문) ---
    odo_sig = np.array([0.05, 0.05, 0.02])
    Om = np.diag(1.0 / odo_sig**2)
    lc_sig = np.array([0.05, 0.05, 0.02])
    Om_lc = np.diag(1.0 / lc_sig**2)

    odo = {k: (k - 1, k, rel(true[k - 1], true[k]) + rng.normal(0, odo_sig))
           for k in range(1, npose)}
    loops = {}
    for j in range(per_lap, npose):
        i = j - per_lap                        # 정확히 한 바퀴 전 pose
        if j % 5 == 0:                          # 가끔 루프클로저
            loops.setdefault(j, []).append((i, j, rel(true[i], true[j]) + rng.normal(0, lc_sig)))
    n_loops = sum(len(x) for x in loops.values())

    ODO_WIN = 6      # 오도메트리 스텝: 최근 pose만 갱신(국소 스무딩)
    NAIVE_ITERS = 4

    def run(strategy):
        """strategy in {'batch','incremental','naive'}. 반환 (X, cost_curve)."""
        X = np.zeros((npose, 3))
        X[0] = true[0]
        edges = []
        cum = 0.0
        cost_curve = np.zeros(npose)      # 스텝별 누적 계산량(=풀이한 변수 수 합)
        for k in range(1, npose):
            i, j, z = odo[k]
            X[k] = t2v(v2t(X[k - 1]) @ v2t(z))     # 새 pose 초기화(warm start)
            edges.append(Edge(i, j, z, Om))
            fired = loops.get(k, [])
            for (li, lj, lz) in fired:
                edges.append(Edge(li, lj, lz, Om_lc))

            if strategy == "batch":
                # 매 스텝 전체를 처음부터 재풀이(재선형화 전체) — O(N)/스텝
                var_ids = list(range(1, k + 1))
                its = gn_step(X, edges, var_ids, iters=4)
            elif strategy == "incremental":
                # iSAM 근사: 오도메트리는 최근 윈도우만, 루프클로저는 영향 구간 전체를
                # 재선형화(요구 시 재선형화 + 영향 변수만 갱신)
                if fired:
                    lo = min(li for (li, _, _) in fired)
                    var_ids = list(range(max(1, lo), k + 1))
                    its = gn_step(X, edges, var_ids, iters=6)   # 루프: 조금 더 반복
                else:
                    var_ids = list(range(max(1, k - ODO_WIN + 1), k + 1))
                    its = gn_step(X, edges, var_ids, iters=2)   # 탐사: 소수 반복
            elif strategy == "naive":
                # warm-start만: 방금 추가한 pose 하나만 갱신 → 과거로 파급 못 함
                var_ids = [k]
                its = gn_step(X, edges, var_ids, iters=NAIVE_ITERS)
            else:
                raise ValueError(strategy)

            cum += 3.0 * len(var_ids) * max(its, 1)   # 계산량 프록시: 변수수×반복
            cost_curve[k] = cum
        return X, cost_curve

    Xb, cb = run("batch")
    Xi, ci = run("incremental")
    Xn, cn = run("naive")

    def rmse(X):
        return float(np.sqrt(np.mean(np.sum((X[:, :2] - true[:, :2])**2, axis=1))))

    rb, ri, rn = rmse(Xb), rmse(Xi), rmse(Xn)
    Cb, Ci, Cn = float(cb[-1]), float(ci[-1]), float(cn[-1])

    print("=== 증분 스무딩 (iSAM 계열): 매 스텝 전체 재풀이 없이 MAP 갱신 ===")
    print(f"pose {npose}개, 오도메트리 {npose-1}, 루프클로저 {n_loops}")
    print(f"(1) full batch  : RMSE {rb:.3f} m,  누적계산량 {Cb:,.0f}  (스텝당 O(N))")
    print(f"(2) incremental : RMSE {ri:.3f} m,  누적계산량 {Ci:,.0f}  (경계 있는 스텝당 비용)")
    print(f"(3) naive warm  : RMSE {rn:.3f} m,  누적계산량 {Cn:,.0f}  (과거 재선형화 안 함)")
    print(f"→ incremental 는 batch 대비 계산량 {Cb/max(Ci,1e-9):.1f}x 절감, "
          f"정확도 격차 {abs(ri-rb):.3f} m ({ri/max(rb,1e-9):.2f}x)")
    print(f"→ naive 는 싸지만({Cb/max(Cn,1e-9):.0f}x) 루프클로저를 과거로 못 퍼뜨려 "
          f"RMSE {rn/max(rb,1e-9):.1f}x 열화")

    # --- 그림: (좌) 궤적 오버레이, (우) 누적 계산량 곡선 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax1.plot(true[:, 0], true[:, 1], "g-", lw=2.6, label=f"true ({laps} laps)")
    ax1.plot(Xb[:, 0], Xb[:, 1], "r--", lw=1.6, alpha=0.8, label=f"full batch ({rb:.2f} m)")
    ax1.plot(Xi[:, 0], Xi[:, 1], "b-", lw=1.3, label=f"incremental iSAM ({ri:.2f} m)")
    ax1.plot(Xn[:, 0], Xn[:, 1], "0.55", lw=1.0, ls=":", label=f"naive warm-start ({rn:.2f} m)")
    ax1.plot(true[0, 0], true[0, 1], "ko", ms=8, label="start")
    ax1.set_aspect("equal")
    ax1.set_title("Trajectories: incremental overlays batch")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    steps = np.arange(npose)
    ax2.plot(steps, cb, "r-", lw=1.8, label=f"full batch (total {Cb:,.0f})")
    ax2.plot(steps, ci, "b-", lw=1.8, label=f"incremental (total {Ci:,.0f})")
    ax2.plot(steps, cn, "0.55", lw=1.4, ls=":", label=f"naive (total {Cn:,.0f})")
    for j in loops:
        ax2.axvline(j, color="c", alpha=0.18, lw=1.0)
    ax2.set_title("Cumulative compute: batch grows O(N²), incremental stays cheap")
    ax2.set_xlabel("poses processed")
    ax2.set_ylabel("cumulative variables solved (proxy)")
    ax2.legend(fontsize=8, loc="upper left"); ax2.grid(alpha=0.3)

    fig.suptitle("Incremental smoothing (iSAM-style): near-batch accuracy at a fraction of the cumulative compute")
    fig.tight_layout()
    for out in ("outputs", "assets"):
        Path(out).mkdir(exist_ok=True)
        fig.savefig(Path(out) / "35_incremental_smoothing.png", dpi=130)
    print("\n[plot] outputs/35_incremental_smoothing.png, assets/35_incremental_smoothing.png")

    return rb, ri, Cb, Ci


if __name__ == "__main__":
    main()
