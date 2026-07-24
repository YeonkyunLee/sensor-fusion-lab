"""모델예측제어(MPC): 궤적 추종을 '최적화 문제'로 푼다.

exp19의 pure-pursuit는 전방주시점 하나만 보고 비례 조향하는 기하학적 추종기다.
빠르고 단순하지만, 곡률이 큰 구간에서 코너를 안쪽으로 자르고(코너 컷) 액추에이터
한계를 명시적으로 다루지 못한다. MPC는 매 스텝마다 운동모델로 N스텝 앞을 예측하고,
'추종오차 + 제어노력'을 최소화하는 제어열을 액추에이터 한계 제약 하에서 푼 뒤,
그 첫 제어만 적용하고 한 스텝 전진한다(receding horizon). 예측과 제약을 정면으로
다루므로 급커브에서 더 조밀하게 추종하고 |v|,|w| 한계를 항상 만족한다.

구현: 유니사이클(x,y,theta / 제어 v,w)을 기준궤적 주위에서 선형화해 시변(LTV)
모델을 얻고, 예측오차를 제어편차의 선형함수로 응축(condense)하면 비용은 제어편차에
대한 볼록 이차식(QP)이 된다. 이를 박스제약(액추에이터 한계) 하에서 L-BFGS-B로 푼다.
경쟁 상대는 exp19식 pure-pursuit. 동일한 8자(lemniscate) 기준궤적에서 교차추종오차
(cross-track RMSE)를 비교한다.

    python scripts/24_mpc_tracking.py
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DT = 0.1
N = 12                      # 예측 지평(스텝)
V_MAX, W_MAX = 2.6, 1.5     # 액추에이터 한계: 병진속도[m/s], 각속도[rad/s]
A_V, A_W = 4.0, 6.0         # 가속 한계: [m/s^2], [rad/s^2]


# --------------------------------------------------------------------------
# 기준 궤적: 베르누이 8자(lemniscate) — 두 로브 끝과 중앙 교차부에서 곡률이 커
# 코너 컷/오버슈트가 드러나는 고전적 난이도 궤적. 등호길이 재샘플 후 일정 속도.
# --------------------------------------------------------------------------
def make_reference(a=8.0, v_nom=2.0):
    phi = np.linspace(0, 2 * np.pi, 20000)
    den = 1 + np.sin(phi) ** 2
    x = a * np.cos(phi) / den
    y = a * np.sin(phi) * np.cos(phi) / den
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0], np.cumsum(d)])
    L = s[-1]
    nref = int((L / v_nom) / DT)
    snew = np.linspace(0, L, nref)
    xr = np.interp(snew, s, x)
    yr = np.interp(snew, s, y)
    th = np.unwrap(np.arctan2(np.gradient(yr), np.gradient(xr)))
    # 기준 제어(전진차분): 유니사이클이 대체로 따라갈 수 있는 실현가능한 기준
    v = np.hypot(np.diff(xr), np.diff(yr)) / DT
    w = np.diff(th) / DT
    v = np.append(v, v[-1])
    w = np.append(w, w[-1])
    Xref = np.stack([xr, yr, th], axis=1)          # (nref,3)
    Uref = np.stack([v, w], axis=1)                # (nref,2)
    return Xref, Uref


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def step_unicycle(X, u):
    x, y, th = X
    v, w = u
    return np.array([x + v * np.cos(th) * DT,
                     y + v * np.sin(th) * DT,
                     th + w * DT])


# --------------------------------------------------------------------------
# 선형화 MPC (응축형 QP)
# --------------------------------------------------------------------------
def _lin(Xr, Ur):
    """기준점 (Xr,Ur) 주위 유니사이클 야코비안 A,B."""
    th = Xr[2]
    v = Ur[0]
    A = np.array([[1, 0, -v * np.sin(th) * DT],
                  [0, 1,  v * np.cos(th) * DT],
                  [0, 0, 1.0]])
    B = np.array([[np.cos(th) * DT, 0],
                  [np.sin(th) * DT, 0],
                  [0, DT]])
    return A, B


def mpc_control(X, t, Xref, Uref, Q, Qf, R, warm, u_prev):
    n, m = 3, 2
    nref = len(Xref)
    idx = [min(t + k, nref - 1) for k in range(N)]      # 지평 내 기준 인덱스
    As, Bs = [], []
    for k in idx:
        A, B = _lin(Xref[k], Uref[k])
        As.append(A)
        Bs.append(B)

    # 응축: e_k = S_k e0 + sum_j G_{k,j} du_j,  e0 = wrap(X - Xref[t])
    e0 = X - Xref[t]
    e0[2] = wrap(e0[2])
    Aprod = [np.eye(n)]
    for k in range(N):
        Aprod.append(As[k] @ Aprod[-1])
    S = np.zeros((n * N, n))
    G = np.zeros((n * N, m * N))
    for k in range(1, N + 1):
        S[(k - 1) * n:k * n, :] = Aprod[k]
        for j in range(k):
            phi = np.eye(n)
            for i in range(k - 1, j, -1):
                phi = phi @ As[i]
            G[(k - 1) * n:k * n, j * m:(j + 1) * m] = phi @ Bs[j]

    Qbar = np.zeros((n * N, n * N))
    for k in range(1, N + 1):
        Qbar[(k - 1) * n:k * n, (k - 1) * n:k * n] = Qf if k == N else Q
    Rbar = np.kron(np.eye(N), R)

    GtQ = G.T @ Qbar
    H = 2 * (GtQ @ G + Rbar)
    f = 2 * (GtQ @ (S @ e0))

    def cost(u):
        return 0.5 * u @ H @ u + f @ u

    def grad(u):
        return H @ u + f

    # 박스제약: 절대 한계 |v|<=V_MAX,|w|<=W_MAX  →  du = u - Uref
    # 추가로 첫 스텝은 가속한계로 u_prev 대비 변화량 제한(실제 적용값 부드럽게)
    bounds = []
    for kk, k in enumerate(idx):
        lo_v, hi_v = -V_MAX - Uref[k, 0], V_MAX - Uref[k, 0]
        lo_w, hi_w = -W_MAX - Uref[k, 1], W_MAX - Uref[k, 1]
        if kk == 0:
            lo_v = max(lo_v, (u_prev[0] - A_V * DT) - Uref[k, 0])
            hi_v = min(hi_v, (u_prev[0] + A_V * DT) - Uref[k, 0])
            lo_w = max(lo_w, (u_prev[1] - A_W * DT) - Uref[k, 1])
            hi_w = min(hi_w, (u_prev[1] + A_W * DT) - Uref[k, 1])
        bounds += [(lo_v, hi_v), (lo_w, hi_w)]

    res = minimize(cost, warm, jac=grad, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 60})
    du = res.x
    u0 = Uref[idx[0]] + du[:2]
    u0 = np.array([np.clip(u0[0], -V_MAX, V_MAX), np.clip(u0[1], -W_MAX, W_MAX)])
    warm_next = np.concatenate([du[2:], du[-2:]])       # 시프트 warm start
    return u0, warm_next


def run_mpc(Xref, Uref):
    Q = np.diag([12.0, 12.0, 0.5])
    Qf = np.diag([30.0, 30.0, 1.0])
    R = np.diag([0.2, 0.05])
    X = Xref[0].copy()
    traj = [X[:2].copy()]
    ctrl = []
    warm = np.zeros(2 * N)
    u_prev = Uref[0].copy()
    for t in range(len(Xref)):
        u, warm = mpc_control(X, t, Xref, Uref, Q, Qf, R, warm, u_prev)
        X = step_unicycle(X, u)
        u_prev = u
        traj.append(X[:2].copy())
        ctrl.append(u)
    return np.array(traj), np.array(ctrl)


# --------------------------------------------------------------------------
# pure-pursuit 기준선 (exp19 방식). 폐곡선을 감싸도록 전방주시점을 순환 탐색.
# --------------------------------------------------------------------------
def run_pure_pursuit(Xref, v=2.0, Ld=1.6, kp=2.5):
    path = Xref[:, :2]
    npt = len(path)
    X = Xref[0].copy()
    traj = [X[:2].copy()]
    ctrl = []
    gi = 0
    for _ in range(npt):
        # 현재 위치에서 Ld 이상 떨어진 다음 경로점(순환)
        cnt = 0
        while np.hypot(*(path[gi % npt] - X[:2])) < Ld and cnt < npt:
            gi += 1
            cnt += 1
        tgt = path[gi % npt]
        ang = np.arctan2(tgt[1] - X[1], tgt[0] - X[0])
        derr = wrap(ang - X[2])
        w = np.clip(kp * derr, -W_MAX, W_MAX)
        vv = np.clip(v, -V_MAX, V_MAX)
        X = step_unicycle(X, np.array([vv, w]))
        traj.append(X[:2].copy())
        ctrl.append([vv, w])
    return np.array(traj), np.array(ctrl)


def cross_track_stats(traj, Xref):
    tree = cKDTree(Xref[:, :2])
    d, _ = tree.query(traj)
    return float(np.sqrt(np.mean(d ** 2))), float(d.max())


def main():
    Xref, Uref = make_reference()

    mpc_traj, mpc_u = run_mpc(Xref, Uref)
    pp_traj, pp_u = run_pure_pursuit(Xref)

    mpc_rmse, mpc_max = cross_track_stats(mpc_traj, Xref)
    pp_rmse, pp_max = cross_track_stats(pp_traj, Xref)

    mpc_vmax, mpc_wmax = np.abs(mpc_u[:, 0]).max(), np.abs(mpc_u[:, 1]).max()

    print("=== MPC vs pure-pursuit: 8자 궤적 추종 ===")
    print(f"기준궤적 점 {len(Xref)}개, 지평 N={N}, dt={DT}s, 한계 |v|<={V_MAX} |w|<={W_MAX}")
    print(f"교차추종 RMSE     MPC {mpc_rmse:.3f} m   |   pure-pursuit {pp_rmse:.3f} m")
    print(f"최대 오차(코너컷) MPC {mpc_max:.3f} m   |   pure-pursuit {pp_max:.3f} m")
    print(f"MPC 제어 최대     |v|={mpc_vmax:.3f}<= {V_MAX}, |w|={mpc_wmax:.3f}<= {W_MAX} "
          f"({'한계 준수' if mpc_vmax <= V_MAX + 1e-9 and mpc_wmax <= W_MAX + 1e-9 else '위반!'})")
    print(f"개선율  RMSE {100*(pp_rmse-mpc_rmse)/pp_rmse:.1f}%   "
          f"최대오차 {100*(pp_max-mpc_max)/pp_max:.1f}%")

    # ---------------- 플롯 ----------------
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(Xref[:, 0], Xref[:, 1], "k--", lw=1.3, alpha=0.6, label="reference (figure-8)")
    ax.plot(pp_traj[:, 0], pp_traj[:, 1], color="#d9534f", lw=1.8, alpha=0.9,
            label=f"pure-pursuit (RMSE {pp_rmse:.2f} m)")
    ax.plot(mpc_traj[:, 0], mpc_traj[:, 1], color="#1f77b4", lw=2.0,
            label=f"MPC (RMSE {mpc_rmse:.2f} m)")
    ax.plot(*Xref[0, :2], "ko", ms=8, label="start")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Trajectory tracking: MPC follows tight, pure-pursuit cuts the tight lobes")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2)

    tvec = np.arange(len(mpc_u)) * DT
    axv = fig.add_subplot(gs[0, 1])
    axv.plot(tvec, mpc_u[:, 0], color="#1f77b4", label="MPC v")
    axv.plot(tvec, pp_u[:, 0], color="#d9534f", lw=1, alpha=0.7, label="pursuit v")
    axv.axhline(V_MAX, color="gray", ls=":", lw=1)
    axv.axhline(-V_MAX, color="gray", ls=":", lw=1)
    axv.set_ylabel("v [m/s]")
    axv.set_title("Control inputs stay within actuator limits", fontsize=9)
    axv.legend(fontsize=7, loc="lower right")
    axv.grid(alpha=0.2)

    axw = fig.add_subplot(gs[1, 1])
    axw.plot(tvec, mpc_u[:, 1], color="#1f77b4", label="MPC w")
    axw.plot(tvec, pp_u[:, 1], color="#d9534f", lw=1, alpha=0.7, label="pursuit w")
    axw.axhline(W_MAX, color="gray", ls=":", lw=1)
    axw.axhline(-W_MAX, color="gray", ls=":", lw=1)
    axw.set_ylabel("w [rad/s]")
    axw.set_xlabel("time [s]")
    axw.legend(fontsize=7, loc="lower right")
    axw.grid(alpha=0.2)

    fig.tight_layout()
    for p in ("outputs/24_mpc_tracking.png", "assets/24_mpc_tracking.png"):
        fig.savefig(p, dpi=130)
    print("\n[plot] outputs/24_mpc_tracking.png, assets/24_mpc_tracking.png")
    return mpc_rmse, pp_rmse


if __name__ == "__main__":
    main()
