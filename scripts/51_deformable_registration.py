"""변형 정합: 강체 가정이 무너질 때, 깊은 곳을 메우는 것은 데이터가 아니라 모델이다.

exp 41·44·49 의 정합은 전부 **강체**였다. 그런데 조직은 눌리고 밀린다 — 개두 후 뇌는 중력과
뇌척수액 유출로 내려앉고(brain shift, 문헌상 수 mm~2 cm), 견인기가 국소를 밀어낸다. 그러면
수술 전 영상과 수술 중 환자 사이에 **강체변환으로 표현할 수 없는 차이**가 남는다.
exp 49 가 "강체 가정이 남긴 가장 큰 구멍"이라고 적어둔 지점이 여기다.

exp 49 의 **실제 인체 MR 표면**에 물리적으로 그럴듯한 변형장을 주고(정답을 알아야 심부 오차를
잴 수 있으므로 해석적 장을 쓴다) 네 가지를 같은 지표로 비교한다.

  (a) **강체 ICP**            — exp 49 파이프라인. 변형을 평균 이동으로 흡수하려다 편향된다.
  (b) **자유형 워프(TPS)**    — 표면 대응점만으로 얇은 판 스플라인. 데이터에만 충실하다.
  (c) **TPS + 두개골 사전지식** — "창 밖 두피는 두개골이 잡고 있으니 변위 0" 앵커를 추가.
  (d) **물리 정규화(조화 확장)** — 같은 사전지식을 경계조건으로 두고 부피 내부에서 ∇²u = 0.

--- 이 실험이 답하려는 질문 ---
표면은 볼 수 있지만 **표적은 깊은 곳에 있다.** 표면만으로 심부를 맞추는 것은 본질적으로
**외삽**이고, 외삽의 품질은 데이터가 아니라 **모델(=사전지식)** 이 정한다. 그러니 물어야 할 것은
"표면 잔차가 얼마나 작은가"가 아니라 **"깊은 곳에서 얼마나 틀리는가"** 다.
exp 41·49 의 FRE≠TRE 함정이 변형 정합에서 훨씬 사납게 돌아온다.

임상 조건 하나 더: **노출된 표면은 일부뿐이다**(개두창). 창 밖에는 데이터가 없다. (b)와 (c)를
나눈 이유가 그것이다 — 보간기를 바꾼 것이 아니라 **사전지식을 넣은 것**의 값을 재려는 것이다.

--- 미리 말해두는 결론 ---
1. 강체는 표적에서 5.7 mm 를 남긴다. 변형 복원이 그걸 0.6 mm 로 줄인다.
2. 좁은 노출(표면의 4%)에서 자유형 워프는 3.31 mm, 같은 보간기에 두개골 사전지식만 넣으면
   0.60 mm. **이긴 것은 보간기가 아니라 가정**이다. 넓은 노출에서는 그 이득이 사라진다.
3. 조화 확장은 **격자를 촘촘히 할수록 나빠졌다**. 이산화 오차가 아니라 모델 편향이다 —
   틀린 방정식을 더 정확히 풀었을 뿐이고, 성긴 격자가 나았던 건 우연한 오차 상쇄였다.
4. 자유도(λ, 제어점 수)의 최적점이 노출 면적에 따라 움직인다. 고정하면 한쪽에서 손해다.

    python scripts/51_deformable_registration.py
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sensor_fusion.se3 import so3_exp  # noqa: E402

reg = import_module("41_surgical_registration")       # 점-대-평면 ICP
real = import_module("44_registration_real_scans")    # 2단계 ICP
anat = import_module("49_registration_real_anatomy")  # 실 인체 MR 표면

# --- 변형 규모: 문헌의 brain shift 자릿수 ---
SAG_MM = 12.0            # 개두창 부위 중력 침하 최대치 [mm]
BULGE_MM = 5.0           # 견인기 국소 변형 [mm]
DECAY_MM = 45.0          # 창에서 멀어질수록 줄어드는 감쇠 길이 [mm]
RETRACT_MM = 22.0        # 견인 영향 반경 [mm]

PROBE_NOISE = 1.0e-3     # 수술 중 디지타이징 잡음 σ [m]
N_PROBE = 900
TARGET_DEPTHS_MM = (20, 35, 50, 70)   # 개두창 아래 표적 깊이 — 병변은 노출면 아래에 있다
GRID_N = 34              # 조화 확장 격자 해상도(한 축)
N_TPS_CTRL = 170         # TPS 제어점(관측) — 고정 예산. 스윕에서 이 선택의 대가를 잰다
N_TPS_ANCHOR = 130       # TPS 앵커(창 밖 = 변위 0 사전지식)
TPS_LAMBDA = 1e-4
LANDMARK_NOISE = 2.5e-3
N_CTRL_SWEEP = (85, 170, 340, 600)
GRID_SWEEP = (34, 44, 54)


# --------------------------------------------------------------------------- #
# 1) 변형장 — 정답을 아는 합성 변형 (개두창 최대, 거리로 감쇠)
# --------------------------------------------------------------------------- #
def deformation(pts, window_c, sag_dir, inward):
    """수술 중 변위 u(x). 두 성분:

    - **중력 침하**: 개두창 중심에서의 거리로 지수 감쇠하는 sag_dir 방향 변위.
      창에서 최대이고 반대편 두피·심부로 갈수록 사라진다 — 두개골이 나머지를 잡고 있으므로.
    - **국소 견인**: 창 주변을 안으로 미는 가우시안 범프.

    실제 brain shift 는 뇌척수액 유출·자세·견인·절제가 얽혀 환자마다 다르다. 여기서는
    **정답이 필요해서** 해석적 장을 쓴다 — 방법 비교용이지 예측 모델이 아니다."""
    d = np.linalg.norm(pts - window_c, axis=1)
    u = (SAG_MM * 1e-3) * np.exp(-d / (DECAY_MM * 1e-3))[:, None] * sag_dir[None, :]
    u += (BULGE_MM * 1e-3) * np.exp(-(d / (RETRACT_MM * 1e-3)) ** 2)[:, None] * inward[None, :]
    return u


# --------------------------------------------------------------------------- #
# 2) 자유형 워프 — 3D 얇은 판 스플라인
# --------------------------------------------------------------------------- #
def tps_fit(ctrl, disp, lam=TPS_LAMBDA):
    """제어점 ctrl 에서 변위 disp 를 보간하는 3D TPS.

    커널은 **φ(r) = −r** 이다. 3D(d=3, m=2)의 Duchon 커널이 조건부 양정치가 되는 부호가 이쪽이고,
    부호를 뒤집으면 +λI 정규화가 고윳값을 반대로 밀어 특정 λ에서 시스템이 거의 특이해진다
    (실제로 처음엔 +r 로 짜서 λ 스윕이 비단조로 폭주했다 — 부호가 정규화의 방향을 정한다).

    lam=0 이면 제어점을 정확히 통과(프로브 잡음까지 학습), 크면 아핀에 가까워진다."""
    n = len(ctrl)
    K = -np.linalg.norm(ctrl[:, None, :] - ctrl[None, :, :], axis=2) + lam * np.eye(n)
    P = np.hstack([np.ones((n, 1)), ctrl])
    A = np.zeros((n + 4, n + 4))
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    rhs = np.zeros((n + 4, 3))
    rhs[:n] = disp
    sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return dict(ctrl=ctrl, w=sol[:n], a=sol[n:])


def tps_apply(m, pts):
    K = -np.linalg.norm(pts[:, None, :] - m["ctrl"][None, :, :], axis=2)
    return pts + K @ m["w"] + np.hstack([np.ones((len(pts), 1)), pts]) @ m["a"]


# --------------------------------------------------------------------------- #
# 3) 물리 정규화 — 조화 확장 (∇²u = 0, 머리 내부에서)
# --------------------------------------------------------------------------- #
def build_harmonic_solver(mask, D, origin, bbox, n=GRID_N):
    """머리 마스크 내부 격자에서 라플라스 연산자를 조립하고 LU 분해해 둔다.

    경계 노드(내부이면서 바깥 이웃을 가진 노드)는 전부 Dirichlet 이다. 노출된 창 안에서는
    **관측 변위**를, 창 밖에서는 **0**(두개골이 잡고 있다는 사전지식)을 준다. 나머지 내부는
    ∇²u = 0. 진짜 생체역학 모델은 선형 탄성 방정식을 조직별 물성으로 풀지만, 조화 확장은
    그 **단순화된 사촌**이다 — '데이터 없는 곳을 모델이 통제한다'는 성질은 같다.

    행렬 구조는 설정과 무관(경계 노드 집합이 같고 값만 바뀜)하므로 한 번만 분해한다."""
    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
    axes = [np.linspace(lo[i], hi[i], n) for i in range(3)]
    h = (hi - lo) / (n - 1)
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    # 격자점이 머리 안인지 (MR 마스크 조회: world → voxel index)
    vi = np.rint((G - origin) @ np.linalg.inv(D)).astype(int)
    shape_k = np.array(mask.shape)[::-1]          # (k0,k1,k2) 상한
    ok = np.all((vi >= 0) & (vi < shape_k), axis=1)
    inside = np.zeros(len(G), bool)
    inside[ok] = mask[vi[ok, 2], vi[ok, 1], vi[ok, 0]]
    lin = np.arange(n ** 3).reshape(n, n, n)

    # 이웃 쌍 (축별 가중 1/h²)
    rows, cols, wts = [], [], []
    for ax in range(3):
        for sgn in (+1, -1):
            sa, sb = [slice(None)] * 3, [slice(None)] * 3
            sa[ax] = slice(0, n - 1) if sgn > 0 else slice(1, n)
            sb[ax] = slice(1, n) if sgn > 0 else slice(0, n - 1)
            rows.append(lin[tuple(sa)].ravel())
            cols.append(lin[tuple(sb)].ravel())
            wts.append(np.full(rows[-1].size, 1.0 / h[ax] ** 2))
    R, C, W = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    keep = inside[R] & inside[C]                  # 머리 내부끼리만 연결
    R, C, W = R[keep], C[keep], W[keep]

    idx_in = np.where(inside)[0]
    remap = -np.ones(n ** 3, int)
    remap[idx_in] = np.arange(len(idx_in))
    deg = np.bincount(R, weights=W, minlength=n ** 3)
    n_nb = np.bincount(R, minlength=n ** 3)
    bnd = inside & (n_nb < 6)                     # 바깥 이웃이 있는 내부 노드 = 경계

    free_glob = idx_in[~bnd[idx_in]]
    bnd_glob = idx_in[bnd[idx_in]]
    off = ~bnd[R]                                 # 자유 노드 행의 비대각 항만
    N = len(idx_in)
    data = np.concatenate([-W[off], deg[free_glob], np.ones(len(bnd_glob))])
    ri = np.concatenate([remap[R[off]], remap[free_glob], remap[bnd_glob]])
    ci = np.concatenate([remap[C[off]], remap[free_glob], remap[bnd_glob]])
    A = sp.coo_matrix((data, (ri, ci)), shape=(N, N)).tocsc()
    return dict(n=n, lo=lo, hi=hi, G=G, inside=inside, idx_in=idx_in,
                bnd_local=remap[bnd_glob], bnd_pts=G[bnd_glob], lu=splu(A), N=N)


def harmonic_solve(H, obs_pts, obs_disp, r_obs):
    """경계값을 채우고(창 안=관측, 창 밖=0) 풀어서 격자 변위장을 만든다."""
    bc = np.zeros((H["N"], 3))
    if len(obs_pts):
        dist, idx = cKDTree(obs_pts).query(H["bnd_pts"], k=1)
        seen = dist < r_obs
        vals = np.zeros((len(H["bnd_pts"]), 3))
        vals[seen] = obs_disp[idx[seen]]
        bc[H["bnd_local"]] = vals
    U_in = np.column_stack([H["lu"].solve(bc[:, c]) for c in range(3)])
    U = np.zeros((H["n"] ** 3, 3))
    U[H["idx_in"]] = U_in
    out = ~H["inside"]                            # 바깥 노드는 최근접 내부값으로 채움
    if out.any():
        _, nn = cKDTree(H["G"][H["inside"]]).query(H["G"][out], k=1)
        U[out] = U_in[nn]
    return dict(U=U.reshape(H["n"], H["n"], H["n"], 3), n=H["n"], lo=H["lo"], hi=H["hi"])


def harmonic_apply(field, pts):
    """격자 변위장을 삼선형 보간해 임의 점에 적용."""
    n, lo, hi = field["n"], field["lo"], field["hi"]
    t = np.clip((pts - lo) / (hi - lo) * (n - 1), 0, n - 1 - 1e-9)
    i0 = np.floor(t).astype(int)
    f = t - i0
    U, out = field["U"], np.zeros_like(pts)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0])
                     * (f[:, 1] if dy else 1 - f[:, 1])
                     * (f[:, 2] if dz else 1 - f[:, 2]))
                out += w[:, None] * U[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return pts + out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
METHODS = ("rigid", "tps", "tps_prior", "harmonic")
LABEL = {"rigid": "rigid ICP", "tps": "free-form warp (TPS)",
         "tps_prior": "TPS + skull prior", "harmonic": "physics-regularized"}


def main(seed=5, quick=False):
    anat.fetch()
    vol, D, origin = anat.load_nrrd()
    surface, mask = anat.head_surface(vol, D, origin)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(surface), min(anat.MODEL_POINTS, len(surface)), replace=False)
    model = surface[sel]                       # 수술 전 영상의 표면(계획 좌표계)
    normals = reg.estimate_normals(model, k=12)

    center = model.mean(0)
    window_c = model[int(np.argmax(model[:, 2]))]        # 개두창 = 정수리 근처
    sag_dir = np.array([0.0, 0.0, -1.0])
    inward = (center - window_c) / np.linalg.norm(center - window_c)
    targets = window_c + np.array(TARGET_DEPTHS_MM)[:, None] * 1e-3 * inward
    depths = np.array(TARGET_DEPTHS_MM)

    print("=== 51. 변형 정합: 표면 데이터, 심부 표적, 그 사이의 모델 ===")
    print(f"데이터: exp 49 와 같은 실 인체 MR 표면 {len(model)}점")
    print(f"표적: 개두창 바로 아래 깊이 {TARGET_DEPTHS_MM} mm — 병변은 노출면 아래에 있다")
    print(f"변형: 개두창 중력 침하 {SAG_MM:.0f} mm(감쇠 {DECAY_MM:.0f} mm) + 견인 "
          f"{BULGE_MM:.0f} mm — 문헌의 brain shift 자릿수")

    r = np.random.default_rng(seed)
    axis = r.normal(size=3); axis /= np.linalg.norm(axis)
    T_place = reg.pose_T(so3_exp(axis * np.deg2rad(6.0)), r.uniform(-0.015, 0.015, 3))

    def to_patient(pts, deformed=True):
        """계획 좌표 → (변형) → (미지의 배치) 수술 중 좌표. 정답 경로."""
        q = pts + deformation(pts, window_c, sag_dir, inward) if deformed else pts
        return reg.apply_T(T_place, q)

    tgt_true = to_patient(targets)
    u_tgt = np.linalg.norm(deformation(targets, window_c, sag_dir, inward), axis=1)
    u_srf = np.linalg.norm(deformation(model, window_c, sag_dir, inward), axis=1)
    print("변형량: 개두창 표면 {:.2f} mm → 표적 ".format(u_srf.max() * 1e3)
          + " ".join(f"{d:.0f}mm:{v*1e3:.2f}" for d, v in zip(depths, u_tgt)) + " mm")
    print("  ↑ 표적에서의 이 몫이 강체 정합으로는 **원리적으로** 잡을 수 없는 오차다")

    # ---- 단계 1: 개두 **전** 강체 정합 (온전한 두피 전체, 변형 없음) ----
    # 임상 워크플로 그대로다. 이렇게 해야 이후 스윕이 '강체 ICP 조건화'가 아니라
    # '변형 복원'만 재도록 분리된다.
    land = anat.pick_landmarks(surface, k=5)
    digit = to_patient(surface[land], deformed=False) + r.normal(0, LANDMARK_NOISE, (len(land), 3))
    init = anat.procrustes(digit, surface[land])             # probe → image
    pre_idx = r.choice(len(model), size=N_PROBE, replace=False)
    pre_dst = to_patient(model[pre_idx], deformed=False) + r.normal(0, PROBE_NOISE, (N_PROBE, 3))
    T_rigid = real._icp2(pre_dst, model, normals, init)
    T_inv = np.linalg.inv(T_rigid)
    rigid_only = float(np.mean(np.linalg.norm(
        reg.apply_T(T_inv, targets) - to_patient(targets, deformed=False), axis=1)))
    print(f"단계 1 — 개두 전 강체 정합(온전한 두피 {N_PROBE}점): 변형이 없을 때 표적 오차 "
          f"{rigid_only*1e3:.3f} mm (exp 49 수준). 이제 머리를 연다.")

    # 조화 확장 연산자: 설정과 무관하므로 한 번만 조립·LU 분해
    pad = 0.01
    bbox = (model.min(0) - pad, model.max(0) + pad)
    H = build_harmonic_solver(mask, D, origin, bbox)
    step = float(np.max((bbox[1] - bbox[0]) / (GRID_N - 1)))
    print(f"조화 확장 격자 {GRID_N}³ → 머리 내부 노드 {H['N']}개 "
          f"(경계 {len(H['bnd_local'])}개, 간격 {step*1e3:.1f} mm)")

    v_model = (model - center) / np.linalg.norm(model - center, axis=1, keepdims=True)
    w_dir = (window_c - center) / np.linalg.norm(window_c - center)
    ang_model = np.degrees(np.arccos(np.clip(v_model @ w_dir, -1, 1)))

    cache: dict[float, dict] = {}

    def observe(exposure_deg):
        """단계 2 — 개두 **후** 창 안 표면만 디지타이징 → 영상좌표에서 본 잔여 변위."""
        if exposure_deg in cache:
            return cache[exposure_deg]
        rr = np.random.default_rng(int(exposure_deg))
        cand = np.where(ang_model < exposure_deg)[0]
        idx = rr.choice(cand, size=min(N_PROBE, len(cand)), replace=False)
        src = model[idx]                                     # 영상좌표 대응점
        dst = to_patient(src) + rr.normal(0, PROBE_NOISE, (len(idx), 3))
        disp = reg.apply_T(T_rigid, dst) - src
        out_idx = np.where(ang_model > exposure_deg + 25.0)[0]   # 창 밖 = 두개골이 잡는 곳
        anchors = model[rr.choice(out_idx, size=min(N_TPS_ANCHOR, len(out_idx)),
                                  replace=False)]
        cache[exposure_deg] = dict(src=src, disp=disp, anchors=anchors,
                                   field=harmonic_solve(H, src, disp, r_obs=1.5 * step),
                                   frac=len(cand) / len(model))
        return cache[exposure_deg]

    def evaluate(exposure_deg, lam=TPS_LAMBDA, verbose=True):
        st = observe(exposure_deg)
        src, disp = st["src"], st["disp"]
        rr = np.random.default_rng(7)
        sub = rr.choice(len(src), size=min(N_TPS_CTRL, len(src)), replace=False)

        m_tps = tps_fit(src[sub], disp[sub], lam=lam)
        m_pri = tps_fit(np.vstack([src[sub], st["anchors"]]),
                        np.vstack([disp[sub], np.zeros_like(st["anchors"])]), lam=lam)
        warp = {"rigid": lambda p: p,
                "tps": lambda p, m=m_tps: tps_apply(m, p),
                "tps_prior": lambda p, m=m_pri: tps_apply(m, p),
                "harmonic": lambda p, f=st["field"]: harmonic_apply(f, p)}

        tre, per_depth, srf = {}, {}, {}
        for k in METHODS:
            est = reg.apply_T(T_inv, warp[k](targets))       # 표적의 추정 환자좌표
            e = np.linalg.norm(est - tgt_true, axis=1)
            per_depth[k], tre[k] = e, float(e.mean())
            srf[k] = float(np.sqrt(np.mean(np.sum((warp[k](src) - (src + disp)) ** 2, axis=1))))
        if verbose:
            print(f"  노출 {exposure_deg:3.0f}° (표면의 {st['frac']*100:2.0f}%) │ TRE " +
                  " ".join(f"{k} {tre[k]*1e3:6.2f}" for k in METHODS) +
                  " mm │ 표면잔차 " + " ".join(f"{srf[k]*1e3:5.2f}" for k in METHODS))
        return dict(exposure=exposure_deg, frac=st["frac"], tre=tre, srf=srf,
                    per_depth=per_depth)

    print("-" * 104)
    print("[노출 창 스윕] 강체 정합은 고정해 두고, 창을 통해 본 표면만으로 변형을 복원한다")
    print("  TRE = 심부 표적 오차(중요한 것) · 표면잔차 = 수술 중 실제로 볼 수 있는 지표")
    rows = [evaluate(e) for e in (25.0, 45.0, 70.0, 110.0)]

    print("-" * 104)
    print("[깊이별] 표면 정보는 얼마나 깊이까지 닿는가 (노출 70°, 표적 깊이별 오차 mm)")
    mid = rows[2]
    print("  깊이[mm] " + " ".join(f"{d:7.0f}" for d in depths)
          + "   ← 변형량 " + " ".join(f"{v*1e3:.1f}" for v in u_tgt))
    for k in METHODS:
        print(f"  {LABEL[k]:<22}" + " ".join(f"{v*1e3:7.2f}" for v in mid["per_depth"][k]))
    dr = mid["per_depth"]["harmonic"] / np.maximum(mid["per_depth"]["rigid"], 1e-9)
    print(f"  → 물리 정규화가 남기는 비율: 얕은 표적 {dr[0]*100:.0f}% → "
          f"깊은 표적 {dr[-1]*100:.0f}% (깊을수록 복원이 어렵다)"
          if dr[0] < dr[-1] else
          f"  → 물리 정규화가 남기는 비율: 얕은 표적 {dr[0]*100:.0f}% → "
          f"깊은 표적 {dr[-1]*100:.0f}%")

    wide, tight = rows[-1], rows[0]
    print("-" * 104)
    print("[핵심 1] 표면을 더 잘 맞춘다고 표적을 더 잘 맞히는 것이 아니다")
    for tag, rw in (("넓은 노출", wide), ("좁은 노출", tight)):
        best_s = min(METHODS, key=lambda k: rw["srf"][k])
        best_t = min(METHODS, key=lambda k: rw["tre"][k])
        print(f"  {tag}({rw['exposure']:.0f}°): 표면잔차 최소 = {best_s} "
              f"({rw['srf'][best_s]*1e3:.2f} mm) / 심부 최소 = {best_t} "
              f"({rw['tre'][best_t]*1e3:.2f} mm)"
              + ("  ← 일치" if best_s == best_t else "  ← **불일치**"))
    print(f"  강체 대비: 넓은 노출 {wide['tre']['rigid']*1e3:.2f} → "
          f"{min(wide['tre'][k] for k in METHODS)*1e3:.2f} mm / 좁은 노출 "
          f"{tight['tre']['rigid']*1e3:.2f} → {min(tight['tre'][k] for k in METHODS)*1e3:.2f} mm")

    print("[핵심 2] 좁은 노출에서 갈린 것은 보간기가 아니라 **사전지식**이다")
    print(f"  같은 TPS 보간기에 두개골 사전지식(창 밖 변위 0)만 넣었을 때: "
          f"{tight['tre']['tps']*1e3:.2f} → {tight['tre']['tps_prior']*1e3:.2f} mm "
          f"({tight['tre']['tps']/max(tight['tre']['tps_prior'],1e-9):.1f}배). "
          f"보간기는 그대로다 — 바뀐 것은 가정 하나뿐이다.")
    print(f"  넓은 노출({wide['exposure']:.0f}°)에서는 그 이득이 사라진다: "
          f"{wide['tre']['tps']*1e3:.2f} → {wide['tre']['tps_prior']*1e3:.2f} mm "
          f"— 데이터가 충분하면 사전지식은 할 일이 없다")
    harmful = [r0 for r0 in rows if r0["tre"]["tps"] > r0["tre"]["rigid"]]
    if harmful:
        print("  ⚠ 사전지식 없는 자유형 워프가 **강체보다 나빠지는** 구간: "
              + ", ".join(f"{r0['exposure']:.0f}°" for r0 in harmful))

    # ---- 자유도: λ 와 제어점 수는 같은 손잡이다 ----
    print("-" * 104)
    print("[자유도 1 — λ] 작으면 표면에 과적합, 크면 아핀에 수렴 (노출 45°)")
    lam_rows = []
    for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1e0):
        rr = evaluate(45.0, lam=lam, verbose=False)
        lam_rows.append((lam, rr))
        print(f"  λ={lam:7.0e} │ 표면잔차 {rr['srf']['tps']*1e3:5.2f} mm │ "
              f"심부 TRE {rr['tre']['tps']*1e3:6.2f} mm  (사전지식 포함: "
              f"{rr['tre']['tps_prior']*1e3:6.2f} mm)")
    by_srf = min(lam_rows, key=lambda z: z[1]["srf"]["tps"])
    by_tre = min(lam_rows, key=lambda z: z[1]["tre"]["tps"])
    print(f"  → 표면 최적 λ={by_srf[0]:.0e} / 심부 최적 λ={by_tre[0]:.0e} — "
          + ("일치" if by_srf[0] == by_tre[0] else
             f"불일치, 볼 수 있는 지표로 튜닝하면 심부에서 "
             f"{(by_srf[1]['tre']['tps']-by_tre[1]['tre']['tps'])*1e3:.2f} mm 손해"))

    print("[자유도 2 — 제어점 수] 같은 손잡이의 다른 이름. 노출 면적에 맞춰야 한다")
    ctrl_rows = {}
    for exp_deg in (45.0, 110.0):
        st = observe(exp_deg)
        seq = []
        for nc in N_CTRL_SWEEP:
            rr = np.random.default_rng(7)
            sub = rr.choice(len(st["src"]), size=min(nc, len(st["src"])), replace=False)
            mdl = tps_fit(st["src"][sub], st["disp"][sub], lam=TPS_LAMBDA)
            est = reg.apply_T(T_inv, tps_apply(mdl, targets))
            seq.append(float(np.mean(np.linalg.norm(est - tgt_true, axis=1))))
        ctrl_rows[exp_deg] = seq
        best = N_CTRL_SWEEP[int(np.argmin(seq))]
        print(f"  노출 {exp_deg:3.0f}° │ " +
              " ".join(f"{n}:{v*1e3:5.2f}" for n, v in zip(N_CTRL_SWEEP, seq)) +
              f" mm  → 최적 {best}개")
    b45 = N_CTRL_SWEEP[int(np.argmin(ctrl_rows[45.0]))]
    b110 = N_CTRL_SWEEP[int(np.argmin(ctrl_rows[110.0]))]
    print(f"  → 최적 제어점 수가 노출에 따라 {b45} → {b110} 로 움직인다. 고정해 두면 "
          f"한쪽에서 손해를 본다 — 앞의 노출 스윕에서 TPS 가 넓은 노출에서 되레 나빠진 이유가 "
          f"이것이다(제어점 {N_TPS_CTRL}개 고정).")

    # ---- 격자 해상도: 더 정확히 푸는 것이 항상 더 정확한 답은 아니다 ----
    print("-" * 104)
    print("[모델 편향] 조화 확장의 격자를 촘촘히 하면 답이 좋아질까?")
    print(f"  (노출 70°, 가장 얕은 표적 깊이 {depths[0]:.0f} mm 의 참 변형 "
          f"{u_tgt[0]*1e3:.2f} mm 과 비교)")
    grid_rows = []
    st70 = observe(70.0)
    for n in (GRID_SWEEP[::2] if quick else GRID_SWEEP):
        Hn = build_harmonic_solver(mask, D, origin, bbox, n=n)
        stp = float(np.max((bbox[1] - bbox[0]) / (n - 1)))
        f = harmonic_solve(Hn, st70["src"], st70["disp"], r_obs=1.5 * stp)
        pred = harmonic_apply(f, targets) - targets          # 예측한 변위
        est = reg.apply_T(T_inv, targets + pred)
        e = float(np.mean(np.linalg.norm(est - tgt_true, axis=1)))
        ratio = float(np.linalg.norm(pred[0]) / np.linalg.norm(u_tgt[0]))
        grid_rows.append((n, stp, Hn["N"], e, ratio))
        print(f"  격자 {n}³ (간격 {stp*1e3:4.1f} mm, 내부 노드 {Hn['N']:6d}) → "
              f"심부 TRE {e*1e3:5.2f} mm │ 얕은 표적 변위 예측 "
              f"{np.linalg.norm(pred[0])*1e3:5.2f} mm = 참값의 {ratio*100:3.0f}%")
    if grid_rows[-1][3] > grid_rows[0][3]:
        print(f"  → **나빠진다** ({grid_rows[0][3]*1e3:.2f} → {grid_rows[-1][3]*1e3:.2f} mm). "
              "이산화 오차가 아니라 **모델 편향**이다.")
        print(f"     조화 해는 참 변형보다 **빨리 감쇠**해서 깊이에서 과소예측한다"
              f"(참값의 {grid_rows[0][4]*100:.0f}% → {grid_rows[-1][4]*100:.0f}%). "
              "성긴 격자가 더 나았던 건 정확해서가 아니라, 경계 노드가 실제 표면보다 한 칸 "
              "안쪽에 놓여 표면 변위를 더 깊이 밀어 넣었기 때문이다 — **맞는 답이 아니라 "
              "우연히 상쇄된 오차**다.")
        print("     격자를 촘촘히 할수록 **틀린 방정식을 더 정확히** 풀 뿐이다 — "
              "수치 수렴과 모델 타당성은 다른 문제이고, 수렴 검사는 후자를 증명하지 못한다.")

    # ---- 그림 ----
    fig, axg = plt.subplots(2, 3, figsize=(16.8, 9.2))
    axes = axg.ravel()
    COLORS = ("0.45", "crimson", "darkorange", "seagreen")

    ax = axes[0]
    s = model[::4]
    mag = np.linalg.norm(deformation(s, window_c, sag_dir, inward), axis=1) * 1e3
    sc = ax.scatter(s[:, 1] * 1e3, s[:, 2] * 1e3, s=4, c=mag, cmap="magma")
    plt.colorbar(sc, ax=ax, label="deformation [mm]")
    ax.scatter(targets[:, 1] * 1e3, targets[:, 2] * 1e3, marker="*", s=140,
               color="tab:cyan", edgecolor="k", linewidth=0.5, label="deep targets", zorder=3)
    ax.scatter([window_c[1] * 1e3], [window_c[2] * 1e3], marker="v", s=90,
               color="lime", edgecolor="k", linewidth=0.5, label="craniotomy", zorder=3)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title("Brain shift on a real MR head surface", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    ex = [r0["exposure"] for r0 in rows]
    for k, c in zip(METHODS, COLORS):
        ax.plot(ex, [r0["tre"][k] * 1e3 for r0 in rows], "-o", color=c, label=LABEL[k])
    ax.set_xlabel("exposed surface [deg of arc]")
    ax.set_ylabel("deep-target error [mm]")
    ax.set_title("Less exposure = more extrapolation", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    for k, c in zip(METHODS, COLORS):
        ax.plot(depths, mid["per_depth"][k] * 1e3, "-o", color=c, label=LABEL[k])
    ax.plot(depths, u_tgt * 1e3, "--", color="k", lw=1.2, label="true deformation")
    ax.set_xlabel("target depth below craniotomy [mm]")
    ax.set_ylabel("target error [mm]")
    ax.set_title(f"How far surface data reaches (exposure {mid['exposure']:.0f}°)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[3]
    for e_deg, c in zip((45.0, 110.0), ("crimson", "navy")):
        ax.plot(N_CTRL_SWEEP, np.array(ctrl_rows[e_deg]) * 1e3, "-o", color=c,
                label=f"exposure {e_deg:.0f}°")
        b = N_CTRL_SWEEP[int(np.argmin(ctrl_rows[e_deg]))]
        ax.axvline(b, color=c, ls=":", alpha=0.6)
    ax.axvline(N_TPS_CTRL, color="0.4", ls="--", lw=1.2, label=f"fixed budget ({N_TPS_CTRL})")
    ax.set_xscale("log"); ax.set_xlabel("TPS control points  (degrees of freedom)")
    ax.set_ylabel("deep-target error [mm]")
    ax.set_title("The right amount of freedom moves with the data", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[4]
    gn = [g[1] * 1e3 for g in grid_rows]
    ax.plot(gn, [g[3] * 1e3 for g in grid_rows], "-o", color="seagreen")
    for g in grid_rows:
        ax.annotate(f"{g[0]}³", (g[1] * 1e3, g[3] * 1e3), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel("grid spacing [mm]   (finer →)")
    ax.set_ylabel("deep-target error [mm]")
    ax.set_title("Solving it more accurately, more accurately wrong", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[5]
    for k, c, mk in zip(METHODS, COLORS, ("o", "s", "D", "^")):
        ax.plot([r0["srf"][k] * 1e3 for r0 in rows], [r0["tre"][k] * 1e3 for r0 in rows],
                mk, color=c, ms=8, ls="none", label=LABEL[k])
    ax.set_xlabel("surface residual [mm]   (what you can measure)")
    ax.set_ylabel("deep-target error [mm]   (what matters)")
    ax.set_title("Fitting the surface better ≠ hitting the target", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle("51. Deformable registration — surface data, deep targets, "
                 "and the model in between", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "51_deformable_registration.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/51_deformable_registration.png, "
          "assets/51_deformable_registration.png")

    return dict(rows=rows, lam_rows=lam_rows, ctrl_rows=ctrl_rows, grid_rows=grid_rows,
                per_depth=mid["per_depth"], depths=depths, u_target=u_tgt,
                rigid_only_mm=rigid_only * 1e3, u_surface_mm=float(u_srf.max() * 1e3))


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 변형장이 **합성**이다. 심부 TRE 를 재려면 정답이 필요해서 해석적 장을 썼다. 실제
#     brain shift 는 뇌척수액 유출·자세·견인·절제가 얽혀 환자마다 다르고, 여기 쓴 지수 감쇠보다
#     불규칙하다. 절대값이 아니라 **방법 간 상대 비교**로 읽어야 한다.
#   - 조화 확장(∇²u=0)은 선형 탄성 FEM 의 **단순화된 사촌**이며, 이 실험에서 실제로 **편향된**
#     모델로 드러났다(깊이에서 과소예측, 격자를 촘촘히 할수록 악화). 진짜 모델은 조직별
#     탄성계수·비압축성·경막/낫 같은 내부 경계를 넣는다. 그러니 여기서 읽을 것은 "물리 모델이
#     이긴다"가 아니라 **"정규화는 가정을 사는 것이고, 가정이 틀리면 값을 치른다"** 이다.
#   - 이 실험의 사전지식("창 밖 변위 0")은 합성 변형이 실제로 그렇게 생겼기 때문에 맞는다.
#     사전지식이 틀린 환자(광범위 절제·양측 개두)에서는 같은 정규화가 **편향의 원인**이 된다 —
#     정규화는 공짜가 아니라 가정을 사는 것이다.
#   - TPS 대응점을 **정답 대응**으로 줬다(프로브 잡음만 얹음). 실제로는 대응 자체를 non-rigid
#     ICP 로 찾아야 하고, 그 오대응이 또 하나의 오차원이다.
#   - 표적이 해부학적 병변이 아니라 기하적으로 정의한 심부 점이다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
