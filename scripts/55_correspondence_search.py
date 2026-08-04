"""대응을 스스로 찾을 때: 표면을 따라 미끄러진 몫은 표면으로 볼 수 없다.

exp 51~54 의 변형 정합은 전부 **대응을 정답으로 받았다** — 프로브로 찍은 점이 모델의 *어느*
점이었는지 알고 있었다. 실제로는 그걸 찾아야 하고, 찾는 방법은 보통 최근접점(ICP 계열)이다.
이 실험은 그 마지막 "주어진 것"을 뺀다.

--- 왜 이게 단순한 추가 오차원이 아닌가 ---
변형은 두 성분으로 쪼갤 수 있다.
  - **법선 성분**: 표면이 안팎으로 움직인 몫. 기하가 바뀌므로 최근접점이 볼 수 있다.
  - **접선 성분**: 표면이 **자기 자신을 따라 미끄러진** 몫(경막 아래 피질 활강 등).
    매끄러운 표면에서 이건 **기하를 바꾸지 않는다** — 미끄러진 표면은 원래 표면과 같은 모양이다.
    즉 최근접점은 원리적으로 볼 수 없고, 그럼에도 잔차는 0 이 된다.

이건 광학 흐름의 **구멍 문제(aperture problem)** 와 같은 구조다. 그리고 exp 52 의 "표면에
자취를 남기지 않는 심부 모드"와 같은 계열이지만, 이번엔 **표면 위**에서 벌어진다.

--- 미리 말해두는 결론 ---
1. 접선 변형을 0 → 8 mm 로 키우면 **대응 오차가 0.48 → 1.53 mm** 로 따라 늘지만
   **표면 잔차는 0.92 → 1.17 mm 로 거의 그대로**고, 법선 성분은 94% 회수된다.
   보이는 몫은 잘 잡고, 안 보이는 몫은 잔차조차 남기지 않는다.
2. **대응을 찾는 비용은 2.6배다**: 정답 대응 0.54 → 최근접점 **1.41 mm**.
   exp 51~54 가 '주어진 것'으로 두고 있던 몫이 이만큼이다.
3. 표준 처방 셋이 다 실패한다 — 점-대-평면은 오히려 1.75 mm(최근접점이 이미 접선을 지운 뒤라
   남은 접선 잔차에 **진짜 신호가 섞여** 있다), 식별 가능한 특징점 2~16개는 1.37~1.80 mm 로
   무효(편향이 창 **전체**에 깔려 앵커 몇 개로 못 덮는다), 로버스트는 거의 무력하다
   (1.76 → 1.62 mm) — 최근접점이 튄 관측점 근처의 다른 표면점을 찾아 주기 때문에 **대응은
   틀렸는데 변위는 작고 그럴듯해서** 잔차로 볼 것이 없다.
   **이상치는 '틀린 데이터'이고 접선은 '없는 데이터'다. 다른 병이라 다른 약이 필요하다.**
4. 그래서 질문을 바꿨다: 대응 중 몇 %가 기하가 아닌 근거(질감·혈관 패턴·마커)로 잡혀야 하나.
   0% 1.41 → 10% 1.23 → 25% 1.13 → 50% 0.74 → 100% 0.54 mm. **거의 선형이다** — 싼 해법이
   없고, 랜드마크 몇 개가 아니라 **표면 전체에 걸친 대응**이 필요하다.

    python scripts/55_correspondence_search.py
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

from sensor_fusion.se3 import so3_exp  # noqa: E402

reg = import_module("41_surgical_registration")
real = import_module("44_registration_real_scans")
anat = import_module("49_registration_real_anatomy")
deform = import_module("51_deformable_registration")     # tps_fit / tps_apply
probe = import_module("52_probing_the_prior")            # deformation / axis_frame
meas = import_module("53_measurement_changes_it")        # robust_tps_fit (어닐링 + Tukey)

NORMAL_MM = 6.0            # 법선 방향 변형(표면이 들어가고 나온다) [mm]
TANGENT_MM = 5.0           # **접선 방향** 변형(표면이 자기를 따라 미끄러진다) [mm]
DECAY_MM = 45.0
PROBE_NOISE = 1.0e-3
N_PROBE = 900
N_TPS_CTRL = 170
N_TPS_ANCHOR = 130
TPS_LAMBDA = 1e-2
LANDMARK_NOISE = 2.5e-3
EXPOSURE_DEG = 45.0
TARGET_DEPTHS_MM = (20, 35, 50, 70)
MISS_TOL = 2.0e-3
N_FEATURE_SWEEP = (0, 2, 4, 8, 16)
OUTLIER_FRAC = 0.10        # 대응이 아예 엉뚱한 곳에 붙는 비율(주름 건너뛰기 등)


# --------------------------------------------------------------------------- #
# 1) 법선/접선으로 분해된 변형장
# --------------------------------------------------------------------------- #
def field(pts, center, window_c, slide_dir, normal_mm=NORMAL_MM,
          tangent_mm=TANGENT_MM, decay_mm=DECAY_MM):
    """**부피 전체에 하나로 정의된** 변형장. 두 성분의 합이다.

        u(p) = w(p) · [ normal_mm · r̂(p)  +  tangent_mm · ŝ ]

    r̂ 는 무게중심에서 바깥으로 향하는 반경 방향(표면에서는 법선의 매끄러운 대용), ŝ 는 고정된
    미끄러짐 방향, w 는 개두창에서의 거리로 감쇠. 표면 위에서는 첫 항이 주로 **법선**,
    둘째 항이 주로 **접선**으로 분해된다 — 그 분해는 기하가 정하고 내가 손으로 넣지 않는다.

    (처음엔 표면용/심부용 장을 따로 썼다가, 정답 대응으로도 1.17 mm 가 남는 것을 보고 잡았다.
     두 장이 표면에서 일치하지 않으면 '대응의 효과'를 재는 실험이 아니라 '장 불일치'를 재게 된다.)"""
    d = np.linalg.norm(pts - window_c, axis=1)
    w = np.exp(-d / (decay_mm * 1e-3))[:, None]
    r = pts - center
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)
    return w * ((normal_mm * 1e-3) * r + (tangent_mm * 1e-3) * slide_dir[None, :])


def split_on_surface(u, normals):
    """표면 위 변위를 법선/접선으로 쪼갠다 — **보고용**(장을 만드는 데 쓰지 않는다).

    접선 몫이 최근접점에게 보이지 않는 부분이다."""
    un = (np.sum(u * normals, axis=1))[:, None] * normals
    return un, u - un


# --------------------------------------------------------------------------- #
# 2) 대응 찾기
# --------------------------------------------------------------------------- #
def find_correspondence(obs_img, model, tree, mode="p2p", normals=None):
    """관측점(영상좌표)에 대응하는 모델점을 **최근접점으로 찾는다**.

    mode="p2p"   : 최근접점을 그대로 대응으로 쓴다(점-대-점).
    mode="p2plane": 잔차를 **법선 방향만** 남긴다 — 접선 성분이 관측 불가라는 사실을
                    인정하고 억지로 만들어내지 않는 것. exp 41 의 점-대-평면 ICP 와 같은 발상."""
    _, idx = tree.query(obs_img, k=1)
    src = model[idx]
    disp = obs_img - src
    if mode == "p2plane":
        n = normals[idx]
        disp = (np.sum(disp * n, axis=1))[:, None] * n
    return idx, src, disp


def feature_anchors(model, normals, in_win, k, seed=0):
    """접선 방향을 고정해 주는 **식별 가능한 특징점** k개.

    국소 곡률 변화가 큰 점은 미끄러져도 다른 점과 혼동되지 않는다(exp 49 의 랜드마크와 같은 논리).
    farthest-point 로 흩어 고른다 — 한쪽에 몰리면 접선 자유도가 다 구속되지 않는다."""
    if k == 0:
        return np.zeros(0, int)
    sv = anat.surface_variation(model[in_win], k=14)
    cand = in_win[np.argsort(sv)[-max(8 * k, k):]]              # 특징적인 후보들
    rng = np.random.default_rng(seed)
    picked = [int(cand[rng.integers(len(cand))])]
    d = np.linalg.norm(model[cand] - model[picked[0]], axis=1)
    for _ in range(k - 1):
        picked.append(int(cand[int(np.argmax(d))]))
        d = np.minimum(d, np.linalg.norm(model[cand] - model[picked[-1]], axis=1))
    return np.array(picked)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(seed=5, quick=False):
    anat.fetch()
    vol, D, origin = anat.load_nrrd()
    surface, mask = anat.head_surface(vol, D, origin)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(surface), min(anat.MODEL_POINTS, len(surface)), replace=False)
    model = surface[sel]
    normals = reg.estimate_normals(model, k=12)
    tree = cKDTree(model)

    center = model.mean(0)
    window_c = model[int(np.argmax(model[:, 2]))]
    inward = (center - window_c) / np.linalg.norm(center - window_c)
    _, slide_dir, _ = probe.axis_frame(inward)                 # 표면을 따라 미끄러지는 방향
    depths = np.array(TARGET_DEPTHS_MM)
    targets = window_c + depths[:, None] * 1e-3 * inward

    # 법선을 바깥쪽으로 정렬(모양 변화의 부호를 일관되게)
    outward = model - center
    flip = np.sum(normals * outward, axis=1) < 0
    normals[flip] *= -1.0

    print("=== 55. 대응을 스스로 찾을 때: 접선 방향은 표면으로 볼 수 없다 ===")
    print(f"데이터: exp 49·51 과 같은 실 인체 MR 표면 {len(model)}점, 노출 {EXPOSURE_DEG:.0f}°")
    print(f"변형을 둘로 쪼갰다 — **법선 {NORMAL_MM:.0f} mm**(표면 모양이 바뀜, 보인다) + "
          f"**접선 {TANGENT_MM:.0f} mm**(표면을 따라 미끄러짐, 안 보인다), 감쇠 {DECAY_MM:.0f} mm")

    v = (model - center) / np.linalg.norm(model - center, axis=1, keepdims=True)
    w_dir = (window_c - center) / np.linalg.norm(window_c - center)
    ang = np.degrees(np.arccos(np.clip(v @ w_dir, -1, 1)))
    in_win = np.where(ang < EXPOSURE_DEG)[0]
    out_win = np.where(ang > EXPOSURE_DEG + 25.0)[0]

    # ---- 개두 전 강체 정합 (변형 무관, 한 번만) ----
    r0 = np.random.default_rng(seed)
    ax0 = r0.normal(size=3); ax0 /= np.linalg.norm(ax0)
    T_place = reg.pose_T(so3_exp(ax0 * np.deg2rad(6.0)), r0.uniform(-0.015, 0.015, 3))
    land = anat.pick_landmarks(surface, k=5)
    digit = reg.apply_T(T_place, surface[land]) + r0.normal(0, LANDMARK_NOISE, (len(land), 3))
    init = anat.procrustes(digit, surface[land])
    pre_idx = r0.choice(len(model), size=N_PROBE, replace=False)
    pre_dst = reg.apply_T(T_place, model[pre_idx]) + r0.normal(0, PROBE_NOISE, (N_PROBE, 3))
    T_rigid = real._icp2(pre_dst, model, normals, init)
    T_inv = np.linalg.inv(T_rigid)
    print(f"개두 전 강체 정합: 변형 없을 때 표적 오차 "
          f"{np.mean(np.linalg.norm(reg.apply_T(T_inv, targets) - reg.apply_T(T_place, targets), axis=1))*1e3:.3f} mm")

    def observe(rr, tangent_mm=TANGENT_MM, normal_mm=NORMAL_MM, outliers=0.0):
        """창 안 표면을 찍는다. 정답 대응(어느 모델점이었나)도 같이 들고 있는다 — **비교용**."""
        idx = rr.choice(in_win, size=min(N_PROBE, len(in_win)), replace=False)
        src_true = model[idx]
        u = field(src_true, center, window_c, slide_dir, normal_mm, tangent_mm)
        obs_pat = reg.apply_T(T_place, src_true + u) \
            + rr.normal(0, PROBE_NOISE, (len(idx), 3))
        obs_img = reg.apply_T(T_rigid, obs_pat)
        if outliers:                                            # 대응이 엉뚱하게 붙을 관측
            bad = rr.random(len(idx)) < outliers
            if bad.any():
                jump = rr.normal(size=(int(bad.sum()), 3))
                jump /= np.linalg.norm(jump, axis=1, keepdims=True)
                obs_img[bad] = obs_img[bad] + jump * (rr.uniform(6e-3, 15e-3, int(bad.sum())))[:, None]
        return idx, src_true, obs_img

    def fit_and_score(rr, src, disp, n_feat=0, robust=False, feat=None):
        """앵커(창 밖 변위 0) + 대응으로 TPS 를 맞추고 심부 표적 오차를 잰다."""
        sub = np.random.default_rng(7).choice(len(src), size=min(N_TPS_CTRL, len(src)),
                                             replace=False)
        anchors = model[rr.choice(out_win, size=min(N_TPS_ANCHOR, len(out_win)), replace=False)]
        ctrl = [src[sub], anchors]
        dd = [disp[sub], np.zeros_like(anchors)]
        lam = [np.full(len(sub), TPS_LAMBDA), np.full(len(anchors), TPS_LAMBDA)]
        mk = [np.ones(len(sub), bool), np.zeros(len(anchors), bool)]
        if feat is not None and len(feat):                      # 특징점은 정답 대응을 준다
            fp = model[feat]
            ctrl.append(fp)
            dd.append(field(fp, center, window_c, slide_dir)
                      + rr.normal(0, LANDMARK_NOISE, (len(feat), 3)))
            lam.append(np.full(len(feat), TPS_LAMBDA * 0.01))   # 믿을 수 있으니 강하게
            mk.append(np.zeros(len(feat), bool))
        C, DD, LM = np.vstack(ctrl), np.vstack(dd), np.concatenate(lam)
        m = (meas.robust_tps_fit(C, DD, LM, mask=np.concatenate(mk)) if robust
             else deform.tps_fit(C, DD, lam=LM))
        est = reg.apply_T(T_inv, deform.tps_apply(m, targets))
        true = reg.apply_T(T_place, targets + field(targets, center, window_c, slide_dir))
        srf = float(np.sqrt(np.mean(np.sum(
            (deform.tps_apply(m, src[sub]) - (src[sub] + disp[sub])) ** 2, axis=1))))
        return float(np.mean(np.linalg.norm(est - true, axis=1))), srf

    # ------------------------------------------------------------------ #
    # A. 접선 성분은 최근접점이 볼 수 없다
    # ------------------------------------------------------------------ #
    print("-" * 100)
    print("[A] 접선 변형을 키우면 **대응 오차가 그만큼 늘고, 표면 잔차는 그대로다**")
    print("     설정[mm]   실접선[mm]   대응 오차[mm]   표면 잔차[mm]   법선 회수율")
    A_rows = []
    for tan in (0.0, 2.0, 5.0, 8.0):
        rr = np.random.default_rng([seed, int(tan * 10)])
        idx, src_true, obs = observe(rr, tangent_mm=tan)
        nn_idx, src_nn, disp_nn = find_correspondence(obs, model, tree, "p2p", normals)
        corr_err = float(np.mean(np.linalg.norm(model[nn_idx] - src_true, axis=1)))
        _, srf = fit_and_score(rr, src_nn, disp_nn)
        u_true = field(src_true, center, window_c, slide_dir, NORMAL_MM, tan)
        u_n, u_t = split_on_surface(u_true, normals[idx])
        tan_true = float(np.mean(np.linalg.norm(u_t, axis=1)))
        got_n = float(np.mean(np.sum(disp_nn * normals[nn_idx], axis=1))
                      / max(np.mean(np.sum(u_n * normals[idx], axis=1)), 1e-12))
        A_rows.append((tan, tan_true, corr_err, srf, got_n))
        print(f"     {tan:6.1f}   {tan_true*1e3:8.2f}   {corr_err*1e3:11.2f}   "
              f"{srf*1e3:14.2f}   {got_n*100:12.0f}%")
    print(f"  → 접선을 0 → {A_rows[-1][0]:.0f} mm 로 키우면 대응 오차가 "
          f"{A_rows[0][2]*1e3:.2f} → {A_rows[-1][2]*1e3:.2f} mm 로 늘어난다. 그런데 표면 잔차는 "
          f"{A_rows[0][3]*1e3:.2f} → {A_rows[-1][3]*1e3:.2f} mm 로 거의 그대로고, "
          f"법선 성분은 {A_rows[-1][4]*100:.0f}% 회수된다.")
    print("     **보이는 몫은 잘 잡고, 안 보이는 몫은 잔차조차 남기지 않는다** — 구멍 문제의 정의다.")

    # ------------------------------------------------------------------ #
    # B. 무엇이 접선 자유도를 되찾아 주는가
    # ------------------------------------------------------------------ #
    print("-" * 100)
    n_reps = 4 if quick else 10
    print(f"[B] 심부 표적 오차 — 대응을 어떻게 다루느냐로 갈린다 (시드 {n_reps}개 평균)")

    def bench(mode, k=0, robust=False):
        acc_e, acc_s = [], []
        for rep in range(n_reps):
            rr2 = np.random.default_rng([seed, 100 + rep])
            _, st, ob = observe(rr2)
            feat = feature_anchors(model, normals, in_win, k, seed=rep)
            if mode == "gt":
                e, s = fit_and_score(rr2, st, ob - st, feat=feat, robust=robust)
            else:
                _, sn, dp = find_correspondence(ob, model, tree, mode, normals)
                e, s = fit_and_score(rr2, sn, dp, feat=feat, robust=robust)
            acc_e.append(e); acc_s.append(s)
        return float(np.mean(acc_e)), float(np.mean(acc_s))

    e_gt, s_gt = bench("gt")
    e_p2p, s_p2p = bench("p2p")
    e_p2pl, s_p2pl = bench("p2plane")
    print("     대응 처리                                심부 오차   표면 잔차")
    print(f"     정답 대응 (exp 51~54 의 가정)            {e_gt*1e3:7.2f} mm  {s_gt*1e3:7.2f} mm")
    print(f"     최근접점, 점-대-점                       {e_p2p*1e3:7.2f} mm  {s_p2p*1e3:7.2f} mm")
    print(f"     최근접점, 점-대-평면(법선만)              {e_p2pl*1e3:7.2f} mm  {s_p2pl*1e3:7.2f} mm")
    B_rows = []
    for k in N_FEATURE_SWEEP:
        e_f, s_f = bench("p2p", k=k)
        B_rows.append((k, e_f, s_f))
        if k:
            print(f"       점-대-점 + 식별 가능한 특징점 {k:2d}개      "
                  f"{e_f*1e3:7.2f} mm  {s_f*1e3:7.2f} mm")
    best_k, best_e, _ = min(B_rows, key=lambda z: z[1])
    print(f"  → **대응을 찾는 비용이 {e_p2p/e_gt:.1f}배다**: 정답 대응 {e_gt*1e3:.2f} → "
          f"최근접점 {e_p2p*1e3:.2f} mm. exp 51~54 가 '주어진 것'으로 두고 있던 몫이 이만큼이다.")
    if e_p2pl > e_p2p:
        print(f"     점-대-평면은 오히려 나쁘다({e_p2pl*1e3:.2f} mm). 접선 잔차를 **전부** 버리는데,")
        print(f"     최근접점이 이미 접선을 상당히 지운 뒤라 남은 접선 잔차에는 **진짜 신호도 섞여**")
        print(f"     있기 때문이다. 점-대-평면은 접선 잔차가 순수 잡음일 때 옳은 선택이고, 여기는 아니다.")
    else:
        print(f"     점-대-평면이 {e_p2pl*1e3:.2f} mm 로 낫다 — 가짜 접선 변위를 만들지 않는 값이다.")
    gain = e_p2p - best_e
    if gain > 0.05e-3:
        print(f"     그리고 **식별 가능한 점 {best_k}개**가 접선 자유도를 고정해 {best_e*1e3:.2f} mm "
              f"({gain*1e3:.2f} mm 회복, 정답 대응까지 간극의 "
              f"{gain/max(e_p2p-e_gt,1e-12)*100:.0f}%). exp 49 의 랜드마크가 여기서 **다른 이유로** "
              f"다시 필요해진다 — 그때는 조대정렬용이었고 지금은 **접선 자유도 고정용**이다.")
    else:
        print(f"     특징점을 {best_k}개까지 넣어도 {best_e*1e3:.2f} mm 로 거의 나아지지 않는다 —")
        print(f"     편향이 창 **전체**에 깔려 있어서 앵커 몇 개로는 못 덮는다.")

    # ---- 그럼 얼마나 많은 대응이 기하가 아닌 근거로 잡혀야 하는가 ----
    print(f"\n     그래서 질문을 바꾼다: **대응 중 몇 %가 기하가 아닌 근거**(질감·혈관 패턴·마커)로")
    print(f"     잡혀야 하는가. 나머지는 최근접점으로 둔다.")
    print("     정답 대응 비율   심부 오차[mm]")
    F_rows = []
    for frac in (0.0, 0.1, 0.25, 0.5, 1.0):
        acc = []
        for rep in range(n_reps):
            rr2 = np.random.default_rng([seed, 100 + rep])   # [B] 와 같은 시드 → frac=0 이 p2p 와 일치
            _, st, ob = observe(rr2)
            _, sn, _ = find_correspondence(ob, model, tree, "p2p", normals)
            good = np.random.default_rng([seed, 400 + rep]).random(len(st)) < frac
            src_mix = np.where(good[:, None], st, sn)
            acc.append(fit_and_score(rr2, src_mix, ob - src_mix)[0])
        F_rows.append((frac, float(np.mean(acc))))
        print(f"     {frac*100:12.0f}%   {F_rows[-1][1]*1e3:12.2f}")
    ok = next((f for f, e in F_rows if e < 1.2 * e_gt), None)
    print(f"  → 정답 대응 수준({e_gt*1e3:.2f} mm)의 20% 안으로 들어오려면 "
          + (f"**약 {ok*100:.0f}%**가 비기하학적 근거로 잡혀야 한다." if ok
             else "100% 로도 부족하다.")
          + " 소수의 랜드마크로는 안 되고,")
    print("     **표면 전체에 걸친 대응**(피질 혈관 패턴 스테레오비전, 삽입 마커)이 필요하다는 뜻이다.")

    # ------------------------------------------------------------------ #
    # C. 대응이 아예 틀리면 (주름 건너뛰기)
    # ------------------------------------------------------------------ #
    print("-" * 100)
    print(f"[C] 대응 {OUTLIER_FRAC*100:.0f}% 가 엉뚱한 곳에 붙을 때 (exp 53 의 로버스트 재사용)")
    acc = {"ls": [], "rb": []}
    for rep in range(n_reps):
        rr2 = np.random.default_rng([seed, 200 + rep])
        _, _, ob = observe(rr2, outliers=OUTLIER_FRAC)
        _, sn, dp = find_correspondence(ob, model, tree, "p2p", normals)
        feat = feature_anchors(model, normals, in_win, best_k, seed=rep)
        acc["ls"].append(fit_and_score(rr2, sn, dp, feat=feat)[0])
        acc["rb"].append(fit_and_score(rr2, sn, dp, feat=feat, robust=True)[0])
    e_ls, e_rb = float(np.mean(acc["ls"])), float(np.mean(acc["rb"]))
    print(f"     최소제곱    심부 {e_ls*1e3:6.2f} mm")
    print(f"     로버스트    심부 {e_rb*1e3:6.2f} mm  (exp 53 의 어닐링+Tukey)")
    print(f"  → 로버스트가 여기서는 거의 힘을 못 쓴다({e_ls*1e3:.2f} → {e_rb*1e3:.2f} mm). "
          f"이유가 이 실험답다 —")
    print("     관측점을 6~15 mm 튀게 만들면 최근접점은 **그 자리 근처의 다른 표면점**을 찾아 준다.")
    print("     즉 대응은 틀렸는데 **변위 벡터는 작고 그럴듯해서**, 잔차 기반 로버스트가 볼 것이 없다.")
    print("     exp 53 이 '실제 오대응은 비슷하게 생긴 구조에 붙어 더 못 잡는다'고 적어둔 그 경우다.")
    print(f"     그리고 어느 쪽이든 정답 대응({e_gt*1e3:.2f} mm)까지 못 간다 — **이상치는 '틀린")
    print("     데이터'이고 접선은 '없는 데이터'다. 다른 병이라 다른 약이 필요하다.**")

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    ax = axes[0]
    tans = [a[0] for a in A_rows]
    ax.plot(tans, [a[1] * 1e3 for a in A_rows], "-o", color="crimson",
            label="correspondence error")
    ax.plot(tans, [a[2] * 1e3 for a in A_rows], "-o", color="0.45",
            label="surface residual (measurable)")
    ax.plot(tans, tans, ":", color="crimson", alpha=0.5, label="tangential slide (truth)")
    ax.set_xlabel("tangential deformation [mm]"); ax.set_ylabel("[mm]")
    ax.set_title("Sliding along the surface leaves no residual", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    names = ["ground-truth\ncorrespondence", "nearest point\n(p2p)", "nearest point\n(p2plane)",
             f"+ {best_k} identifiable\nfeatures"]
    vals = [e_gt * 1e3, e_p2p * 1e3, e_p2pl * 1e3, best_e * 1e3]
    ax.bar(np.arange(4), vals, color=("seagreen", "crimson", "darkorange", "royalblue"),
           alpha=0.8)
    ax.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1.2, label="tolerance")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("deep-target error [mm]")
    ax.set_title("What restores the tangential degree of freedom", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    ax = axes[2]
    fr = np.array([f[0] for f in F_rows]) * 100
    ax.plot(fr, [f[1] * 1e3 for f in F_rows], "-o", color="royalblue",
            label="deep-target error")
    ax.plot([0, 100], [e_p2p * 1e3, e_gt * 1e3], ":", color="0.5", label="linear reference")
    ax.axhline(e_gt * 1e3, color="seagreen", ls=":", lw=1.2, label="ground-truth corr.")
    ax.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1, label="tolerance")
    ks = [b[0] for b in B_rows if b[0]]
    ax.plot([0] * len(ks), [b[1] * 1e3 for b in B_rows if b[0]], "x", color="crimson",
            ms=7, label="2–16 landmark anchors")
    ax.set_xlabel("correspondences fixed by non-geometric evidence [%]")
    ax.set_ylabel("deep-target error [mm]")
    ax.set_title("No cheap fix — the gain is roughly linear", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle("55. Searching for correspondence — the aperture problem on a "
                 "deforming surface", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "55_correspondence_search.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/55_correspondence_search.png, assets/55_correspondence_search.png")

    return dict(A_rows=A_rows, B_rows=B_rows, F_rows=F_rows, best_k=best_k, best_e=best_e,
                e_gt=e_gt, e_p2p=e_p2p, e_p2pl=e_p2pl, s_gt=s_gt, s_p2p=s_p2p,
                s_p2pl=s_p2pl, outlier=(e_ls, e_rb), MISS_TOL=MISS_TOL)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 변형장을 **법선/접선으로 깔끔히 쪼갠 합성**이다. 실제 변형은 그렇게 분리되지 않고,
#     표면 곡률이 큰 곳에서는 접선 이동도 기하를 바꿔 일부 관측 가능해진다. 여기 결과는
#     "접선 몫은 원리적으로 덜 관측된다"는 성질을 극단으로 밀어 본 것이다.
#   - 특징점 대응은 **정답으로 줬다**. 실제로는 그 대응 자체도 찾아야 하고 틀릴 수 있다
#     (exp 49 의 랜드마크 식별 오차 2~3 mm 가 여기 들어와야 한다).
#   - 심부 변형장을 표면장의 매끄러운 연장으로 뒀다. exp 52 가 보인 대로 표면과 무관한
#     심부 모드가 따로 있을 수 있고, 그건 이 실험이 다루지 않는다.
#   - 최근접점 대응을 **한 번만** 계산했다(교대 최적화를 수렴까지 돌리지 않음). 반복하면
#     법선 성분은 조금 더 회수되지만 접선 성분은 원리상 그대로다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
