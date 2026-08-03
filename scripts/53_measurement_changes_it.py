"""관측이 대상을 바꾼다: 계통 오차는 평균되지 않는다.

exp 52 는 심부 관측(수술 중 초음파) 하나가 표면이 못 보는 실패 모드를 잡아낸다는 것을 보였다
(표면 게이트 AUROC 0.52 → 심부 관측 1개 0.81). 그런데 그 초음파는 **이상적**이었고,
무엇보다 **검산 관측까지 이상적**이었다 —
정답 변위 + 등방 가우시안 잡음. 실제 iUS 는 세 가지가 다르다.

  (a) **프로브가 조직을 누른다.** 뇌 표면에 대고 스캔하니 국소 함몰이 생기고, 그 함몰은 깊은
      곳까지 조직을 민다. 즉 **재려는 상태가 아니라 재면서 바뀐 상태**를 잰다. 그리고 누르는
      방향은 항상 안쪽이라 **계통 오차**다 — 많이 잰다고 상쇄되지 않는다.
  (b) **깊을수록 나빠진다.** 감쇠와 빔 확산으로 σ 가 깊이에 따라 커진다. 하필 **정보가 가장
      필요한 곳이 가장 못 믿을 곳**이다.
  (c) **특징 대응이 틀린다.** 일부 관측은 그냥 다른 구조에 붙는다 — 거친 이상치.

--- 답하려는 질문 ---
1. 프로브 압박은 얼마나 비싼가? 그리고 **더 많이 재면 좋아지는가?**
2. 깊이에 따라 커지는 잡음은 가중치로 되찾을 수 있는가?
3. 오대응은? (exp 11·15 의 로버스트 커널이 정합에서 다시 나온다)
4. 셋을 다 넣으면 **exp 52 의 AUROC 0.81 은 얼마가 되는가** — 그게 정직한 숫자다.

--- 미리 말해두는 결론 ---
- 잡음(무작위)은 관측을 늘리면 √N 으로 줄지만 **압박 편향(계통)은 그대로 남는다.** 두 곡선이
  나란히 가고 격차가 +0.5 mm 로 고정된다 — 그래서 압박의 몫이 전체 오차의 13% → 31% 로
  **커진다.** 데이터를 모을수록 계통 오차가 지배항이 된다.
- 압박을 **모델로 빼면** 완전히 회복되지만(1.16 mm), 조직 응답 길이를 30% 틀리게 알면 1.37 mm.
  모델을 알아야 뺄 수 있고, 그 모델도 틀린다.
- 깊이 가중(σ(d) 를 λ_i 로)은 12% 회복 — 의미 있지만 작다. **가중은 정보를 만들지 못한다.**
- 오대응 15% 는 최소제곱 TPS 를 1.16 → 2.34 mm 로 망가뜨리고, Tukey IRLS 가 1.63 mm 로 되돌린다
  (exp 11·15 의 로버스트 커널이 정합에서 재사용된다). 단 그 로버스트를 제대로 굴리는 데
  **어닐링 + 재하강 가중 + 알려진 σ 기준**이 다 필요했다 — 아래 `robust_tps_fit` 참고.
- **가장 아픈 것**: exp 52 는 검산 관측까지 이상 센서였다. 검산도 같은 센서로 하면 게이트
  AUROC 가 0.73 → 0.61 로 떨어지고, **대책을 다 써도 0.62 로 제자리다**(검산점을 5개로 늘려도
  0.65). 같은 대책이 교정은 2.84 → 1.82 mm 로 살리는데 말이다.
  이유는 명확하다 — **검산은 편향이 아니라 잡음에 묶여 있다**. 표적 깊이에서 신호(심부 모드
  2~6 mm) 대 검산 1점 잡음(σ(d)·√3 = 3.6~6.2 mm)이 이미 1 이하다.
  **대책은 자기가 겨냥한 오차원만 고치고, 게이트의 천장은 통계량이 아니라 모달리티가 정한다.**

    python scripts/53_measurement_changes_it.py
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

from sensor_fusion.se3 import so3_exp  # noqa: E402

reg = import_module("41_surgical_registration")
real = import_module("44_registration_real_scans")
anat = import_module("49_registration_real_anatomy")
deform = import_module("51_deformable_registration")   # tps_fit / tps_apply
probe = import_module("52_probing_the_prior")          # deformation / cone_points / axis_frame

# --- exp 52 에서 그대로 가져오는 설정 ---
PROBE_NOISE = probe.PROBE_NOISE
US_NOISE = probe.US_NOISE                 # 얕은 곳에서의 σ (이제 깊이의 함수가 된다)
US_DEPTH_RANGE = probe.US_DEPTH_RANGE
N_PROBE = probe.N_PROBE
N_TPS_CTRL = probe.N_TPS_CTRL
N_TPS_ANCHOR = probe.N_TPS_ANCHOR
TPS_LAMBDA = probe.TPS_LAMBDA
LANDMARK_NOISE = probe.LANDMARK_NOISE
EXPOSURE_DEG = probe.EXPOSURE_DEG
TARGET_DEPTHS_MM = probe.TARGET_DEPTHS_MM
MISS_TOL = probe.MISS_TOL
DEEP_MM = probe.DEEP_MM

# --- 이번 실험이 추가하는 비이상성 ---
INDENT_MM = (2.0, 5.0)     # 프로브 압박 깊이 범위 [mm] — 술자·부위마다 다르다
INDENT_L_MM = 40.0         # 압박이 조직 안으로 전달되는 감쇠 길이 [mm]
INDENT_L_ERR = 0.30        # 그 길이를 모델로 뺄 때의 상대 오차(30% 틀리게 안다)
DEPTH_SIGMA_MM = 50.0      # σ(d) = σ0·(1 + d/DEPTH_SIGMA) — 감쇠·빔 확산
OUTLIER_FRAC = 0.15        # 특징 오대응 비율
OUTLIER_MM = (8.0, 20.0)   # 오대응이 만드는 가짜 변위 크기
TUKEY_C = 4.0              # 로버스트 임계(정규화 잔차 기준) — 이 밖은 가중치 0

N_SWEEP = (1, 2, 4, 8, 16, 32)


# --------------------------------------------------------------------------- #
# 비이상 초음파 모델
# --------------------------------------------------------------------------- #
def indentation_field(pts, contact, inward, delta_m, L_m=INDENT_L_MM * 1e-3):
    """프로브 압박이 만드는 변위장.

    접촉점에서 최대이고 거리로 지수 감쇠하며, 방향은 **항상 안쪽**이다. 이 '항상 안쪽'이
    핵심 — 여러 번 재도 부호가 바뀌지 않아 평균으로 상쇄되지 않는다(계통 오차)."""
    d = np.linalg.norm(pts - contact, axis=1)
    return delta_m * np.exp(-(d / L_m) ** 2)[:, None] * inward[None, :]


def depth_sigma(pts, window_c, inward, sigma0=US_NOISE):
    """깊이에 따라 커지는 관측 잡음 σ(d) — 감쇠·빔 확산의 1차 근사."""
    d = (pts - window_c) @ inward
    return sigma0 * (1.0 + np.maximum(d, 0.0) / (DEPTH_SIGMA_MM * 1e-3))


def contact_for(pts, window_c, inward):
    """관측점 바로 위 노출면에 프로브를 댄다 — 보고 싶은 곳을 보려면 그 위를 눌러야 한다."""
    rel = pts - window_c
    return window_c + (rel - (rel @ inward)[:, None] * inward[None, :])


# --------------------------------------------------------------------------- #
# 로버스트 TPS (IRLS) — exp 11·15 의 로버스트 커널이 정합에서 다시 나온다
# --------------------------------------------------------------------------- #
def robust_tps_fit(ctrl, disp, lam, sigma=None, mask=None, iters=4, c=TUKEY_C, anneal=100.0):
    """Tukey biweight IRLS + **어닐링**. 가중치를 낮추는 대신 그 점의 λ 를 키운다(부드러운
    스플라인에서 λ_i 는 그 관측을 얼마나 못 믿는지와 같은 역할을 한다).

    두 가지가 다 필요했다.
    - **어닐링**: TPS 는 자유도가 높아서 **이상치를 그냥 보간해버린다.** 그러면 잔차가 이상치
      에서도 작아 IRLS 가 아무것도 못 본다. λ 를 100배에서 시작해 기하적으로 낮춰, 먼저
      뻣뻣하게 맞춰 이상치를 **튀어나오게** 만든 뒤 풀어준다.
    - **재하강(redescending) 가중**: Huber 의 1/r 꼬리는 20 mm 짜리 오대응에도 w≈0.25 밖에
      안 줘서 λ 가 4배로 커질 뿐, 커널(≈0.05) 앞에서는 여전히 보간된다. Tukey 는 임계 밖을
      **0 으로** 보내므로 λ 가 1000배가 되어 실제로 배제된다.
      (처음에 Huber + 어닐링 없이 짰을 때 로버스트 이득이 사실상 0 이었고, 그 두 가지를
       넣고서야 살아났다.)
    - **알려진 σ 기준, 그리고 적용 범위 제한**: 스케일을 MAD 로 데이터에서 뽑았더니 이질적인
      제어점 집합(표면 170 + 앵커 130 + 심부 몇 개)에서 표면점이 스케일을 지배해, 잔차가
      원래 큰 **심부 관측을 통째로 버렸다** — 소수의 유익한 관측을 이상치로 오인한 것이다.
      로버스트 추정은 "이상치는 같은 분포의 소수"를 전제하는데 여기 심부 관측은 애초에 다른
      집단이다. 그래서 잔차를 **알려진 σ_i 로** 정규화하고, 이상치 기전이 실제로 있는
      (mask 로 지정된) 관측에만 적용한다.
    exp 15 의 DCS 가 잔차에 따라 정보행렬을 깎던 것과 같은 계열이다.

    lam 은 제어점별 벡터. exp 51 의 tps_fit 이 K + diag(λ) 를 풀기 때문에 그대로 쓸 수 있다."""
    lam = np.asarray(lam, float)
    n = len(ctrl)
    sig = np.full(n, 1.0) if sigma is None else np.asarray(sigma, float)
    msk = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    sched = np.geomspace(anneal, 1.0, iters + 1)
    model = deform.tps_fit(ctrl, disp, lam=lam * sched[0])
    w = np.ones(n)
    for it in range(iters):
        r = np.linalg.norm(deform.tps_apply(model, ctrl) - (ctrl + disp), axis=1)
        if sigma is None:                              # σ 를 모르면 MAD 로 (마스크 안에서만)
            s = 1.4826 * np.median(r[msk]) + 1e-12
            sig = np.full(n, s)
        u = np.where(msk, r / (c * sig), 0.0)
        w = np.minimum(w, np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0))  # 한 번 버리면 유지
        model = deform.tps_fit(ctrl, disp, lam=lam * sched[it + 1] / np.maximum(w, 1e-3))
    return model


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(seed=5, n_trials=400, quick=False):
    anat.fetch()
    vol, D, origin = anat.load_nrrd()
    surface, mask = anat.head_surface(vol, D, origin)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(surface), min(anat.MODEL_POINTS, len(surface)), replace=False)
    model = surface[sel]
    normals = reg.estimate_normals(model, k=12)

    center = model.mean(0)
    window_c = model[int(np.argmax(model[:, 2]))]
    sag_dir = np.array([0.0, 0.0, -1.0])
    inward = (center - window_c) / np.linalg.norm(center - window_c)
    _, lateral, _ = probe.axis_frame(inward)
    depths = np.array(TARGET_DEPTHS_MM)
    targets = window_c + depths[:, None] * 1e-3 * inward

    print("=== 53. 관측이 대상을 바꾼다: 계통 오차는 평균되지 않는다 ===")
    print(f"exp 52 와 같은 실 인체 MR 표면·심부 모드({DEEP_MM:.0f} mm, 표면 자취 0.03 mm)")
    print(f"추가한 비이상성 3종: 프로브 압박 {INDENT_MM[0]:.0f}~{INDENT_MM[1]:.0f} mm"
          f"(감쇠 {INDENT_L_MM:.0f} mm) · 깊이 잡음 σ(d)=σ₀(1+d/{DEPTH_SIGMA_MM:.0f}mm) · "
          f"오대응 {OUTLIER_FRAC*100:.0f}%")

    d_chk = np.array([20e-3, 45e-3, 70e-3])
    p_chk = window_c + d_chk[:, None] * inward[None, :]
    bias = np.linalg.norm(indentation_field(p_chk, window_c, inward, 3.5e-3), axis=1)
    sg = depth_sigma(p_chk, window_c, inward)
    print("  깊이[mm]      " + " ".join(f"{d*1e3:6.0f}" for d in d_chk))
    print("  압박 편향[mm] " + " ".join(f"{b*1e3:6.2f}" for b in bias) + "   ← 부호가 늘 같다")
    print("  관측 잡음 σ[mm]" + " ".join(f"{s*1e3:5.2f}" for s in sg) + "   ← 깊을수록 커진다")

    # ---- 개두 전 강체 정합 (변형과 무관, 한 번만) ----
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

    v = (model - center) / np.linalg.norm(model - center, axis=1, keepdims=True)
    w_dir = (window_c - center) / np.linalg.norm(window_c - center)
    ang = np.degrees(np.arccos(np.clip(v @ w_dir, -1, 1)))
    in_win = np.where(ang < EXPOSURE_DEG)[0]
    out_win = np.where(ang > EXPOSURE_DEG + 25.0)[0]

    def u_of(p, deep_mm):
        return probe.deformation(p, window_c, sag_dir, inward, lateral, deep_mm)

    def observe(rr, deep_mm, n_us, n_hold=1, press=True, depth_noise=True, outliers=True):
        """한 세션의 관측 일습. 세 비이상성을 개별 스위치로 켠다."""
        idx = rr.choice(in_win, size=min(N_PROBE, len(in_win)), replace=False)
        src = model[idx]
        dst = reg.apply_T(T_place, src + u_of(src, deep_mm)) \
            + rr.normal(0, PROBE_NOISE, (len(idx), 3))
        disp = reg.apply_T(T_rigid, dst) - src
        anchors = model[rr.choice(out_win, size=min(N_TPS_ANCHOR, len(out_win)), replace=False)]

        us = probe.cone_points(rr, n_us, window_c, inward)
        hold = targets[rr.integers(len(targets), size=n_hold)] + rr.normal(0, 6e-3, (n_hold, 3))
        out = {}
        for key, pts in (("us", us), ("hold", hold)):
            if not len(pts):
                out[key], out[key + "_disp"], out[key + "_sig"] = pts, pts.copy(), np.zeros(0)
                continue
            u = u_of(pts, deep_mm)
            if press:                                   # (a) 프로브가 누른다
                delta = rr.uniform(*INDENT_MM, len(pts)) * 1e-3
                cont = contact_for(pts, window_c, inward)
                u = u + np.stack([indentation_field(pts[i:i + 1], cont[i], inward,
                                                    delta[i])[0] for i in range(len(pts))])
            sig = depth_sigma(pts, window_c, inward) if depth_noise \
                else np.full(len(pts), US_NOISE)
            u = u + rr.normal(0, 1.0, (len(pts), 3)) * sig[:, None]
            if outliers:                                # (c) 특징 오대응
                bad = rr.random(len(pts)) < OUTLIER_FRAC
                if bad.any():
                    d = rr.normal(size=(int(bad.sum()), 3))
                    d /= np.linalg.norm(d, axis=1, keepdims=True)
                    u[bad] = d * (rr.uniform(*OUTLIER_MM, int(bad.sum())) * 1e-3)[:, None]
            out[key], out[key + "_disp"], out[key + "_sig"] = pts, u, sig
        return dict(src=src, disp=disp, anchors=anchors,
                    tgt_true=reg.apply_T(T_place, targets + u_of(targets, deep_mm)), **out)

    def fit(case, weight_depth=False, robust=False, deindent=None, sub_seed=7):
        """워프 적합. weight_depth = σ(d) 를 λ_i 로, robust = Tukey IRLS,
        deindent = 압박을 모델로 빼기(추정 감쇠 길이를 넘긴다)."""
        rr = np.random.default_rng(sub_seed)
        sub = rr.choice(len(case["src"]), size=min(N_TPS_CTRL, len(case["src"])), replace=False)
        ctrl = [case["src"][sub], case["anchors"]]
        dd = [case["disp"][sub], np.zeros_like(case["anchors"])]
        lm = [np.full(len(sub), TPS_LAMBDA), np.full(len(case["anchors"]), TPS_LAMBDA)]
        sg = [np.full(len(sub), PROBE_NOISE), np.full(len(case["anchors"]), PROBE_NOISE)]
        mk = [np.zeros(len(sub), bool), np.zeros(len(case["anchors"]), bool)]
        if len(case["us"]):
            u = case["us_disp"].copy()
            if deindent is not None:                    # 압박 보정(추적 프로브 + 힘센서 가정)
                cont = contact_for(case["us"], window_c, inward)
                u = u - np.stack([indentation_field(case["us"][i:i + 1], cont[i], inward,
                                                    np.mean(INDENT_MM) * 1e-3, deindent)[0]
                                  for i in range(len(case["us"]))])
            ctrl.append(case["us"]); dd.append(u)
            s = case["us_sig"] if weight_depth else np.full(len(case["us"]), US_NOISE)
            lm.append(TPS_LAMBDA * (s / US_NOISE) ** 2)
            sg.append(case["us_sig"])                  # 오대응 판정은 실제 σ(d) 기준으로
            mk.append(np.ones(len(case["us"]), bool))  # 이상치 기전이 있는 관측만 로버스트
        C, DD, LM = np.vstack(ctrl), np.vstack(dd), np.concatenate(lm)
        if not robust:
            return deform.tps_fit(C, DD, lam=LM)
        return robust_tps_fit(C, DD, LM, sigma=np.concatenate(sg), mask=np.concatenate(mk))

    def score(case, m, j=None):
        est = reg.apply_T(T_inv, deform.tps_apply(m, targets))
        e = np.linalg.norm(est - case["tgt_true"], axis=1)
        return float(e.mean() if j is None else e[j])

    n_reps = 3 if quick else 6
    n_list = N_SWEEP[:-1] if quick else N_SWEEP

    def sweep(label, **kw):
        row = []
        for n in n_list:
            acc = []
            for rep in range(n_reps):
                c = observe(np.random.default_rng([n, rep, 11]), DEEP_MM, n, **kw["obs"])
                acc.append(score(c, fit(c, **kw["fit"])))
            row.append(float(np.mean(acc)))
        print(f"    {label:<34}" + " ".join(f"{v*1e3:5.2f}" for v in row))
        return row

    # ------------------------------------------------------------------ #
    # A. 프로브 압박 — 계통 오차는 평균되지 않는다
    # ------------------------------------------------------------------ #
    print("-" * 100)
    print(f"[A] 프로브 압박만 켜고 관측 수를 늘려본다 (심부 표적 오차 mm, 시드 {n_reps}개 평균)")
    print("     관측 수                            " + " ".join(f"{n:5d}" for n in n_list))
    off = dict(press=False, depth_noise=False, outliers=False)
    A = {}
    A["ideal"] = sweep("이상 센서 (exp 52)", obs=off, fit={})
    A["press"] = sweep("+ 프로브 압박", obs=dict(off, press=True), fit={})
    A["deind_err"] = sweep(f"  ↳ 압박 보정(감쇠길이 {INDENT_L_ERR*100:.0f}% 오차)",
                           obs=dict(off, press=True),
                           fit=dict(deindent=INDENT_L_MM * 1e-3 * (1 + INDENT_L_ERR)))
    A["deind_ok"] = sweep("  ↳ 압박 보정(감쇠길이 정확)", obs=dict(off, press=True),
                          fit=dict(deindent=INDENT_L_MM * 1e-3))
    gap0 = A["press"][0] - A["ideal"][0]
    gapN = A["press"][-1] - A["ideal"][-1]
    print(f"  → 두 곡선이 **나란히** 간다. 관측 {n_list[0]}개에서 격차 {gap0*1e3:+.2f} mm, "
          f"{n_list[-1]}개에서 {gapN*1e3:+.2f} mm — 관측을 32배로 늘려도 그대로다.")
    print(f"     잡음은 무작위라 √N 으로 줄지만 압박은 **늘 안쪽**이라 줄지 않는다. 그래서 "
          f"압박이 전체 오차에서 차지하는 몫이 "
          f"{gap0/max(A['press'][0],1e-12)*100:.0f}% → {gapN/max(A['press'][-1],1e-12)*100:.0f}% 로 "
          f"커진다 — **데이터를 모을수록 계통 오차가 지배항이 된다.**")
    print("     (exp 42 의 '오차 예산 지배항은 상황마다 뒤집힌다'가 여기서는 표본 수를 따라 뒤집힌다.)")
    print(f"  → 보정하면 회복된다: 감쇠길이를 정확히 알면 {A['deind_ok'][-1]*1e3:.2f} mm(이상 센서와 동일), "
          f"{INDENT_L_ERR*100:.0f}% 틀리게 알면 {A['deind_err'][-1]*1e3:.2f} mm. "
          f"**모델을 알아야 뺄 수 있고, 그 모델도 틀린다.**")

    # ------------------------------------------------------------------ #
    # B. 깊이별 잡음 — 가장 필요한 곳이 가장 못 믿을 곳
    # ------------------------------------------------------------------ #
    print("-" * 100)
    print("[B] 깊이에 따라 커지는 잡음 (압박·오대응은 끔)")
    print("     관측 수                            " + " ".join(f"{n:5d}" for n in n_list))
    B = {}
    B["flat"] = sweep("등방 잡음 σ₀ (exp 52 가정)", obs=off, fit={})
    B["depth"] = sweep("σ(d), 가중 없음", obs=dict(off, depth_noise=True), fit={})
    B["weighted"] = sweep("σ(d), λᵢ∝σᵢ² 가중", obs=dict(off, depth_noise=True),
                          fit=dict(weight_depth=True))
    gain = (B["depth"][-1] - B["weighted"][-1]) / max(B["depth"][-1], 1e-12)
    print(f"  → σ(d) 가 붙으면 {B['flat'][-1]*1e3:.2f} → {B['depth'][-1]*1e3:.2f} mm. "
          f"올바른 가중으로 {B['weighted'][-1]*1e3:.2f} mm ({gain*100:+.0f}%) 회복 — "
          f"{'의미 있지만 작다' if gain < 0.3 else '크다'}.")
    print("     가중은 **정보를 만들지 못한다.** 깊은 관측이 원래 정보가 적다면 그 사실을 "
          "정확히 반영할 뿐이다.")

    # ------------------------------------------------------------------ #
    # C. 오대응 — 로버스트 커널이 정합에서 다시 나온다
    # ------------------------------------------------------------------ #
    print("-" * 100)
    print(f"[C] 특징 오대응 {OUTLIER_FRAC*100:.0f}% (압박·깊이잡음은 끔)")
    print("     관측 수                            " + " ".join(f"{n:5d}" for n in n_list))
    C = {}
    C["clean"] = sweep("오대응 없음", obs=off, fit={})
    C["ls"] = sweep("오대응 + 최소제곱", obs=dict(off, outliers=True), fit={})
    C["robust"] = sweep("오대응 + Tukey IRLS", obs=dict(off, outliers=True),
                        fit=dict(robust=True))
    print(f"  → 오대응 {OUTLIER_FRAC*100:.0f}% 가 최소제곱을 {C['clean'][-1]*1e3:.2f} → "
          f"{C['ls'][-1]*1e3:.2f} mm 로 망가뜨리고, Tukey IRLS 가 {C['robust'][-1]*1e3:.2f} mm 로 "
          f"되돌린다. exp 11·15 의 로버스트 커널이 정합에서 그대로 재사용된다.")
    lose = [n for n, r, l in zip(n_list, C["robust"], C["ls"]) if r > l]
    if lose:
        print(f"     단 관측이 적을 때는 손해다({', '.join(str(n) for n in lose)}개). "
              f"이상치 하나를 버리는 대가가 표본이 적을수록 크다 — **로버스트도 공짜가 아니다.**")

    # ------------------------------------------------------------------ #
    # D. 셋을 다 넣으면 — exp 52 의 게이트는 얼마가 되는가
    # ------------------------------------------------------------------ #
    print("-" * 100)
    trials = max(n_trials // 2, 120) if quick else n_trials
    n_use, K_CHECK = 4, (1, 2, 3, 5)
    print(f"[D] 현실 센서에서 다시 잰 게이트 — 환자 {trials}명. **검산 관측도 같은 센서로 잰다** "
          f"(exp 52 는 검산까지 이상 센서였다)")
    real_obs = dict(press=True, depth_noise=True, outliers=True)
    real_fix = dict(weight_depth=True, robust=True,
                    deindent=INDENT_L_MM * 1e-3 * (1 + INDENT_L_ERR))
    # 시드 salt 는 **고정 정수**로 준다. hash("ideal") 같은 문자열 해시는 파이썬이 프로세스마다
    # 무작위화해서(PYTHONHASHSEED) 실행할 때마다 결과가 달라진다 — 처음에 그렇게 짰다가
    # 같은 코드가 매 실행 다른 AUROC 를 뱉는 걸 보고 잡았다.
    SENSORS = ((0, "ideal", "이상 (exp 52)", off, {}),
               (1, "real", "현실 (보정 없음)", real_obs, {}),
               (2, "real_fix", "현실 + 세 대책", real_obs, real_fix))
    recs = {tag: [] for _, tag, *_ in SENSORS}
    for i in range(trials):
        rr = np.random.default_rng([seed, i])
        has_deep = bool(i % 2)
        deep = float(abs(rr.normal(0.0, 0.4))) if not has_deep else float(rr.uniform(4.0, 9.0))
        j = int(rr.integers(len(targets)))
        for h, tag, _, obs_kw, fit_kw in SENSORS:
            c = observe(np.random.default_rng([seed, i, 700 + h]), deep, 0,
                        n_hold=max(K_CHECK), **obs_kw)
            m = fit(c, **fit_kw)
            pred = deform.tps_apply(m, c["hold"]) - c["hold"]
            meas = c["hold_disp"].copy()
            if fit_kw.get("deindent"):                  # 검산 관측에도 같은 보정을 적용
                cont = contact_for(c["hold"], window_c, inward)
                meas = meas - np.stack([
                    indentation_field(c["hold"][t:t + 1], cont[t], inward,
                                      np.mean(INDENT_MM) * 1e-3, fit_kw["deindent"])[0]
                    for t in range(len(c["hold"]))])
            r = np.linalg.norm(pred - meas, axis=1)
            # 대책을 쓰면 **절사평균**(최댓값 하나를 버리고 평균) — 중앙값은 이상치를 막지만
            # 점을 늘려도 평균 효과가 없어서 검산 수를 늘린 보람이 없었다.
            gates = {k: float(np.mean(np.sort(r[:k])[:-1]) if fit_kw.get("robust") and k >= 3
                              else (np.median(r[:k]) if fit_kw.get("robust")
                                    else np.sqrt(np.mean(r[:k] ** 2)))) for k in K_CHECK}
            cf = observe(np.random.default_rng([seed, i, 4, 700 + h]), deep, n_use, **obs_kw)
            recs[tag].append(dict(has_deep=has_deep, prior=score(c, m, j),
                                  fixed=score(cf, fit(cf, **fit_kw), j), **
                                  {f"g{k}": gates[k] for k in K_CHECK}))

    def auroc(G, bad):
        pos, neg = G[bad], G[~bad]
        if not len(pos) or not len(neg):
            return float("nan")
        return float(np.greater.outer(pos, neg).mean() + 0.5 * np.equal.outer(pos, neg).mean())

    print("     센서                   " + "  ".join(f"검산{k}개" for k in K_CHECK)
          + "   관측 4개 교정 후")
    summary = {}
    for _, tag, name, _, _ in SENSORS:
        Ep = np.array([x["prior"] for x in recs[tag]])
        Ef = np.array([x["fixed"] for x in recs[tag]])
        bad = Ep > MISS_TOL
        au = {k: auroc(np.array([x[f"g{k}"] for x in recs[tag]]), bad) for k in K_CHECK}
        summary[tag] = dict(auroc=au, fixed=float(np.median(Ef)),
                            unsafe_fixed=float(np.mean(Ef > MISS_TOL)))
        print(f"     {name:<20}" + "  ".join(f"{au[k]:7.2f}" for k in K_CHECK)
              + f"   {np.median(Ef)*1e3:9.2f} mm")
    print(f"  → exp 52 가 **검산까지 이상 센서로** 보고한 값(검산 1개 "
          f"{summary['ideal']['auroc'][1]:.2f})은 현실 센서에서 "
          f"{summary['real']['auroc'][1]:.2f} 로 무너진다. 검산 관측 하나가 오대응이면 게이트가 "
          f"그냥 울리기 때문이다 — **검산도 센서로 하는 일이다.**")
    k_need = next((k for k in K_CHECK
                   if summary["real_fix"]["auroc"][k] >= summary["ideal"]["auroc"][1]), None)
    print(f"     대책(압박 보정 + 절사평균 통계)을 검산에도 적용해도 검산 "
          + " → ".join(f"{k}개 {summary['real_fix']['auroc'][k]:.2f}" for k in K_CHECK)
          + " — **거의 나아지지 않는다.**")
    sg_t = depth_sigma(targets, window_c, inward)
    sig_t = np.linalg.norm(u_of(targets, DEEP_MM) - u_of(targets, 0.0), axis=1)
    print("     이유는 신호 대 잡음이다. 표적 깊이별 신호(심부 모드) vs 검산 1점 잡음(σ(d)·√3): "
          + " · ".join(f"{d:.0f}mm {s*1e3:.1f}/{n*1e3*np.sqrt(3):.1f}"
                       for d, s, n in zip(depths, sig_t, sg_t)) + " mm")
    print("     **검산은 편향이 아니라 잡음에 묶여 있다.** 그래서 압박 보정·로버스트(편향과 "
          "이상치를 겨냥한 대책)는 교정은 살리지만 검산은 못 살린다 —")
    print(f"     교정은 {summary['real']['fixed']*1e3:.2f} → "
          f"{summary['real_fix']['fixed']*1e3:.2f} mm 로 회복되는데 게이트는 제자리다. "
          f"**대책은 자기가 겨냥한 오차원만 고친다.**")
    if k_need:
        print(f"     그나마 검산 관측 {k_need}개면 이상 센서의 1개 수준을 되찾는다.")
    else:
        print(f"     검산 {max(K_CHECK)}개로도 이상 센서의 1개 수준"
              f"({summary['ideal']['auroc'][1]:.2f})에 못 미친다. 게이트의 천장을 정하는 것은 "
              f"통계량이 아니라 **모달리티**이고, exp 52 의 이상 센서 가정이 그 비용을 통째로 "
              f"숨기고 있었다.")
    print(f"  → 교정 쪽도 같다: 관측 4개가 이상 센서에서 "
          f"{summary['ideal']['fixed']*1e3:.2f} mm 를 만드는 자리에서 현실 센서는 "
          f"{summary['real']['fixed']*1e3:.2f} mm, 대책을 다 쓰면 "
          f"{summary['real_fix']['fixed']*1e3:.2f} mm.")

    # ---- 그림 ----
    fig, axg = plt.subplots(2, 3, figsize=(16.8, 9.2))
    axes = axg.ravel()

    ax = axes[0]
    dd = np.linspace(0, 90, 120) * 1e-3
    pp = window_c + dd[:, None] * inward[None, :]
    for delta, ls in ((2e-3, ":"), (3.5e-3, "-"), (5e-3, "--")):
        ax.plot(dd * 1e3, np.linalg.norm(
            indentation_field(pp, window_c, inward, delta), axis=1) * 1e3,
            ls, color="crimson", label=f"indentation {delta*1e3:.1f} mm")
    ax.plot(dd * 1e3, depth_sigma(pp, window_c, inward) * 1e3, "-", color="royalblue",
            label="observation noise σ(d)")
    for d in TARGET_DEPTHS_MM:
        ax.axvline(d, color="0.85", lw=1, zorder=0)
    ax.set_xlabel("depth below craniotomy [mm]"); ax.set_ylabel("[mm]")
    ax.set_title("Systematic bias vs random noise, by depth", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    for k, c, lbl in (("ideal", "seagreen", "ideal sensor"),
                      ("press", "crimson", "+ probe indentation"),
                      ("deind_err", "darkorange", "  de-indented (30% model error)"),
                      ("deind_ok", "0.45", "  de-indented (exact model)")):
        ax.plot(n_list, np.array(A[k]) * 1e3, "-o", color=c, label=lbl)
    ax.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xticks(list(n_list))
    ax.set_xticklabels([str(n) for n in n_list])
    ax.set_xlabel("depth observations used"); ax.set_ylabel("deep-target error [mm]")
    ax.set_title("More measurements do not average out a bias", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[2]
    for k, c, lbl in (("flat", "seagreen", "isotropic σ₀"),
                      ("depth", "crimson", "σ(d), unweighted"),
                      ("weighted", "royalblue", "σ(d), λᵢ ∝ σᵢ²")):
        ax.plot(n_list, np.array(B[k]) * 1e3, "-o", color=c, label=lbl)
    ax.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xticks(list(n_list))
    ax.set_xticklabels([str(n) for n in n_list])
    ax.set_xlabel("depth observations used"); ax.set_ylabel("deep-target error [mm]")
    ax.set_title("Weighting reports information, it does not create it", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[3]
    for k, c, lbl in (("clean", "seagreen", "no outliers"),
                      ("ls", "crimson", f"{OUTLIER_FRAC*100:.0f}% outliers, least squares"),
                      ("robust", "royalblue", f"{OUTLIER_FRAC*100:.0f}% outliers, Tukey IRLS")):
        ax.plot(n_list, np.array(C[k]) * 1e3, "-o", color=c, label=lbl)
    ax.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xticks(list(n_list))
    ax.set_xticklabels([str(n) for n in n_list])
    ax.set_xlabel("depth observations used"); ax.set_ylabel("deep-target error [mm]")
    ax.set_title("Robust kernels, again (cf. #11, #15)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[4]
    tags = ("ideal", "real", "real_fix")
    names = ["ideal\n(exp 52)", "realistic\n(uncorrected)", "realistic\n+ 3 remedies"]
    for t, c, lbl in zip(tags, ("seagreen", "crimson", "royalblue"),
                         ("ideal sensor", "realistic, uncorrected", "realistic + remedies")):
        ax.plot(K_CHECK, [summary[t]["auroc"][k] for k in K_CHECK], "-o", color=c, label=lbl)
    ax.axhline(0.52, color="0.3", ls="--", lw=1, label="surface gate (chance)")
    ax.set_xticks(list(K_CHECK)); ax.set_xlabel("check observations held out")
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("violation AUROC")
    ax.set_title("The check is made with the same sensor", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[5]
    x = np.arange(3)
    ax.bar(x, [summary[t]["fixed"] * 1e3 for t in tags], 0.5,
           color=("seagreen", "crimson", "royalblue"), alpha=0.75)
    ax.axhline(MISS_TOL * 1e3, color="0.3", ls="--", lw=1, label="tolerance")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("target error [mm]")
    ax.set_title(f"And to the correction ({n_use} observations)", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)

    fig.suptitle("53. When measuring changes what you measure — probe pressure, "
                 "depth-dependent noise, mismatched features", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "53_measurement_changes_it.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/53_measurement_changes_it.png, assets/53_measurement_changes_it.png")

    return dict(A=A, B=B, C=C, n_list=list(n_list), summary=summary, recs=recs,
                bias_mm=(bias * 1e3).tolist(), sigma_mm=(sg * 1e3).tolist(),
                MISS_TOL=MISS_TOL)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 압박 변위장을 **가우시안 하나**로 모델링했다. 실제 조직은 비선형·점탄성이라 누른 뒤
#     시간에 따라 이완하고(creep), 뇌척수액이 빠지면 응답 자체가 달라진다. 여기서 보이는 것은
#     "계통 오차는 평균되지 않는다"는 성질이지 실제 압박량 예측이 아니다.
#   - 압박 보정에서 **접촉 위치와 압박 깊이를 안다**고 가정했다(추적 프로브 + 힘센서). 감쇠
#     길이만 틀리게 준 것이라 실제보다 낙관적이다.
#   - σ(d) 선형 증가는 1차 근사다. 실제 iUS 는 방향 이방성(축방향 ≫ 횡방향 분해능)이 크고,
#     여기서는 등방으로 뒀다 — 이 부분은 낙관적이다.
#   - 오대응을 **균일 무작위 방향**으로 만들었다. 실제 오대응은 비슷하게 생긴 구조에 붙어서
#     구조적으로 편향되며, 그런 이상치는 로버스트 커널이 더 못 잡는다.
#   - 변형장·대응은 exp 51·52 와 같이 합성이다. 방법 간 상대 비교로 읽어야 한다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
