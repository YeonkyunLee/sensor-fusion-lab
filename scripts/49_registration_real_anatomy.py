"""실 인체 MR 스캔에서의 표면 정합 — "어디를 찍어야 하는가".

exp 41·42 의 정합·안전게이트는 합성 해부(매끄러운 타원체 캡) 위에서 만들었고, exp 44 는
그것을 **실측 점군**(Stanford Bunny)으로 옮겨 검증했다. 남은 간극은 데이터의 종류다:
Bunny 는 실측이지만 **해부 구조가 아니다**. 이 실험은 마지막 앵커로 **실제 사람의 MR 스캔**을
쓴다(공개 데이터: 3D Slicer 샘플 `MRHead`, 256×256×130).

시나리오는 신경외과 내비게이션의 표면 정합이다. 수술 전 영상에서 머리 표면을 얻고, 수술 중
프로브로 두피·얼굴 일부를 찍어 두 좌표계를 맞춘 뒤, 영상에서 계획한 **심부 표적**으로 도구를
보낸다. 여기서 임상적으로 실제 문제가 되는 질문이 하나 있다.

    **표면 어디를 찍어야 정합이 잘 되는가?**

머리 표면 대부분은 매끄러운 돔(두피)이다. 구(球)의 일부만 찍으면 그 조각은 구면을 따라
**미끄러질 수 있다** — 데이터가 회전을 구속하지 못한다(관측성 문제). 반대로 코·눈두덩·귀처럼
곡률이 특징적인 영역이 들어오면 그 대칭이 깨진다. 이 실험은 그것을 **실제 해부 기하에서
정량화**한다: 표면 조각의 기하적 특징량(surface variation)과 정합 오차(TRE)의 관계.

무엇을 재는가
  1) 조각별 TRE — 어느 영역을 찍었는지에 따라 얼마나 달라지는가
  2) 그 차이를 **정합 자신이 예측할 수 있는가** (exp 42 의 σ 게이트가 실제 해부에서도 되는가)
  3) 커버리지 효과 — 여러 영역을 나눠 찍으면 얼마나 좋아지는가

의존성은 numpy/scipy 뿐이다. nrrd 는 텍스트 헤더 + (gzip) raw 바이너리라 의료영상
라이브러리 없이 직접 읽는다. 데이터는 재배포하지 않고 실행 시 받아 `data_cache/`(git 제외)에 둔다.

    python scripts/49_registration_real_anatomy.py
"""

from __future__ import annotations

import gzip
import sys
import urllib.request
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sensor_fusion.se3 import hat, so3_exp  # noqa: E402

reg = import_module("41_surgical_registration")     # 점-대-평면 ICP 자산
real = import_module("44_registration_real_scans")  # 신뢰도 지표(σ·불일치·겹침) 자산

# 공개 데이터: 3D Slicer 샘플 MRHead (실제 인체 MR)
MRHEAD_URL = ("https://github.com/Slicer/SlicerTestingData/releases/download/MD5/"
              "39b01631b7b38232a220007230624c8e")
CACHE = Path("data_cache")
MRHEAD = CACHE / "MRHead.nrrd"

TISSUE_THR = 40          # 조직/배경 분리 임계(이 볼륨 강도 0~279)
MODEL_POINTS = 12000     # 모델 표면 점군 서브샘플
N_PROBE = 600            # 수술 중 프로브 점 수(한 영역)
PROBE_NOISE = 1.0e-3     # 프로브 잡음 σ [m] (1 mm — 광학 트래커 규모)
N_OUTLIERS = 8
PATCH_FRAC = 0.06        # 한 영역이 표면에서 차지하는 비율
# 표적 허용오차. 신경외과 내비게이션에서 흔히 요구되는 정확도 수준(≈2 mm)을 기준으로 둔다.
MISS_TOL = 2e-3
K_SIGMA = 3.0
DISAGREE_TOL = 3e-3
INLIER_MIN = 0.75


# --------------------------------------------------------------------------- #
# nrrd 읽기 (의료영상 라이브러리 없이)
# --------------------------------------------------------------------------- #
_DTYPES = {"short": "<i2", "unsigned short": "<u2", "int": "<i4", "uint": "<u4",
           "float": "<f4", "double": "<f8", "unsigned char": "u1", "signed char": "i1",
           "int16": "<i2", "uint16": "<u2", "int32": "<i4"}


def fetch(url=MRHEAD_URL, dest=MRHEAD):
    if dest.exists():
        return dest
    dest.parent.mkdir(exist_ok=True)
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001
        pass
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def load_nrrd(path=MRHEAD):
    """(volume, 인덱스→월드 방향벡터(3,3) [m], 원점 [m]) 반환.

    nrrd 는 `key: value` 텍스트 헤더 뒤 빈 줄, 그 다음이 데이터다. `space directions`
    각 벡터가 축 하나의 인덱스 증가에 대응하는 월드 변위이므로, 축 순서가 뒤바뀐
    볼륨(이 데이터가 그렇다)도 그 행렬만 제대로 쓰면 좌표가 맞는다."""
    raw = Path(path).read_bytes()
    sep = raw.find(b"\n\n")
    meta = {}
    for line in raw[:sep].decode("utf-8", "replace").splitlines():
        if ":" in line and not line.startswith("#"):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().lstrip("=").strip()

    sizes = [int(s) for s in meta["sizes"].split()]
    data = raw[sep + 2:]
    if meta.get("encoding", "raw") == "gzip":
        data = gzip.decompress(data)
    vol = np.frombuffer(data, dtype=_DTYPES[meta["type"]]).reshape(sizes[::-1])

    dirs = []
    for tok in meta["space directions"].replace(" ", "").split(")"):
        tok = tok.strip("(,")
        if tok:
            dirs.append([float(x) for x in tok.split(",")])
    D = np.array(dirs) * 1e-3                       # mm → m
    origin = np.zeros(3)
    if "space origin" in meta:
        origin = np.array([float(x) for x in
                           meta["space origin"].strip("()").split(",")]) * 1e-3
    return vol, D, origin


# --------------------------------------------------------------------------- #
# 표면 점군 추출 (임계 → 최대 연결성분 → 구멍 채움 → 경계 복셀)
# --------------------------------------------------------------------------- #
def head_surface(vol, D, origin, thr=TISSUE_THR):
    """머리 **외부 표면** 점군을 월드좌표(m)로. 내부 구조는 제외한다."""
    mask = vol > thr
    mask = ndimage.binary_closing(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n > 1:                                        # 가장 큰 연결성분만 = 머리
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = lab == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_fill_holes(mask)
    surf = mask & ~ndimage.binary_erosion(mask)      # 경계 복셀
    k2, k1, k0 = np.nonzero(surf)                    # vol[k2,k1,k0] ↔ 축 인덱스 (k0,k1,k2)
    idx = np.stack([k0, k1, k2], axis=1)
    return idx @ D + origin, mask


def surface_variation(pts, k=14):
    """국소 PCA 의 최소고윳값 비율 λ0/(λ0+λ1+λ2) — 평면이면 0, 특징적이면 커진다."""
    tree = cKDTree(pts)
    _, nb = tree.query(pts, k=k)
    c = pts[nb] - pts[nb].mean(axis=1, keepdims=True)
    C = np.einsum("nki,nkj->nij", c, c) / k
    ev = np.linalg.eigvalsh(C)                       # 오름차순
    return ev[:, 0] / (ev.sum(axis=1) + 1e-18)


# --------------------------------------------------------------------------- #
# 프로빙 · 정합
# --------------------------------------------------------------------------- #
def make_patch_probe(rng, surface, center_idx, T_true, n=N_PROBE,
                     frac=PATCH_FRAC, noise=PROBE_NOISE, n_out=N_OUTLIERS):
    """표면의 한 국소 영역만 찍은 프로브 점군(미지의 SE(3) 적용)."""
    d = np.linalg.norm(surface - surface[center_idx], axis=1)
    keep = np.argsort(d)[:max(int(frac * len(surface)), n)]
    sel = rng.choice(keep, size=min(n, len(keep)), replace=False)
    P = surface[sel] + rng.normal(0, noise, (len(sel), 3))
    if n_out:
        span = P.max(0) - P.min(0)
        P = np.vstack([P, P.mean(0) + rng.uniform(-1, 1, (n_out, 3)) * span * 0.7])
    return reg.apply_T(T_true, P), sel


def deep_targets(surface, k=4, seed=0):
    """심부 표적(종양 대용) — 표면 무게중심 쪽으로 당긴 점들. 정합에 쓰지 않는다."""
    rng = np.random.default_rng(seed)
    c = surface.mean(0)
    sel = rng.choice(len(surface), k, replace=False)
    return c + 0.35 * (surface[sel] - c)


def procrustes(src, dst):
    """대응하는 점쌍에서 강체변환(SVD, Kabsch/Horn) — 랜드마크 조대정렬용."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:                      # 반사 제거
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return reg.pose_T(R, cd - R @ cs)


def pick_landmarks(surface, k=4, seed=0):
    """멀리 퍼진 표면점 k개 = 임상의 해부학적 랜드마크(비근점·이주점 등) 대용.

    farthest-point 샘플링으로 서로 최대한 떨어지게 고른다 — 랜드마크가 한쪽에 몰리면
    조대정렬의 회전이 부실해진다는 임상 상식과 같은 이유다."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(surface)))]
    d = np.linalg.norm(surface - surface[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(surface - surface[idx[-1]], axis=1))
    return np.array(idx)


def landmark_init(rng, surface, land_idx, T_true, noise=2.5e-3):
    """랜드마크 조대정렬: 술자가 찍은(=잡음 섞인) 랜드마크로 첫 변환을 만든다.

    반환값은 ICP 초기값(프로브좌표 → 영상좌표). 랜드마크 식별 오차(2~3 mm)가 임상에서
    이 단계의 지배 오차이며, 그래서 표면 정합으로 다시 다듬는다."""
    model_pts = surface[land_idx]                            # 영상좌표(계획에서 지정)
    digitized = reg.apply_T(T_true, model_pts) + rng.normal(0, noise, (len(land_idx), 3))
    return procrustes(digitized, model_pts)                  # probe → image


def register_patch(probe, model, normals, tree, verify_target, init,
                   n_verify=3, rng=None):
    """랜드마크 초기값에서 출발하는 2단계 ICP + 세 신뢰도 신호.

    exp 44 의 자산(`_icp2`, `information_matrix`, `target_sigma`)을 그대로 쓰되,
    조대정렬만 임상 워크플로(랜드마크)로 바꾼다. 다중초기값 검증도 여기서는 '180° 뒤집기'가
    아니라 **랜드마크 조대정렬의 흔들림**(술자가 조금 다르게 찍었을 때)으로 준다."""
    rng = np.random.default_rng(0) if rng is None else rng
    T_reg = real._icp2(probe, model, normals, init)
    A, fre, inlier = real.information_matrix(T_reg, probe, tree, model, normals)
    if A is None or not np.isfinite(fre):
        return T_reg, np.inf, np.inf, np.inf, inlier
    sigma = max(fre, PROBE_NOISE)
    try:
        Cov = sigma ** 2 * np.linalg.inv(A + 1e-12 * np.eye(6))
        sig_t = real.target_sigma(Cov, verify_target)
    except np.linalg.LinAlgError:
        sig_t = np.inf

    base = reg.apply_T(T_reg, verify_target[None, :])[0]
    disagree = 0.0
    for _ in range(n_verify):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        jitter = reg.pose_T(so3_exp(ax * np.deg2rad(rng.uniform(-8, 8))),
                            rng.uniform(-5e-3, 5e-3, 3))
        T_r = real._icp2(probe, model, normals, jitter @ init)
        _, fre_r, _ = real.information_matrix(T_r, probe, tree, model, normals)
        if not np.isfinite(fre_r) or fre_r > 1.3 * max(fre, PROBE_NOISE):
            continue
        p_r = reg.apply_T(T_r, verify_target[None, :])[0]
        disagree = max(disagree, float(np.linalg.norm(p_r - base)))
    return T_reg, fre, sig_t, disagree, inlier


def tre_of(T_reg, T_true, targets):
    comp = T_reg @ T_true
    return float(np.mean(np.linalg.norm(reg.apply_T(comp, targets) - targets, axis=1)))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def verification_point(rng, surface, land_idx, probe_center, T_true, noise=1.0e-3):
    """임상의 **검증점**: 정합에 쓰지 않은 점을 하나 더 찍어 정합을 검산한다.

    실제 내비게이션에서 술자가 하는 절차 그대로다. 랜드마크·프로브 영역에서 멀리 떨어진
    점을 골라, 정합 후 예측 위치와 실제 찍은 위치의 차이를 본다.
    반환 (모델좌표 점, 트래커좌표에서 찍은 점)."""
    d_land = np.min(np.linalg.norm(surface[:, None] - surface[land_idx][None], axis=2),
                    axis=1)
    d_probe = np.linalg.norm(surface - surface[probe_center], axis=1)
    score = np.minimum(d_land, d_probe)              # 둘 다에서 먼 점
    idx = int(np.argmax(score))
    p_model = surface[idx]
    p_digit = reg.apply_T(T_true, p_model[None, :])[0] + rng.normal(0, noise, 3)
    return p_model, p_digit


def main(n_region=12, n_gate=20, seed=5):
    fetch()
    vol, D, origin = load_nrrd()
    surface, mask = head_surface(vol, D, origin)

    rng = np.random.default_rng(0)
    sel = rng.choice(len(surface), min(MODEL_POINTS, len(surface)), replace=False)
    model = surface[sel]
    normals = reg.estimate_normals(model, k=12)
    tree = cKDTree(model)
    var = surface_variation(model)
    targets = deep_targets(model)

    extent = (model.max(0) - model.min(0)) * 1e3
    print("=== 49. 실 인체 MR 스캔 표면 정합 — 어디를 찍어야 하는가 ===")
    print(f"데이터: 3D Slicer 공개 샘플 MRHead (실제 인체 MR), 볼륨 {vol.shape}, "
          f"복셀 {np.abs(D).sum(0)*1e3} mm")
    print(f"표면 점군 {len(surface)}점 → 모델 {len(model)}점, 머리 크기 "
          f"{extent[0]:.0f}×{extent[1]:.0f}×{extent[2]:.0f} mm")
    print(f"프로브: 한 영역당 {N_PROBE}점(표면의 {PATCH_FRAC:.0%}), 잡음 "
          f"{PROBE_NOISE*1e3:.1f} mm, outlier {N_OUTLIERS}")

    land_idx = pick_landmarks(surface, k=4)
    print(f"랜드마크 4점(farthest-point): 영상좌표에서 서로 "
          f"{np.linalg.norm(surface[land_idx][:, None] - surface[land_idx][None], axis=2).max()*1e3:.0f} mm "
          "까지 퍼져 있음 — 임상의 해부 랜드마크 대용")

    # ---- 조대정렬 없이? 먼저 실패를 확인한다 (baseline) ----
    r0 = np.random.default_rng([seed, 0])
    ax0 = r0.normal(size=3)
    ax0 /= np.linalg.norm(ax0)
    T0 = reg.pose_T(so3_exp(ax0 * np.deg2rad(6.0)), np.array([0.01, -0.008, 0.012]))
    probe0, _ = make_patch_probe(r0, surface, int(r0.integers(len(surface))), T0)
    T_centroid = real._icp2(probe0, model, normals, reg.centroid_init(probe0, model))
    tre_centroid = tre_of(T_centroid, T0, targets)
    T_land = real._icp2(probe0, model, normals,
                        landmark_init(r0, surface, land_idx, T0))
    tre_land = tre_of(T_land, T0, targets)
    print("-" * 78)
    print("[조대정렬이 필수다] 같은 프로브 조각, 초기값만 바꿔서")
    print(f"  무게중심 정렬(exp 44 방식) : TRE {tre_centroid*1e3:8.2f} mm  ← 표면의 "
          f"{PATCH_FRAC:.0%} 조각과 머리 전체의 무게중심은 애초에 맞지 않는다")
    print(f"  랜드마크 4점 조대정렬      : TRE {tre_land*1e3:8.2f} mm "
          f"({tre_centroid/max(tre_land,1e-12):.0f}배)  ← 임상 워크플로(랜드마크→표면 정밀화)")

    def trial(i, hard, tag_seed):
        """한 번의 정합 시행. hard=True 면 적게·좁게 찍고 랜드마크도 더 부정확하다."""
        r = np.random.default_rng([tag_seed, i])
        center = int(r.integers(len(surface)))
        frac = PATCH_FRAC * (0.4 if hard else 1.0)
        n_pts = 250 if hard else N_PROBE
        land_noise = 4.0e-3 if hard else 2.5e-3
        axis = r.normal(size=3)
        axis /= np.linalg.norm(axis)
        T_true = reg.pose_T(so3_exp(axis * np.deg2rad(r.uniform(-8, 8))),
                            r.uniform(-0.02, 0.02, 3))          # 수술대 배치 오차
        probe, _ = make_patch_probe(r, surface, center, T_true, n=n_pts, frac=frac)
        init = landmark_init(r, surface, land_idx, T_true, noise=land_noise)
        T_reg, fre, sig, dis, inlier = register_patch(probe, model, normals, tree,
                                                      targets[0], init, rng=r)
        # ④ 검증점: 정합에 쓰지 않은 점으로 검산 (임상 절차)
        v_model, v_digit = verification_point(r, surface, land_idx, center, T_true)
        v_err = float(np.linalg.norm(reg.apply_T(T_reg, v_digit[None, :])[0] - v_model))
        dm = np.linalg.norm(model - surface[center], axis=1)
        near = dm < np.quantile(dm, frac)
        return dict(sv=float(np.median(var[near])), tre=tre_of(T_reg, T_true, targets),
                    fre=fre, sigma=sig, disagree=dis, inlier=inlier, verify=v_err,
                    hard=hard, center=surface[center], probe=probe)

    # ---- A) 어디를 찍는가 — 조건을 고정하고 영역만 바꾼다 ----
    print("-" * 78)
    print(f"[A. 어디를 찍는가] 조건 고정(표준: {N_PROBE}점, 표면 {PATCH_FRAC:.0%}), "
          f"{n_region}곳의 서로 다른 영역")
    rowsA = [trial(i, hard=False, tag_seed=seed) for i in range(n_region)]
    for i, rw in enumerate(rowsA):
        print(f"  영역 {i+1:2d}: 특징량 {rw['sv']:.4f} | FRE {rw['fre']*1e3:5.2f} | "
              f"TRE {rw['tre']*1e3:6.2f} mm | σ {rw['sigma']*1e3:5.2f} | "
              f"겹침 {rw['inlier']*100:3.0f}%")
    svA = np.array([r["sv"] for r in rowsA])
    treA = np.array([r["tre"] for r in rowsA])
    corr = float(np.corrcoef(svA, np.log10(np.maximum(treA, 1e-6)))[0, 1])
    lo, hi = svA < np.median(svA), svA >= np.median(svA)
    print(f"  → 매끄러운 영역(특징량 하위 절반) TRE 중앙값 {np.median(treA[lo])*1e3:.2f} mm "
          f"vs 특징적인 영역 {np.median(treA[hi])*1e3:.2f} mm "
          f"({np.median(treA[lo])/max(np.median(treA[hi]),1e-12):.1f}배), "
          f"특징량↔log10(TRE) 상관 {corr:+.2f}")
    print("     구(球)에 가까운 조각은 표면을 따라 미끄러진다 — 회전이 데이터로 구속되지 "
          "않는다. 임상에서 코·눈두덩·귀를 함께 찍으라는 지침의 기하적 이유다.")

    # ---- B) 신뢰도 게이트: 난이도를 섞어 실패를 만들고, 네 신호를 비교 ----
    print("-" * 78)
    print(f"[B. 게이트 전이] {n_gate}회(절반은 어려운 조건), unsafe = TRE > "
          f"{MISS_TOL*1e3:.0f} mm (임상 요구 수준)")
    rows = [trial(i, hard=bool(i % 2), tag_seed=seed + 100) for i in range(n_gate)]
    tre_a = np.array([r["tre"] for r in rows])
    sig_a = np.array([r["sigma"] for r in rows])
    dis_a = np.array([r["disagree"] for r in rows])
    inl_a = np.array([r["inlier"] for r in rows])
    ver_a = np.array([r["verify"] for r in rows])
    hard_a = np.array([r["hard"] for r in rows])
    unsafe = tre_a > MISS_TOL
    print(f"  표준 조건 TRE 중앙값 {np.median(tre_a[~hard_a])*1e3:.2f} mm / "
          f"어려운 조건 {np.median(tre_a[hard_a])*1e3:.2f} mm")

    gate_s = K_SIGMA * sig_a > MISS_TOL
    gate_d = dis_a > DISAGREE_TOL
    gate_i = inl_a < INLIER_MIN
    gate_v = ver_a > MISS_TOL                    # ④ 검증점 잔차
    gate_all = gate_s | gate_d | gate_i | gate_v

    def summarize(tag, gate):
        det = 100 * np.sum(unsafe & gate) / max(unsafe.sum(), 1)
        fa = 100 * np.sum(~unsafe & gate) / max((~unsafe).sum(), 1)
        ex = ~gate
        exu = 100 * np.sum(unsafe & ex) / max(ex.sum(), 1)
        print(f"  {tag:28s} 검출 {det:5.1f}% | 오경보 {fa:5.1f}% | 집행분 unsafe "
              f"{exu:5.1f}% ({int(np.sum(unsafe & ex))}/{int(ex.sum())})")
        return det, fa, exu

    print(f"  naive(항상 집행)             unsafe {100*unsafe.mean():.1f}% "
          f"({int(unsafe.sum())}/{n_gate})")
    det_s, fa_s, ex_s = summarize("① k·σ (조건화)", gate_s)
    det_d, fa_d, ex_d = summarize("② 다중초기값 일치성", gate_d)
    det_i, fa_i, ex_i = summarize("③ 겹침 비율", gate_i)
    det_v, fa_v, ex_v = summarize("④ 검증점 잔차(임상 절차)", gate_v)
    det_a, fa_a, ex_a = summarize("전부 결합", gate_all)
    corr_v = float(np.corrcoef(ver_a, tre_a)[0, 1])
    print(f"  검증점 잔차 ↔ TRE 상관 {corr_v:+.2f} — 표면에서 잰 값이 심부 표적오차를 "
          "얼마나 대변하는지. 상관이 높아도 표면 검산은 심부 오차를 **과소평가**한다"
          "(표적이 표면에서 멀수록 지레 효과로 커진다).")
    sv_a, dis_a_out = np.array([r["sv"] for r in rows]), dis_a

    # ---- 커버리지: 한 영역 vs 여러 영역 나눠 찍기 ----
    print("-" * 78)
    print("[커버리지] 같은 점 수를 한 곳에 몰아 찍기 vs 여러 곳에 나눠 찍기")
    cov = {}
    for n_reg in (1, 2, 4):
        vals = []
        for t in range(4):
            r = np.random.default_rng([77, n_reg, t])
            axis = r.normal(size=3)
            axis /= np.linalg.norm(axis)
            T_true = reg.pose_T(so3_exp(axis * np.deg2rad(r.uniform(-8, 8))),
                                r.uniform(-0.02, 0.02, 3))
            parts = []
            for j in range(n_reg):
                c = int(r.integers(len(surface)))
                p, _ = make_patch_probe(r, surface, c, T_true,
                                        n=N_PROBE // n_reg, n_out=N_OUTLIERS // n_reg)
                parts.append(p)
            probe = np.vstack(parts)
            init = landmark_init(r, surface, land_idx, T_true)
            T_reg = register_patch(probe, model, normals, tree, targets[0], init,
                                   n_verify=0, rng=r)[0]
            vals.append(tre_of(T_reg, T_true, targets))
        cov[n_reg] = float(np.median(vals))
        print(f"  {n_reg}곳에 나눠 찍기(총 {N_PROBE}점): TRE 중앙값 {cov[n_reg]*1e3:6.2f} mm")
    print("  → 점 수가 같아도 **흩어 찍는 편이 낫다**: 서로 다른 방향의 법선이 6-DOF 를 "
          "구속한다(임상에서 얼굴·귀·후두부를 함께 찍는 이유)")

    # ---- 그림 ----
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    ax = axes[0, 0]
    s = model[::3]
    sc = ax.scatter(s[:, 0] * 1e3, s[:, 2] * 1e3, s=2, c=var[::3], cmap="viridis",
                    vmax=np.quantile(var, 0.98))
    plt.colorbar(sc, ax=ax, label="surface variation")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title("Real head surface from MR (colour = geometric distinctiveness)",
                 fontsize=10)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")

    ax = axes[0, 1]
    probe_show = rowsA[0]["probe"]
    ax.scatter(model[::3, 1] * 1e3, model[::3, 2] * 1e3, s=2, color="0.75",
               label="pre-op surface")
    ax.scatter(probe_show[:, 1] * 1e3, probe_show[:, 2] * 1e3, s=6, color="crimson",
               alpha=0.7, label="intra-op patch (unknown pose)")
    ax.scatter(targets[:, 1] * 1e3, targets[:, 2] * 1e3, marker="*", s=120, color="k",
               label="deep targets")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title("One probed region + deep targets", fontsize=10)
    ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm]"); ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.scatter(svA, np.maximum(treA, 1e-5) * 1e3, s=30, color="tab:blue")
    ax.axhline(MISS_TOL * 1e3, color="0.4", ls=":", label="clinical tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("surface variation of the probed region")
    ax.set_ylabel("TRE [mm]")
    ax.set_title(f"Where you probe decides accuracy (r={corr:+.2f})", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.scatter(np.clip(sig_a, 1e-5, 1) * 1e3, np.maximum(tre_a, 1e-5) * 1e3, s=28,
               c=np.where(unsafe, "crimson", "tab:blue"))
    ax.axvline(MISS_TOL * 1e3 / K_SIGMA, color="seagreen", ls="--",
               label=f"abort gate (tol/{K_SIGMA:.0f})")
    ax.axhline(MISS_TOL * 1e3, color="0.4", ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predicted σ_target [mm]"); ax.set_ylabel("TRE [mm]")
    ax.set_title("Does the covariance gate work on real anatomy?", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    ax = axes[1, 1]
    vals = [100 * unsafe.mean(), ex_s, ex_d, ex_i, ex_v, ex_a]
    ax.bar(["naive", "① k·σ", "② consist.", "③ overlap", "④ verif. pt", "all"], vals,
           color=["crimson", "tab:orange", "tab:blue", "tab:purple", "seagreen", "0.4"])
    ax.tick_params(axis="x", labelsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("unsafe among executed [%]")
    ax.set_title("Reliability gates on real anatomy", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 2]
    ks = sorted(cov)
    ax.plot(ks, [cov[k] * 1e3 for k in ks], "-o", color="seagreen")
    ax.set_xticks(ks)
    ax.set_xlabel("number of probed regions (same total points)")
    ax.set_ylabel("TRE [mm]")
    ax.set_yscale("log")
    ax.set_title("Spread the probing, not the count", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("49. Surface registration on a real human MR scan — where to probe, "
                 "and whether the gate knows", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "49_registration_real_anatomy.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/49_registration_real_anatomy.png, "
          "assets/49_registration_real_anatomy.png")

    return dict(rowsA=rowsA, rows=rows, corr=corr, corr_verify=corr_v, cov=cov,
                det_sigma=det_s, det_disagree=det_d, det_inlier=det_i,
                det_verify=det_v, det_all=det_a, exec_all=ex_a, exec_verify=ex_v,
                unsafe_rate=float(unsafe.mean()), n_surface=len(surface),
                extent=extent, tre=tre_a, svA=svA, treA=treA,
                tre_centroid=tre_centroid, tre_land=tre_land)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 영상은 실제 사람의 MR 이지만 **프로빙은 여전히 모델링**이다. 실제 광학 트래커는
#     프로브 팁 캘리브레이션 오차·시선 방향 이방성·술자의 찍는 습관이 더 얹힌다.
#   - 강체 정합이다. 두피는 눌리고 밀리는 연조직이라 실제로는 변형이 오차의 큰 몫이며,
#     그래서 임상에서는 뼈 기준(골성 랜드마크·프레임)이나 변형 정합을 쓴다.
#   - 표면은 단순 임계 + 최대 연결성분으로 얻었다(세그멘테이션 알고리즘이 아니다).
#     귀·코 주변은 얇은 구조라 임계에 민감하다.
#   - '특징량'(surface variation)은 국소 PCA 근사다. 정합 조건화를 완전히 대표하지는
#     않으며(법선 분포의 전역 배치가 더 직접적), 여기서는 경향을 보이는 데 쓴다.
#   - 표적은 해부학적 병변이 아니라 기하적으로 정의한 심부 점이다. TRE 의 절대값보다
#     **영역에 따른 상대 변화**를 읽어야 한다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
