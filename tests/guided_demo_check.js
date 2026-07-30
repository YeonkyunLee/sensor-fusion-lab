// guided_demo.html 의 코어(<script id="core">)를 node 에서 헤드리스로 검증한다.
//   node tests/guided_demo_check.js
// 브라우저 데모도 "돌아가는 것처럼 보이는" 것으로 끝내지 않기 위한 장치 — 데모가 주장하는
// 순서(매끄러운 영역 < 특징적 영역 < 흩어 찍기)가 실제로 성립하는지 숫자로 확인한다.
const fs = require("fs");
const path = require("path");

const file = path.join(__dirname, "..", "guided_demo.html");
const html = fs.readFileSync(file, "utf8");
const m = html.match(/<script id="core">([\s\S]*?)<\/script>/);
if (!m) { console.error("core script block not found in guided_demo.html"); process.exit(1); }
const core = new Function(m[1] + "\nreturn CORE;")();

const { anatomy, buildModel, se2, apply, compose, invert, procrustes, icp,
        targetSigma, segClearance, PX_MM, TOL_MM } = core;

let fails = 0;
function check(name, cond, extra) {
  console.log((cond ? "  OK   " : "  FAIL ") + name + (extra ? "  " + extra : ""));
  if (!cond) fails++;
}

// 1) 모델과 법선
const M = buildModel(360);
check("모델 360점 + 법선", M.pts.length === 360 && M.nrm.length === 360);
let outward = 0;
for (let i = 0; i < 360; i++)
  if (M.nrm[i].x * M.pts[i].x + M.nrm[i].y * M.pts[i].y > 0) outward++;
check("법선이 외향", outward === 360, outward + "/360");

// 2) 랜드마크 조대정렬(2D Procrustes)
const Tk = se2(0.31, 40, -25);
const src = [0.3, 1.7, 3.9].map(t => anatomy(t));
const Tp = procrustes(src, src.map(p => apply(Tk, p)));
check("Procrustes 가 강체변환 복원",
      Math.abs(Tp.c - Tk.c) < 1e-9 && Math.abs(Tp.tx - Tk.tx) < 1e-6);
check("invert∘compose = I", (() => {
  const T = se2(0.7, 12, -9), I = compose(invert(T), T);
  return Math.abs(I.c - 1) < 1e-9 && Math.abs(I.tx) < 1e-9 && Math.abs(I.ty) < 1e-9;
})());
check("segClearance", Math.abs(segClearance({ x: 0, y: 0 }, { x: 100, y: 0 },
                                            { x: 50, y: 30 }) - 30) < 1e-9);

// 3) 데모의 핵심 주장: 어디를 찍느냐가 TRE 를 정한다
const T_TRUE = se2(0.38, 470, 300);
const TARGET = { x: 20, y: -18 }, VERIFY_T = 5.4, LAND_T = [0.6, 2.4, 4.3];
const lcg = seed => { let s = seed; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; };
const digitize = (t, noise, rnd) => {
  const p = apply(T_TRUE, anatomy(t));
  return { x: p.x + (rnd() - .5) * noise, y: p.y + (rnd() - .5) * noise };
};

function trial(centers, span, nEach, seed) {
  const rnd = lcg(seed), probe = [];
  centers.forEach(tc => {
    for (let i = 0; i < nEach; i++)
      probe.push(digitize(tc + (i / (nEach - 1) - .5) * span, 3, rnd));
  });
  const init = procrustes(LAND_T.map(t => digitize(t, 8, rnd)), LAND_T.map(t => anatomy(t)));
  const out = icp(probe, M, init);
  const tt = apply(compose(out.T, T_TRUE), TARGET);
  const vd = digitize(VERIFY_T, 3, rnd), vm = anatomy(VERIFY_T), vp = apply(out.T, vd);
  return {
    tre: Math.hypot(tt.x - TARGET.x, tt.y - TARGET.y) * PX_MM,
    fre: out.fre * PX_MM,
    sig: targetSigma(out.A, out.fre, TARGET) * PX_MM,
    ver: Math.hypot(vp.x - vm.x, vp.y - vm.y) * PX_MM,
  };
}

const med = a => { const v = a.slice().sort((x, y) => x - y); return v[Math.floor(v.length / 2)]; };
const smooth = [], feature = [], spread = [];
for (let s = 1; s <= 5; s++) {
  smooth.push(trial([4.9], 0.9, 60, s * 7919));
  feature.push(trial([0.6], 0.9, 60, s * 7919));
  spread.push(trial([0.6, 2.4, 4.9], 0.5, 20, s * 7919));
}
const mS = med(smooth.map(r => r.tre)), mF = med(feature.map(r => r.tre)),
      mP = med(spread.map(r => r.tre));
console.log(`  TRE 중앙값 — 매끄러움 ${mS.toFixed(2)} mm | 특징적 ${mF.toFixed(2)} mm | ` +
            `흩어서 ${mP.toFixed(2)} mm  (실험 49: 1.26 / 0.49 / 0.31)`);
check("특징적 영역 < 매끄러운 영역", mF < mS, `${mF.toFixed(2)} < ${mS.toFixed(2)}`);
check("흩어 찍기 < 한 곳", mP < mS, `${mP.toFixed(2)} < ${mS.toFixed(2)}`);
check(`좋은 조건에서 허용치(${TOL_MM} mm) 이내`, mP < TOL_MM, mP.toFixed(2) + " mm");
check("σ 유한", feature.every(r => isFinite(r.sig)));

// 4) 검증점이 실패를 잡는가 (게이트의 근거)
const all = smooth.concat(feature, spread);
const bad = all.filter(r => r.tre > TOL_MM);
const caught = bad.filter(r => r.ver > TOL_MM).length;
console.log(`  허용 초과 ${bad.length}건 중 검증점이 잡은 것 ${caught}건`);
check("검증점이 실패의 70% 이상 검출", bad.length === 0 || caught >= bad.length * 0.7);

console.log(fails === 0 ? "\n모두 통과" : `\n실패 ${fails}건`);
process.exit(fails === 0 ? 0 : 1);
