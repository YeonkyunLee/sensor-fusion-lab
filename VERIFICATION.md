# Verification & risk analysis — the image-guided targeting chain

This document applies **medical-device engineering practice** (requirements → hazard analysis →
mitigation → objective evidence → residual risk) to the image-guided chain built in experiments 39–49.
It is an engineering exercise on a personal research lab, written because in this domain an algorithm
is only half the work: the other half is being able to say *what could go wrong, what stops it, and
what evidence exists that it does*.

> **What this is not.** Not a medical device, not a regulatory submission, and not clinically validated.
> No patient data: everything runs on synthetic or public data (published UR5 specifications, Stanford
> Bunny laser scans, a public MR head volume). Severity/likelihood below are the author's engineering
> judgement for prioritising work, not a clinical risk assessment. Structure follows the standard
> shape of ISO 14971-style analysis (hazard → cause → mitigation → evidence) because that shape is
> useful, not to claim conformity with it.

## 1. Intended function (of the research prototype)

A robot places a needle-like tool at a target planned in a pre-operative image, avoiding a critical
structure, on a **rigid phantom**. The chain is: register the phantom to the image → map the plan into
robot coordinates → solve inverse kinematics → track the trajectory → (optionally) insert into tissue.

Reference implementation: `scripts/42_image_guided_targeting.py` (planar), `45_image_guided_6dof.py`
(spatial UR5 + real scan), `47_needle_impedance.py` (contact), `49_registration_real_anatomy.py`
(real human MR).

## 2. Requirements (verifiable)

| ID | Requirement | Measured | Evidence |
|--|--|--|--|
| R1 | Registration shall recover the phantom pose to sub-millimetre target error under nominal probing | TRE 0.081 mm / 0.132° (real scan), 0.048 mm (Bunny), median 0.9 mm (real MR) | `test_registration_recovers_pose_on_real_scans`, `test_registration_reaches_clinical_accuracy_on_real_anatomy` |
| R2 | Inverse kinematics shall not command unbounded joint motion near singularities | DLS step 0.086 rad where the pseudo-inverse commands 20.9 rad | `test_dls_bounded_where_pseudo_blows_up`, `test_wrist_singularity_has_zero_manipulability` |
| R3 | Trajectory tracking error shall be small compared with the registration error | servo share ≈0 vs registration 0.081–0.094 mm (calibrated computed torque) | `test_registration_dominates_the_error_budget_when_calibrated`, `test_full_pipeline_registration_dominates_and_control_is_stable` |
| R4 | The planned tool path shall keep a stated clearance from the critical structure, accounting for tool geometry | shaft clearance 2.17 mm, cuts at 9.7° of axis error (a tip-only check is blind to this) | `test_shaft_clearance_sees_what_tip_check_cannot`, `test_plan_creates_a_tight_but_feasible_corridor` |
| R5 | The system shall refuse to execute when registration reliability is insufficient | unsafe executions 37% → 3.8% (synthetic), 45% → 0% with a verification point (real MR) | `test_reliability_gate_catches_bad_registrations`, `test_verification_point_beats_the_covariance_gate` |
| R6 | Interaction force during insertion shall be bounded and selectable against accuracy | peak ≈4 N (tissue-limited); breakthrough lunge 0.655 → 0.000 mm by lowering stiffness | `test_position_control_is_stiffer_impedance_yields`, `test_stiffness_sets_the_tradeoff` |
| R7 | Model error shall be detectable and correctable from robot logs | tracking residual flags it; identification 45.9 → 0.003 mm | `test_loop_closes_the_parametric_gap`, `test_identification_recovers_payload_and_friction` |
| R8 | Dynamics and kinematics implementations shall be independently cross-verified | RNEA vs Lagrangian agree to 1e-10; M to 1e-15; energy conserved to 1e-6; Jacobian vs finite differences 1e-10 | `test_rnea_agrees_with_lagrangian_dynamics`, `test_energy_is_conserved_without_input`, `test_geometric_jacobian_matches_finite_differences` |

## 3. Hazard analysis

Severity: **S3** could injure (wrong tissue cut / critical structure hit), **S2** procedure fails or
must be redone, **S1** reduced accuracy within tolerance. Likelihood is judged *before* mitigation.

| # | Hazard (harm) | Cause | Sev | Lik | Mitigation implemented | Evidence (test) | Residual |
|--|--|--|--|--|--|--|--|
| H1 | Tool driven to the wrong place | Registration converged to a wrong basin, or coarse alignment absent | S3 | High | Landmark coarse alignment → surface ICP; multi-start consistency check; independent verification point | `test_landmark_coarse_alignment_is_required` (92.94 → 0.68 mm), `test_consistency_signal_transfers_better_than_covariance`, `test_verification_point_beats_the_covariance_gate` | **Residual: 3.8–5% of executed plans still unsafe in the synthetic study; the covariance signal alone catches only 5% on real scans.** Needs an independent second modality, not another statistic. |
| H2 | Registration looks good but is wrong (false confidence) | FRE (measurable) does not bound TRE (what matters) | S3 | Med | Never gate on FRE alone; verification point residual (correlates +0.90 with TRE); document the trap | `test_registration_recovers_patient_pose` (FRE vs TRE), exp 41/44 results in README | A surface verification point **underestimates** deep TRE through leverage. Verify near the target, not only on the surface. |
| H3 | Critical structure hit even though the plan cleared it | Tool modelled as a point; the shaft sweeps a volume; needle bends | S3 | Med | Clearance computed along the **shaft**, not the tip; bending model + spin compensation | `test_shaft_clearance_sees_what_tip_check_cannot`, `test_bending_eats_clearance_and_spin_restores_it` (43% of the corridor) | Bending compensation is **open-loop**; tissue inhomogeneity changes curvature. Needs tip tracking. |
| H4 | Uncontrolled joint motion / arm lunge | Inverse kinematics near a singularity; unstable gains | S3 | Med | Damped least squares; manipulability monitored along the path; per-joint gain scaling (joint inertias span 2.4 → 1e-4 kg·m²) | `test_dls_bounded_where_pseudo_blows_up`, `test_full_pipeline_registration_dominates_and_control_is_stable` (w ≥ 0.064) | Naive Cartesian impedance was found to diverge (spin mode, 1.2e-4 kg·m²); fixed by operational-space control, but stability limits with real force-sensor noise/delay are untested. |
| H5 | Excessive force on tissue / breakthrough overshoot | Stiff position control against a discontinuous puncture | S3 | Med | Operational-space impedance with a selectable stiffness; lunge budget picks the operating point | `test_position_control_is_stiffer_impedance_yields`, `test_stiffness_sets_the_tradeoff` | Too-soft control cannot puncture at all — safety and function trade against each other. No force-sensor noise model. |
| H6 | Silent accuracy loss after a hardware change | Tool swapped → payload/friction differ from the controller model | S2 | High | Residual monitor flags it; log-based identification restores accuracy | `test_loop_closes_the_parametric_gap`, `test_uncalibrated_payload_flips_the_budget` (servo becomes 10× registration) | Identification only covers what the model *can* express: a structural gap plateaus at 0.207 mm and needs a model extension **plus** an excitation that visits the regime. |
| H7 | Calibration campaign produces a wrong model | Identification data does not excite the parameters | S2 | Med | Condition number reported; dedicated excitation trajectories; low-speed excitation for friction | `test_excitation_beats_clinical_trajectory_for_identification` (cond 2.1e3 vs 1.9e2), `test_lowspeed_excitation_actually_visits_the_stiction_regime` | Over-parameterising costs accuracy on a clean plant (10× worse) — model selection is a judgement call. |
| H8 | Poor probing gives a poor registration, unnoticed | Digitising a smooth, near-spherical region (rotation unconstrained) | S2 | High | Region distinctiveness quantified; guidance to spread probing over regions | `test_probing_a_feature_rich_region_helps` (1.26 vs 0.49 mm), `test_spreading_the_probe_beats_concentrating_it` (0.68 → 0.31 mm) | Operator behaviour is not modelled; no on-line "your coverage is insufficient" indicator implemented. |
| H9 | Wrong result from an implementation error | Convention mistakes (DH frames), discretisation errors | S3 | Med | Two independent implementations cross-checked; analytic and conservation checks | `test_rnea_agrees_with_lagrangian_dynamics`, `test_beam_matches_analytic_cantilever`, `test_energy_is_conserved_without_input` | Caught two real bugs this way (standard-vs-modified DH recursion; a missing second integration in the beam model). Neither was visible by inspection — assume more remain. |
| H10 | Autonomous motion continues while localisation degrades | Estimate uncertainty grows but the rule trusts the mean | S3 | Med | Uncertainty-aware stopping (mean + k·σ) | `test_uncertainty_aware_prevents_violations` (60% → 0% violations) | Conservative: stops ~1.3 m early. Tuning k is a clinical trade-off, not a technical one. |

## 4. Traceability summary

- **133 tests, all passing** (`pytest -q`, ~3.7 min). Every experiment has at least one test; the
  medical chain (39–49) carries **65** of them, distributed as: kinematics 4, dynamics 3, planar
  capstone 6, sim-to-real loop 7, Bunny scans 4, UR5 6-DOF core 9, 6-DOF capstone 5, structural gap 6,
  contact 6, flexible needle 7, real anatomy 8.
- Requirements R1–R8 above each name the tests that verify them; hazards H1–H10 each name the tests
  that evidence their mitigation.
- Every experiment script ends with an explicit **한계·트레이드오프 (limits & trade-offs)** block, and
  README repeats the honest limits per experiment. Those are the inputs to the "residual" column.
- Numbers quoted here are reproducible by running the named script; they are printed to stdout and
  plotted to `assets/`.

## 5. Residual risk statement

The dominant residual risks are, in order:

1. **Confidently wrong registration** (H1/H2). Every reliability signal tested degrades as the data
   becomes more realistic — 85% detection on synthetic anatomy, 5% on real laser scans, 44% on a real
   MR at a 2 mm tolerance. Only an *independent* check (a verification point not used in the fit)
   reached 100%. The lesson is architectural: safety must not depend on a statistic derived from the
   same fit it is judging.
2. **Rigid-body assumptions** (H2/H3). Registration is rigid while soft tissue deforms; the needle
   model is a small-angle beam with a simplified tissue channel. Both under-estimate real error.
3. **Everything is simulated below the interface** (H4/H5/H6). The "real" arm is a deliberately
   mismatched simulation without backlash, joint elasticity or gear nonlinearity; force sensing is
   ideal. Stability margins that depend on sensor noise and delay are therefore unverified.

## 6. What would come next in a real V&V effort

- Verification points **near the target**, not only on the surface (H2).
- Closed-loop bending compensation with tip tracking (H3).
- Force-sensor noise/delay model and a passivity argument for the contact controller (H5).
- Deformable registration, and bone-based landmarks where available (H2).
- Software lifecycle artefacts (configuration management, change control, unit-level requirements
  tracing) if this were ever more than a research lab.

## References in this repo

- Experiments and measured numbers: [README](README.md) §39–49
- Beginner-oriented walk-through: [LEARNING_PATH.md](LEARNING_PATH.md) stage 8
- Narrative write-ups: [blog/06](blog/06_surgical_arm_error_budget.md),
  [blog/07](blog/07_sim_to_real_and_real_scans.md)
