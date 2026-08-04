# Verification & risk analysis — the image-guided targeting chain

This document applies **medical-device engineering practice** (requirements → hazard analysis →
mitigation → objective evidence → residual risk) to the image-guided chain built in experiments 39–55.
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
Two extensions are covered separately: the tool may be **teleoperated** by a human over a delayed
channel (#50), and the anatomy may **deform** between imaging and intervention (#51–#53, #55).

Reference implementation: `scripts/42_image_guided_targeting.py` (planar), `45_image_guided_6dof.py`
(spatial UR5 + real scan), `47_needle_impedance.py` (contact), `49_registration_real_anatomy.py`
(real human MR), `50_teleoperation_delay.py` (human in the loop),
`51_deformable_registration.py` (non-rigid anatomy), `52_probing_the_prior.py` (checking the
deformation model's assumptions against a sub-surface observation), `53_measurement_changes_it.py`
(the same check with a non-ideal intraoperative modality), `54_closed_loop_needle.py` (closing the
open-loop bending compensation), `55_correspondence_search.py` (removing the correspondence assumption).

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
| R9 | With a human in the loop, the force channel shall remain passive and the displayed force shall represent the environment | channel energy ≥ 0 (0.028–0.032 J for wave variables vs 0.185 J injected by P-P); force error 4.01 → 0.19 N at 200 ms | `test_communication_channel_is_passive_for_wave_variables`, `test_wave_variables_stay_stable_at_large_delay`, `test_pp_reflects_the_coupling_spring_not_the_environment` |
| R10 | An active constraint (virtual fixture) shall be enforced without becoming a source of instability | local rendering holds at 50 000 N/m (17.08 → 0.10 mm); the same wall through a 50 ms path chatters at 800 N/m and diverges above | `test_virtual_fixture_must_be_rendered_locally` |
| R11 | Where the anatomy deforms, the residual error the rigid model cannot remove shall be quantified, and any deformation correction shall be justified by its assumptions rather than by surface fit | rigid leaves the full shift at depth (9.98/6.02/4.10/2.66 mm at 20/35/50/70 mm); the skull prior, not the interpolator, recovers it through a 4% window (3.31 → 0.60 mm) | `test_rigid_registration_leaves_the_whole_deformation_at_the_target`, `test_the_prior_not_the_interpolator_wins_at_narrow_exposure`, `test_surface_residual_does_not_rank_methods_by_deep_accuracy` |
| R12 | The assumptions a deformation correction relies on shall be checkable against an observation that is independent of the data they are applied to | surface residual as a gate scores AUROC 0.52 (chance); one held-out sub-surface observation scores 0.81, three score 0.90 | `test_a_depth_check_beats_the_surface_check`, `test_surface_only_cannot_recover_the_deep_mode`, `test_more_check_points_do_not_hurt` |
| R13 | Accuracy claims shall be stated for the **measurement chain that will actually be used**, including the sensor's bias, its noise-vs-depth behaviour, and its failure rate | with a non-ideal iUS the same gate falls 0.73 → 0.61 and the correction 1.43 → 2.84 mm; remedies return the correction to 1.82 mm but not the gate | `test_the_gate_degrades_when_the_check_uses_the_same_sensor`, `test_remedies_fix_the_correction_not_the_gate`, `test_random_noise_averages_out_but_bias_does_not` |
| R14 | Results shall be reproducible from the published script | identical output across runs, verified by a test that calls the experiment twice | `test_main_is_reproducible` |
| R15 | Where a compensation is open-loop, closing it shall be justified by an ablation that separates the measurement's contribution from the rest | the tip measurement adds nothing over a zero-measurement prior policy (0.49 vs 0.50 mm p90); the gain comes from actuation that can act again (0.98 → 0.37 mm) | `test_measurement_contributes_almost_nothing_with_one_shot_flip`, `test_duty_cycling_beats_the_flip_policy_with_the_same_sensor`, `test_repeated_replanning_is_where_duty_wins` |
| R16 | Any accuracy claim that assumes known correspondences shall be restated with correspondences actually searched for | ground-truth correspondence 0.54 mm vs nearest-point 1.41 mm (2.6x) on the same deformation and exposure | `test_finding_correspondence_costs_a_real_multiple`, `test_correspondence_error_tracks_the_slide_while_residual_does_not` |

## 3. Hazard analysis

Severity: **S3** could injure (wrong tissue cut / critical structure hit), **S2** procedure fails or
must be redone, **S1** reduced accuracy within tolerance. Likelihood is judged *before* mitigation.

| # | Hazard (harm) | Cause | Sev | Lik | Mitigation implemented | Evidence (test) | Residual |
|--|--|--|--|--|--|--|--|
| H1 | Tool driven to the wrong place | Registration converged to a wrong basin, or coarse alignment absent | S3 | High | Landmark coarse alignment → surface ICP; multi-start consistency check; independent verification point | `test_landmark_coarse_alignment_is_required` (92.94 → 0.68 mm), `test_consistency_signal_transfers_better_than_covariance`, `test_verification_point_beats_the_covariance_gate` | **Residual: 3.8–5% of executed plans still unsafe in the synthetic study; the covariance signal alone catches only 5% on real scans.** Needs an independent second modality, not another statistic. |
| H2 | Registration looks good but is wrong (false confidence) | FRE (measurable) does not bound TRE (what matters) | S3 | Med | Never gate on FRE alone; verification point residual (correlates +0.90 with TRE); document the trap | `test_registration_recovers_patient_pose` (FRE vs TRE), exp 41/44 results in README | A surface verification point **underestimates** deep TRE through leverage. Verify near the target, not only on the surface. |
| H3 | Critical structure hit even though the plan cleared it | Tool modelled as a point; the shaft sweeps a volume; needle bends | S3 | Med | Clearance computed along the **shaft**, not the tip; bending model + spin compensation | `test_shaft_clearance_sees_what_tip_check_cannot`, `test_bending_eats_clearance_and_spin_restores_it` (43% of the corridor) | ~~Bending compensation is open-loop~~ — closed in #54, where an ablation showed the tip measurement contributed nothing and the fix was actuation that can act again (p90 0.98 → 0.37 mm). Remaining: duty cycling's own costs (tissue damage, torsional windup) are not modelled. |
| H4 | Uncontrolled joint motion / arm lunge | Inverse kinematics near a singularity; unstable gains | S3 | Med | Damped least squares; manipulability monitored along the path; per-joint gain scaling (joint inertias span 2.4 → 1e-4 kg·m²) | `test_dls_bounded_where_pseudo_blows_up`, `test_full_pipeline_registration_dominates_and_control_is_stable` (w ≥ 0.064) | Naive Cartesian impedance was found to diverge (spin mode, 1.2e-4 kg·m²); fixed by operational-space control, but stability limits with real force-sensor noise/delay are untested. |
| H5 | Excessive force on tissue / breakthrough overshoot | Stiff position control against a discontinuous puncture | S3 | Med | Operational-space impedance with a selectable stiffness; lunge budget picks the operating point | `test_position_control_is_stiffer_impedance_yields`, `test_stiffness_sets_the_tradeoff` | Too-soft control cannot puncture at all — safety and function trade against each other. No force-sensor noise model. |
| H6 | Silent accuracy loss after a hardware change | Tool swapped → payload/friction differ from the controller model | S2 | High | Residual monitor flags it; log-based identification restores accuracy | `test_loop_closes_the_parametric_gap`, `test_uncalibrated_payload_flips_the_budget` (servo becomes 10× registration) | Identification only covers what the model *can* express: a structural gap plateaus at 0.207 mm and needs a model extension **plus** an excitation that visits the regime. |
| H7 | Calibration campaign produces a wrong model | Identification data does not excite the parameters | S2 | Med | Condition number reported; dedicated excitation trajectories; low-speed excitation for friction | `test_excitation_beats_clinical_trajectory_for_identification` (cond 2.1e3 vs 1.9e2), `test_lowspeed_excitation_actually_visits_the_stiction_regime` | Over-parameterising costs accuracy on a clean plant (10× worse) — model selection is a judgement call. |
| H8 | Poor probing gives a poor registration, unnoticed | Digitising a smooth, near-spherical region (rotation unconstrained) | S2 | High | Region distinctiveness quantified; guidance to spread probing over regions | `test_probing_a_feature_rich_region_helps` (1.26 vs 0.49 mm), `test_spreading_the_probe_beats_concentrating_it` (0.68 → 0.31 mm) | Operator behaviour is not modelled; no on-line "your coverage is insufficient" indicator implemented. |
| H9 | Wrong result from an implementation error | Convention mistakes (DH frames), discretisation errors | S3 | Med | Two independent implementations cross-checked; analytic and conservation checks | `test_rnea_agrees_with_lagrangian_dynamics`, `test_beam_matches_analytic_cantilever`, `test_energy_is_conserved_without_input` | Caught two real bugs this way (standard-vs-modified DH recursion; a missing second integration in the beam model). Neither was visible by inspection — assume more remain. |
| H10 | Autonomous motion continues while localisation degrades | Estimate uncertainty grows but the rule trusts the mean | S3 | Med | Uncertainty-aware stopping (mean + k·σ) | `test_uncertainty_aware_prevents_violations` (60% → 0% violations) | Conservative: stops ~1.3 m early. Tuning k is a clinical trade-off, not a technical one. |
| H11 | Delayed force feedback drives the tool into oscillation at the tissue | A delayed bilateral channel injects energy; the hand feels the coupling spring, not the tissue | S3 | High | Wave-variable encoding makes the channel passive for any constant delay; channel energy integrated and checked; position-correction channel added for drift | `test_communication_channel_is_passive_for_wave_variables`, `test_direct_force_reflection_degrades_with_delay` (chatters from 50 ms), `test_position_correction_reduces_wave_variable_drift` | Passivity costs transparency and correspondence (7.4 mm master–slave drift remains). Constant delay only — jitter and packet loss need time-domain passivity control, untested here. |
| H12 | The safety constraint itself destabilises the system | A virtual fixture rendered across the delayed channel | S3 | Med | Constraint rendered **locally** on the master | `test_virtual_fixture_must_be_rendered_locally` (0.10 mm local vs divergence remote) | Local rendering means the constraint uses the master's *model* of where the forbidden zone is — a stale or misregistered model enforces the wrong wall confidently. Links back to H1. |
| H13 | Tissue deforms between imaging and intervention; the rigid registration is confidently wrong | Brain shift (gravity, CSF loss, retraction), breathing, retraction | S3 | High | Deformation recovered from the exposed surface; the correction's prior stated explicitly and its failure mode measured | `test_rigid_registration_leaves_the_whole_deformation_at_the_target`, `test_deformable_recovers_most_of_the_shift`, `test_the_prior_stops_paying_when_data_is_plentiful` | **Residual: the correction is only as good as its prior.** The skull assumption that buys 5.5× through a narrow window is an assumption about *this* patient; where it fails (wide resection, bilateral opening) the same regularization becomes a bias. Correspondences here are given, not found. |
| H14 | A deformation model is trusted because it is "physics-based" | Model bias mistaken for numerical error; grid-convergence read as validation | S2 | Med | Grid refinement run as an experiment, not an assumption; predicted vs true displacement reported at depth | `test_finer_grid_makes_the_harmonic_model_worse` (1.12 → 2.38 mm as the grid refines; 80% → 57% of the true shift) | Convergence testing verifies the solver, not the model. The coarse grid that scored best was cancelling error, not correcting it — with no ground truth, this would have been invisible. |
| H15 | Deformation occurs in a mode the intraoperative data cannot observe, and the correction reports success anyway | Displacement localized at depth (ventricular collapse, deep relaxation) leaves no trace on the exposed surface | S3 | Med | The failure mode is constructed and measured, not assumed away; a sub-surface observation held out of the fit is used as the check, placed near the target | `test_deep_mode_is_invisible_at_the_surface_by_construction` (0.03 mm surface trace vs 3.5 mm at depth), `test_surface_only_cannot_recover_the_deep_mode`, `test_a_depth_check_beats_the_surface_check` (AUROC 0.52 → 0.81) | **This is an observability limit, not an algorithm gap.** No processing of surface data recovers it. The check itself is bounded by the modality: with σ 1.5 mm ultrasound against a 2 mm tolerance the gate ceilings around AUROC 0.9, and a violation outside the imaging cone is not refutable at all. |
| H16 | A safety gate is added where simply taking more measurements would have been better | Gate complexity adopted without comparing against the do-everything baseline | S1 | Med | The gated policy is scored against "always take 4 observations" rather than only against the ungated baseline | `test_knowing_is_cheaper_than_fixing_at_the_first_observation`; policy table in #52 (gated 1.38 mm / 33% unsafe vs always-4 1.47 mm / 28% unsafe) | Recorded as a **negative result**: gating did not beat measuring more. Its value is confined to expensive-observation regimes and to cases the correction cannot fix, where the gate converts a silent failure into a stated one. |
| H17 | The act of measuring changes the thing being measured | Ultrasound probe pressure indents the tissue; the indentation propagates to depth and is always inward | S3 | High | The bias is modelled and measured rather than assumed away; subtracted using tracked contact position and press depth | `test_indentation_is_always_inward_and_decays`, `test_indentation_does_not_average_out_over_contact_points`, `test_de_indentation_recovers_but_needs_the_right_model` | **A bias is not reduced by taking more data** — its share of the error grew 13% → 31% over a 32× increase in observations. Subtracting it requires knowing the tissue response; a 30% error in that model leaves 1.37 vs 1.16 mm. Real tissue is viscoelastic and relaxes, which this model does not capture. |
| H18 | An accuracy or detection figure is quoted for an idealised sensor and read as achievable | Verification measurements modelled as ground truth plus isotropic noise | S2 | High | Every #52 figure re-measured with a modality that has bias, depth-dependent noise and a 15% mismatch rate — including on the **verification** measurement | `test_the_gate_degrades_when_the_check_uses_the_same_sensor`, `test_depth_weighting_helps_but_does_not_restore` | The gate's ceiling is set by the modality: at target depths the signal (2–6 mm) is below the single-check noise (3.6–6.2 mm), so no statistic and no number of check points recovers #52's figure. Remedies aimed at bias and outliers rescued the correction and did nothing for the gate. |
| H19 | A robust estimator discards the informative minority | Robust scale estimated over a heterogeneous control set; the dense majority sets the scale | S2 | Med | Residuals normalised by **known** σᵢ and the robust weight applied only where an outlier mechanism exists; annealed λ so a flexible warp cannot hide outliers | `test_robust_fit_beats_least_squares_with_outliers`, `test_outliers_break_least_squares_and_robust_recovers` | Found the hard way: a MAD scale over surface+anchor+depth points deleted the depth observations. Robustness also costs below four observations, where rejecting one point hurts more than the outlier. Real mismatches attach to similar structures and are correspondingly harder to reject than the random-direction ones modelled here. |
| H20 | Adding feedback is credited with an improvement it did not cause | The closed loop also changes the default and the timing; without an ablation all of it is attributed to the measurement | S2 | High | Two zero-measurement baselines added (plan from the population mean; flip at the decision depth with no estimate) | `test_measurement_contributes_almost_nothing_with_one_shot_flip` (prior-only 0.49 vs MAP 0.50 mm p90), `test_orientation_helps_but_does_not_rescue_the_one_shot_policy` | The honest reading of #54 is that the sensor bought nothing here. Any future "closed loop improved X" claim in this repo is required to carry the same two baselines. |
| H21 | Effort goes to the sensor when the limit is the actuator | The estimate is visibly poor, so sensing looks like the problem | S1 | Med | Actuation varied with sensing held fixed; the residual re-attributed by handing the controller the true parameters | `test_duty_cycling_beats_the_flip_policy_with_the_same_sensor`, `test_estimation_is_no_longer_the_bottleneck_under_duty`, `test_sensor_quality_barely_moves_the_duty_result` | Once actuation can act repeatedly, a 10× quieter tip sensor moves p90 by 0.01 mm. The limit is now replanning granularity, saturation and the small-angle model — none of which more sensing fixes. |
| H22 | The deformation slides along the surface and the surface registration cannot see it | Tangential displacement leaves a smooth surface unchanged, so nearest-point search is blind to it and the residual stays at the noise floor | S3 | High | The normal/tangential split is constructed and measured separately; the correspondence error is reported alongside the residual so the blind component is visible | `test_tangential_slide_does_not_move_a_smooth_surface`, `test_correspondence_error_tracks_the_slide_while_residual_does_not` (0.48 → 1.53 mm correspondence error at a flat 0.92 → 1.17 mm residual) | **Not mitigated by any method tested.** Point-to-plane, landmark anchors and robust kernels all fail; the requirement is non-geometric correspondence (texture, vessel pattern, markers) across the whole exposure, and the benefit is roughly linear in the fraction so partial coverage buys only partial accuracy. |
| H23 | A wrong correspondence does not look wrong | Nearest-point search maps a displaced observation to a nearby surface point, producing a plausible small displacement from a false match | S2 | High | Outliers injected as observation jumps rather than as large residuals, so the failure is realistic; robust fitting scored against it rather than assumed effective | `test_robust_barely_helps_because_the_outliers_do_not_look_like_outliers` (1.76 → 1.62 mm) | Residual-based rejection has almost nothing to work with. Detecting this needs a consistency check that does not go through the residual — e.g. agreement between independently matched subsets. Not implemented. |

## 4. Traceability summary

- **204 tests, all passing** (`pytest -q`, ~11 min). Every experiment has at least one test; the
  medical chain (39–55) carries **134** of them, distributed as: kinematics 4, dynamics 3, planar
  capstone 6, sim-to-real loop 7, Bunny scans 4, UR5 6-DOF core 9, 6-DOF capstone 5, structural gap 6,
  contact 6, flexible needle 7, real anatomy 8, teleoperation 9, deformable registration 12, probing the
  prior 11, non-ideal modality 12, closed-loop steering 15, correspondence search 10. The browser demo's core is separately verified headless
  (`tests/guided_demo_check.js` via node, skipped when node is absent) so the demo cannot claim an
  ordering the maths does not support.
- Requirements R1–R16 above each name the tests that verify them; hazards H1–H23 each name the tests
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
2. **Correspondence, not just the warp** (H22/H23). #55 removed the last thing #51–#54 were given: it
   measured 0.54 mm with true correspondences and **1.41 mm** when they are searched for, and showed the
   blind component (tangential slide) leaves the measurable residual flat. Point-to-plane, landmarks and
   robust kernels all failed on it, and the benefit of non-geometric correspondence is roughly linear in
   coverage — so every deformation number in #51–#54 should be read as **assuming a capability the system
   does not yet have.**
2. **Assumptions bought as regularization** (H13/H14/H15). #51 quantified the rigid gap — the full shift
   at the target, 2.7–10 mm here — and showed it can be recovered to sub-millimetre. But what recovered
   it through a narrow exposure was a *prior* ("the skull holds the scalp outside the window"), not the
   interpolator, and the volumetric physics model was measurably **biased** in a way grid refinement made
   worse, not better. So the residual risk did not disappear; it changed shape, from "the model is too
   simple" to "the model asserts something about this patient that may not be true." #52 then showed the
   sharpest version: a deformation mode that leaves 0.03 mm on the exposed surface and 3.5 mm at the
   target is not detectable by *any* function of the surface data (gate AUROC 0.52), and one held-out
   sub-surface observation raises that to 0.81. **The mitigation for this class of risk is a different
   observation, not a better estimator** — and #53 then showed the bill for that observation: with a
   realistic probe (indentation bias, σ growing with depth, 15% mismatch) the same gate falls to 0.61
   and stays there under every remedy, because at target depth the signal is already below the check's
   own noise. The needle model carries the same shape of risk (small-angle beam, simplified tissue
   channel).
3. **Everything is simulated below the interface** (H4/H5/H6). The "real" arm is a deliberately
   mismatched simulation without backlash, joint elasticity or gear nonlinearity; force sensing is
   ideal. Stability margins that depend on sensor noise and delay are therefore unverified.
4. **The human and the channel** (H11/H12). Delay is handled only for a constant one-way delay with a
   fixed linear operator impedance. Real networks jitter and drop packets, and a real surgeon adapts —
   closing the operator's visual loop through the same delay destabilised the human loop regardless of
   architecture (observed while building #50).

## 6. What would come next in a real V&V effort

- ~~Verification points **near the target**, not only on the surface (H2)~~ — done in #52: a held-out
  sub-surface observation near the planned target scores AUROC 0.81 vs 0.71 for one placed anywhere in
  the imaging cone, and vs 0.52 for the surface residual. What remains is *where exactly* to place it
  when the violation is localized and may fall outside the imaging cone.
- ~~A deformation model whose assumptions are **testable intraoperatively** (H13/H14)~~ — addressed in
  #52 with a modelled ultrasound depth check, and ~~a modality model that is not idealized~~ in #53
  (indentation bias, σ(d), 15% mismatch). What remains there is tissue **viscoelasticity** (the probe's
  effect relaxes over time, so the bias depends on when you measure), anisotropic ultrasound resolution,
  and structured rather than random feature mismatches.
- **A better verification modality, not a better verification statistic** (H18). #53 shows the check is
  noise-limited at the depths that matter; the useful next step is a measurement with lower σ at depth
  (or a standoff that removes the indentation entirely), not more processing.
- Closed-loop bending compensation with tip tracking (H3).
- Force-sensor noise/delay model and a passivity argument for the contact controller (H5).
- Bone-based landmarks where available (H2).
- Time-domain passivity control for a jittering, lossy channel (H11).
- Software lifecycle artefacts (configuration management, change control, unit-level requirements
  tracing) if this were ever more than a research lab.

## References in this repo

- Experiments and measured numbers: [README](README.md) §39–55
- Beginner-oriented walk-through: [LEARNING_PATH.md](LEARNING_PATH.md) stage 8
- Narrative write-ups: [blog/06](blog/06_surgical_arm_error_budget.md),
  [blog/07](blog/07_sim_to_real_and_real_scans.md)
