# Development experiments (July to August 2026)

These are the scripts and result files from the two months before the final run, copied as they were on the lab laptop. They are exploratory. The final, pre-registered experiment is in `avi/` and `results/2026-09-01_final/`, and nothing in the paper's main results depends on this folder.

The files stay in one flat folder because many of them import each other by name. To run one, put `avi/` on the path so the shared modules resolve:

```bash
cd experiments/development
PYTHONPATH=../../avi python <script>.py        # Windows: set PYTHONPATH=..\..\avi
```

Most of them also need the GPU, the arm, or the August desk frames. The desk frames are not published because they were shot in a private room, so scripts that read them will not run from a fresh clone. The `.json` and `.csv` files next to each script are its saved output.

The dated story of these experiments is in `docs/lab-notebook/`, `docs/logs/progress-log-aug-sep.md` and `docs/technical-writeup.md`. The groups below are by topic, based on file names.

## July VLM benchmarks and model choice

Latency and accuracy benchmarks of candidate VLMs (Qwen2.5-VL through Ollama, Moondream, others) and the scripts behind the Polygence assignments. Preliminary.

`assignment2.json` · `assignment2.py` · `assignment3a.json` · `assignment3a.py` · `assignment3b_offline.json` · `assignment3b_offline.py` · `benchmark_results_20260722_171505.csv` · `benchmark_results_20260722_172338.csv` · `benchmark_results_20260722_173214.csv` · `benchmark_results_20260722_175930.csv` · `benchmark_results_20260722_180319.csv` · `benchmark_results_20260722_181137.csv` · `benchmark_vlm.py` · `multimodal_test.py` · `test_moondream_caption.py` · `test_moondream_debug.py` · `test_moondream_detect.py` · `test_moondream_gpu.py` · `test_moondream_prompt.py` · `test_vlm.py` · `vlm_bakeoff.py` · `vlm_bakeoff_results.json` · `vlm_comparison_test.py`

## Local VLM backend, GPU stability and speed

The early-August work on the local model server: the degenerate `@@@@` GPU output, restart and CPU fallback, context and batch sizes, image token budgets, and per-call latency. `llm_backend.py` is the client most of these scripts share.

`alt_backend_test.py` · `b10326_stability_solo.json` · `b10326_stability_solo.py` · `backend_latency_ab.json` · `backend_latency_ab.py` · `backend_stability_replicate.json` · `backend_stability_replicate.py` · `budget_discriminate.json` · `budget_discriminate.py` · `cacheram_test.json` · `cacheram_test.py` · `contention_busy.json` · `contention_quiet.json` · `contention_test.py` · `ctx_test.json` · `ctx_test.py` · `garbage_test.py` · `gpu_cuda_env_test.json` · `gpu_cuda_env_test.py` · `gpu_degen_diag.json` · `gpu_degen_diag.py` · `gpu_imgtokens_test.json` · `gpu_imgtokens_test.py` · `gpu_kv_test.json` · `gpu_kv_test.py` · `gpu_mmproj_test.json` · `gpu_mmproj_test.py` · `gpu_soak_test.py` · `llm_backend.py` · `pipeline_benchmark.json` · `pipeline_benchmark.py` · `pixel_budget.json` · `pixel_budget.py` · `profile_pipeline.json` · `profile_pipeline.py` · `speed_opts.json` · `speed_opts.py` · `speed_opts2.json` · `speed_opts2.py` · `test_b10310.py` · `test_kv_clear.py` · `ubatch_test.json` · `ubatch_test.py` · `unique_frame_stress.json` · `unique_frame_stress.py` · `verifier_speed.py` · `vision_token_sweep.json` · `vision_token_sweep.py` · `vram_test.json` · `vram_test.py`

## Scene listing and hallucination control

Asking the VLM to list what it sees, voting across repeated answers, and checking the lists against held-out ground truth.

`ask.py` · `bottle_phrasing_cycle2.py` · `bottle_phrasing_greedy_results.json` · `bottle_phrasing_results.json` · `confirm_bottle_phrasing_greedy.py` · `disambiguation_test.json` · `disambiguation_test.py` · `heldout_gt.py` · `heldout_validation.json` · `heldout_validation.py` · `listing_heldout_final.json` · `listing_heldout_final.py` · `listing_length_tuning.json` · `listing_length_tuning.py` · `listing_recall_test.json` · `listing_recall_test.py` · `open_listing_heldout.json` · `open_listing_heldout.py` · `open_listing_test.json` · `open_listing_test.py` · `phrasing_experiment.py` · `prompt_variants_test.json` · `prompt_variants_test.py` · `query_fix.py` · `score_bottle_phrasing.py` · `temporal_vote_test.json` · `temporal_vote_test.py` · `test_prompt_stdin.py` · `vote_decision.json` · `vote_decision.py`

## Detection: open-vocabulary detectors, verification, robustness

Choosing the detector (YOLO-World won), cropping and verifying its boxes with the VLM, and stress tests for darkness, blur, size, partial cover and look-alike objects (`nano_*.json`).

`adaptive_detector.py` · `adaptive_sampling_test.json` · `adaptive_sampling_test.py` · `colors.py` · `crop_verify_results.json` · `crop_verify_results_run1.json` · `crop_verify_results_run2.json` · `crop_verify_scores.json` · `crop_verify_scores_run1.json` · `crop_verify_scores_run2.json` · `crop_verify_test.py` · `cue_pipeline_test.py` · `cues.py` · `degrade_bench.json` · `degrade_bench.py` · `detect_experiment.py` · `detect_methods.py` · `detector_decision.json` · `detector_decision.py` · `hybrid_router_results.json` · `hybrid_router_test.py` · `mosaic_direction_check.py` · `mosaic_sanity_check.py` · `nano.py` · `nano_all.json` · `nano_box.json` · `nano_clear.json` · `nano_cloth.json` · `nano_clutter.json` · `nano_dark_06.json` · `nano_dark_12.json` · `nano_dark_25.json` · `nano_dark_50.json` · `nano_far.json` · `nano_glare.json` · `nano_impostor.json` · `nano_mostly.json` · `nano_partial.json` · `nano_shadow.json` · `nano_thin.json` · `nano_transparent.json` · `open_vocab_detect.py` · `openvocab_baseline.py` · `openvocab_batched.py` · `openvocab_batched_results.json` · `openvocab_phrasing_c2.json` · `openvocab_phrasing_c2.py` · `openvocab_phrasing_temp0_c2.py` · `openvocab_results.json` · `openvocab_results_cycle1_baseline.json` · `openvocab_results_perobject_gpu_480.json` · `proposer_verify_results.json` · `proposer_verify_scores.json` · `proposer_verify_test.py` · `region_proposer.py` · `res_boundary.py` · `resolution_sweep.json` · `resolution_sweep.py` · `rotation_test.py` · `rotation_validation.py` · `scene_gate_test.json` · `scene_gate_test.py` · `score_batched.py` · `score_openvocab.py` · `shape_signals.py` · `shape_test.py` · `smooth_verify.py` · `verify_sweep_test.py` · `verify_test.py` · `verify_throttle_test.py` · `yolo_tracker_dryrun.py` · `yolo_world_raw.json` · `yolo_world_test.py`

## Recovery and occlusion studies (synthetic and live corpora)

The mid-August studies behind the final design: synthetic disruption episodes, live desk episodes, the decay-timer sweep, the "what is covering it?" probe, CLIP occluder classification, and replay tools. Exploratory.

`clip_classifier.py` · `clip_occluder_classify.json` · `clip_occluder_classify.py` · `clip_occluder_real2.json` · `clip_occluder_real2.py` · `comfy_generate.py` · `corpus_bench.py` · `decay_sweep.json` · `decay_sweep.py` · `deep_validation.json` · `deep_validation.py` · `dropout_diag.py` · `erase_test.json` · `erase_test.py` · `fa_experiment.py` · `final_ship_test.json` · `final_ship_test.py` · `gen_batch.py` · `generated_bench.json` · `generated_bench.py` · `inpaint_corpus.py` · `inpaint_occluders.py` · `integration_test.py` · `isolation_test.py` · `live_bench.json` · `live_bench.py` · `live_disruption.json` · `live_disruption.py` · `live_experiment.json` · `live_experiment.py` · `live_experiment_chart.png` · `live_validation.py` · `motion_occluder.json` · `motion_occluder.py` · `periphery_calibrate.json` · `periphery_calibrate.py` · `probe_generalise.json` · `probe_generalise.py` · `probe_integration_check.py` · `probe_real.json` · `probe_real.py` · `probe_test.json` · `probe_test.py` · `probe_throttle_test.py` · `replay_live.json` · `replay_live.py` · `study_results.json` · `study_results_mixed.json` · `study_results_prefix.json` · `study_results_surround040.json` · `surround_real.json` · `surround_real.py` · `surround_sweep.json` · `surround_sweep.py`

## Search

Searching for a lost target by panning the arm, including the VLM-guided search that lost to random choices.

`find_live.py` · `find_object.py` · `findobject.py` · `findobject_search.py` · `findsearch_final.py` · `run_search_study.py` · `search_challenge.json` · `search_challenge.py` · `search_prompt_sweep.json` · `search_prompt_sweep.py` · `search_study.json` · `search_trials.json` · `search_trials.py` · `test_tile_map.py` · `tile_map.py`

## Earlier live tracking, control and hardware checks

The July and early-August live tracking loop (`main.py`, `controller.py`, `vlm_recovery.py`) and camera and servo checks.

`camera_probe.py` · `camera_starve_test.py` · `capture_hand_test.py` · `capture_sweep_c2.json` · `capture_sweep_c2.py` · `controller.py` · `controller_test.py` · `hw_check.json` · `hw_check.py` · `live_detect.py` · `live_track.py` · `live_view.py` · `live_yolo_track.py` · `main.py` · `multi_object_capture.py` · `newbin_cycle.py` · `newbin_probe_quick.py` · `newbin_test.json` · `newbin_test.py` · `single_move_test.py` · `tracker.py` · `vlm_recovery.py`

## Reports and plots

Charts made for the August progress report.

`experiment_log.json` · `plot_experiment.py` · `report_accuracy_evolution.png` · `report_charts.py` · `report_latency_evolution.png` · `score_experiment.py`
