# GrayNav SurfaceDepth E0-E3 experiment runbook

This runbook implements the gated four-class SurfaceDepth optimization. It does
not modify the COCO80 detector, A1 SDK, Aurora, or the board rollback image.
Commands run directly in the AutoDL terminal and retain tqdm and TensorBoard.

## Fixed paths

```bash
export GRAYNAV_ROOT=/root/autodl-tmp/graynav/model_optimization
export GRAYNAV_PUBLIC_ROOT=/root/autodl-tmp/graynav_public
export GRAYNAV_PREPARED=/root/autodl-tmp/graynav_surface_depth_prepared_v2
export GRAYNAV_RUN_ROOT=/root/autodl-tmp/graynav_surface_depth_optimized
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

## 1. Archive E0 before creating prepared-v2

This refuses to overwrite an existing archive.

```bash
python ${GRAYNAV_ROOT}/scripts/archive_surface_depth_e0.py \
  --train-dir /root/autodl-tmp/graynav_surface_depth_run/train \
  --output ${GRAYNAV_RUN_ROOT}/e0/archive
```

## 2. Rebuild and audit prepared-v2

The old prepared directory is not touched. The audit must end with
`GRAYNAV_PREPARED_V2_AUDIT_OK` and report zero train/validation overlap and
zero whole-frame step labels.

```bash
bash ${GRAYNAV_ROOT}/prepare_public_datasets.sh
```

Label audit outputs are in `${GRAYNAV_PREPARED}/audit`. They include class
pixel ratios, step-area distributions, manifest hashes, and separate fixed
label visualizations.

## 3. Fairly evaluate E0 on prepared-v2

```bash
python ${GRAYNAV_ROOT}/scripts/evaluate_surface_depth_checkpoint.py \
  --name E0 \
  --checkpoint ${GRAYNAV_RUN_ROOT}/e0/archive/best.pt \
  --data ${GRAYNAV_PREPARED} \
  --output ${GRAYNAV_RUN_ROOT}/e0/evaluation_prepared_v2.json \
  --batch-size 32 --workers 8 --device cuda
```

This does not retrain E0. It supplies the baseline depth-gradient error needed
by the E2/E3 15 percent improvement gate.

## 4. Run E1 and stop at its gate

```bash
export GRAYNAV_EXPERIMENT=e1
bash ${GRAYNAV_ROOT}/run_surface_depth_cloud.sh
```

Inspect:

```bash
cat ${GRAYNAV_RUN_ROOT}/e1/experiment_summary.json
tensorboard --logdir ${GRAYNAV_RUN_ROOT}/e1/tensorboard --host 0.0.0.0 --port 6006
```

Do not start E2 unless `gate_ever_passed` is true. If false, use the fixed
visualizations and safety metrics to correct labels or sampling rather than
changing the model structure.

## 5. Run E2 only after E1 passes

```bash
export GRAYNAV_EXPERIMENT=e2
bash ${GRAYNAV_ROOT}/run_surface_depth_cloud.sh
```

If E2 passes every final check and its fixed depth boundaries are useful, it is
the candidate and E3 is skipped.

## 6. Run E3 only when E2 justifies it

```bash
export GRAYNAV_EXPERIMENT=e3
bash ${GRAYNAV_ROOT}/run_surface_depth_cloud.sh
```

E3 creates a separate random-weight ONNX static audit before training. It uses
the same official single-channel Fast-SCNN initialization and seed 42.

## 7. Compare and return artifacts

Evaluate each chosen checkpoint with `evaluate_surface_depth_checkpoint.py`,
then generate the comparison:

```bash
python ${GRAYNAV_ROOT}/scripts/compare_surface_depth_experiments.py \
  --result E0=${GRAYNAV_RUN_ROOT}/e0/evaluation_prepared_v2.json \
  --result E1=${GRAYNAV_RUN_ROOT}/e1/candidate_evaluation.json \
  --result E2=${GRAYNAV_RUN_ROOT}/e2/candidate_evaluation.json \
  --output ${GRAYNAV_RUN_ROOT}/comparison/E0_E1_E2
```

Add E3 only if it was actually run. Each experiment contains configuration,
history, TensorBoard events, five checkpoint roles (with `best_overall.pt`
only when gates pass), hashes, source-separated metrics, and fixed separate
visualizations.

After a winner is explicitly selected, export only that checkpoint with
`export_graynav_surface_depth.py`, run the local A1 audit and ONNX consistency,
create 160 calibration plus 40 evaluation arrays, and submit that one model to
the official A1 converter. No `.m1model` conversion belongs in E0-E3 training.
