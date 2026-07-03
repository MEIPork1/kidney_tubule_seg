#!/bin/bash
# 完整 pipeline: ensemble 评估 + 分类评估 + 可视化
set -e

CONDA=/root/miniconda3/bin/conda
ENV=cellpose
CODE=/root/kidney/code
MODELS=/root/kidney/models
DATA=/root/kidney/data/multiclass
RESULTS=/root/kidney/results
mkdir -p $RESULTS

FT=0.3   # Best flow_threshold from fold 0 tuning
CT=-0.5  # Best cellprob_threshold from fold 0 tuning

echo "===== Step 1: Ensemble evaluation (5-fold CV) ====="
$CONDA run --no-capture-output -n $ENV python -u $CODE/eval_ensemble.py \
    --data_dir $DATA/all \
    --model_dir $MODELS/cpsam_v2_fold \
    --n_folds 5 \
    --flow_threshold $FT \
    --cellprob_threshold $CT \
    --out_dir $RESULTS/ensemble \
    2>&1 | tee $RESULTS/ensemble.log

echo "===== Step 2: TTA pipeline with classification ====="
for FOLD in 0 1 2 3 4; do
    echo "--- Fold $FOLD ---"
    $CONDA run --no-capture-output -n $ENV python -u $CODE/predict_tubule.py \
        --img_dir $DATA/fold_${FOLD}/test \
        --seg_model $MODELS/cpsam_v2_fold${FOLD}/best_model \
        --cls_model $MODELS/classifier/cnn_classifier_fold0.pth \
        --out_dir $RESULTS/fold${FOLD}_pipeline \
        --flow_threshold $FT \
        --cellprob_threshold $CT \
        2>&1 | tee $RESULTS/fold${FOLD}_pipeline.log
done

echo "===== All done! Results in $RESULTS ====="
