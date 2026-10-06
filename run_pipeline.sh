#!/usr/bin/env bash
set -e

echo "1. Checking duplicate images in PCOSGen training set..."
python src/dedup_check.py \
    --data_dir data/train/images \
    --hash_diff_threshold 0 \
    --out results/dup_t0.csv

echo "2. Running duplicate cluster analysis..."
python analyze_duplicates.py

echo "3. Training ResNet-18 with Duplicate-Aware Group Split..."
python src/train.py \
    --image_dir data/train/images \
    --label_file data/train/class_label.xlsx \
    --group_csv results/dup_t0.csv \
    --split_mode group \
    --epochs 15 \
    --out_dir checkpoints/group

echo "4. Training ResNet-18 with Naive Random Split..."
python src/train.py \
    --image_dir data/train/images \
    --label_file data/train/class_label.xlsx \
    --group_csv results/dup_t0.csv \
    --split_mode random \
    --epochs 15 \
    --out_dir checkpoints/random

echo "5. Evaluating In-Distribution Faithfulness (Grad-CAM vs Random)..."
python src/faithfulness.py \
    --image_dir data/train/images \
    --label_file data/train/class_label.xlsx \
    --split test \
    --checkpoint checkpoints/group/best_model.pt \
    --test_split_paths checkpoints/group/test_split_paths.json \
    --out results/faithfulness_indist.csv

echo "6. Auditing Cross-Split Contamination (Train vs Official Test)..."
python src/cross_dataset_dedup.py \
    --dir_a data/train/images --label_a train \
    --dir_b data/test/images --label_b test \
    --hash_diff_threshold 0 \
    --out results/train_vs_test_duplicates.csv

echo "7. Evaluating Clean PCOSGen Test Set..."
python src/evaluate_external.py \
    --image_dir data/test/images \
    --label_file "data/test/class label.csv" \
    --checkpoint checkpoints/group/best_model.pt \
    --out results/pcosgen_test_eval_clean.csv \
    --exclude_csv results/train_vs_test_duplicates.csv \
    --exclude_col test_path

echo "8. Auditing Cross-Dataset Contamination (PCOSGen vs Kaggle)..."
python src/cross_dataset_dedup.py \
    --dir_a data/train/images --label_a pcosgen_train \
    --dir_b data/external/data/test --label_b kaggle_ext \
    --hash_diff_threshold 0 \
    --out results/pcosgen_vs_kaggle_duplicates.csv

echo "9. Evaluating Clean External Kaggle Dataset..."
python src/evaluate_external.py \
    --image_dir data/external/data/test \
    --positive_class_dir infected \
    --negative_class_dir notinfected \
    --checkpoint checkpoints/group/best_model.pt \
    --out results/kaggle_external_eval_clean.csv \
    --exclude_csv results/pcosgen_vs_kaggle_duplicates.csv \
    --exclude_col kaggle_ext_path

echo "=== Audit Completed Successfully ==="