#!/bin/bash
set -e

TUM_DIR="../dataset/TUM"
RESULT_DIR="./results/test"
OUTPUT_TXT="${RESULT_DIR}/tum_results.txt"

mkdir -p "${RESULT_DIR}"

echo "--- TUM Dataset Evaluation Start ---" > "${OUTPUT_TXT}"

for dataset_name in "freiburg1_desk" "freiburg2_xyz" "freiburg3_long_office_household"
do
    DATASET_DIR="${TUM_DIR}/rgbd_dataset_${dataset_name}"
    SAVE_PATH="${RESULT_DIR}/tum_${dataset_name}"
    
    mkdir -p "${SAVE_PATH}"

    echo "========================================"
    echo "Processing: TUM ${dataset_name}"
    echo "TUM ${dataset_name}" >> "${OUTPUT_TXT}"

    ../build/bin/dso_dataset \
        files="${DATASET_DIR}/rgb" \
        calib="${DATASET_DIR}/camera.txt" \
        gamma="${DATASET_DIR}/pcalib.txt" \
        vignette="${DATASET_DIR}/vignette.png" \
        dataassociation="${DATASET_DIR}/rgb.txt" \
        preset=0 \
        mode=1 \
        quiet=1 \
        nogui=1 \
        cfg_yaml="${HOME}/GSO_SLAM/cfg/gaussian_mapper/Monocular/TUM/tum_${dataset_name}.yaml" \
        save_dir="${SAVE_PATH}" \
        use_gaussian_viewer=0

    echo "Evaluating: ${dataset_name}"
    python3 -W ignore scripts/evaluate_ate_scale_tum.py \
        "${DATASET_DIR}" \
        "${SAVE_PATH}" \
        "${DATASET_DIR}/rgb.txt" >> "${OUTPUT_TXT}"

    echo "Done ${dataset_name}"
    echo "--------------------------" >> "${OUTPUT_TXT}"
done

echo "All TUM experiments finished. Results saved in ${OUTPUT_TXT}"