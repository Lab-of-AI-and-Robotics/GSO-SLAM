#!/bin/bash
set -e

REPLICA_DIR="../dataset/Replica"
RESULT_DIR="./results/test"
OUTPUT_TXT="${RESULT_DIR}/replica_results.txt"

mkdir -p "${RESULT_DIR}"

echo "--- Replica Dataset Evaluation Start ---" > "${OUTPUT_TXT}"

for dataset_name in "room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4"
do
    DATASET_DIR="${REPLICA_DIR}/${dataset_name}"
    SAVE_PATH="${RESULT_DIR}/replica_${dataset_name}"
    
    mkdir -p "${SAVE_PATH}"

    echo "========================================"
    echo "Processing: Replica ${dataset_name}"
    echo "Replica ${dataset_name}" >> "${OUTPUT_TXT}"

    ../build/bin/dso_dataset \
        files="${DATASET_DIR}/results" \
        calib="${DATASET_DIR}/camera.txt" \
        gamma="${DATASET_DIR}/pcalib.txt" \
        vignette="${DATASET_DIR}/vignette.png" \
        preset=0 \
        mode=2 \
        quiet=1 \
        nogui=1 \
        which_dataset=replica \
        cfg_yaml="${HOME}/GSO_SLAM/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml" \
        save_dir="${SAVE_PATH}" \
        use_gaussian_viewer=0

    echo "Evaluating ATE for ${dataset_name}..."
    python3 -W ignore scripts/evaluate_ate_scale_replica.py \
        "${DATASET_DIR}/traj.txt" \
        "${SAVE_PATH}" >> "${OUTPUT_TXT}"

    echo "Done: ${dataset_name}"
    echo "----------------------------------------" >> "${OUTPUT_TXT}"
done

echo "All Replica experiments finished. Results saved in ${OUTPUT_TXT}"