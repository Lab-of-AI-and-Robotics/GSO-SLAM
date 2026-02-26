REPLICA_DIR="../dataset/Replica"

RESULT_DIR="./results/test"
OUTPUT_TXT=${RESULT_DIR}/replica_results.txt

echo "" > ${OUTPUT_TXT}

for dataset_name in "room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4"
do
    DATASET_DIR=${REPLICA_DIR}/${dataset_name}
    SAVE_DIR=${RESULT_DIR}/replica_${dataset_name}
    
    echo "Replica ${dataset_name}"
    echo "Replica ${dataset_name}" >> ${OUTPUT_TXT}

    python3 -W ignore scripts/eval_rendering.py \
        ${DATASET_DIR}/traj.txt \
        ${RESULT_DIR}/replica_${dataset_name} \
        ${DATASET_DIR} >> ${OUTPUT_TXT}
done