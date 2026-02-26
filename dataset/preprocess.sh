#!/bin/bash
set -e

DATA_SRC="./data"
REPLICA_DST="./Replica"
TUM_DST="./TUM"

echo "=========================================="
echo "    GSO-SLAM Data Preprocessing Start"
echo "=========================================="

echo "[1/2] Processing Replica Datasets..."
REPLICA_CAM_SRC="${DATA_SRC}/Replica/camera.txt"

if [ -f "$REPLICA_CAM_SRC" ]; then
    for scene in room{0..2} office{0..4}
    do
        TARGET_DIR="${REPLICA_DST}/${scene}"
        mkdir -p "$TARGET_DIR"
        cp "$REPLICA_CAM_SRC" "$TARGET_DIR/"
        echo "  -> Copied camera.txt to $TARGET_DIR"
    done
else
    echo "  [Error] Replica camera.txt not found at $REPLICA_CAM_SRC"
fi

echo ""

echo "[2/2] Processing TUM Datasets..."
TUM_NAMES=("freiburg1_desk" "freiburg2_xyz" "freiburg3_long_office_household")

for name in "${TUM_NAMES[@]}"
do
    DIR_NAME="rgbd_dataset_${name}"
    SRC_DIR="${DATA_SRC}/TUM/${DIR_NAME}"
    DST_DIR="${TUM_DST}/${DIR_NAME}"
    
    mkdir -p "$DST_DIR"
    
    if [ -f "${SRC_DIR}/camera.txt" ]; then
        cp "${SRC_DIR}/camera.txt" "${DST_DIR}/"
        echo "  -> [$name] Copied camera.txt"
    fi
    
    ASSOCIATION_FILE="tum_${name}.txt"
    if [ -f "${SRC_DIR}/${ASSOCIATION_FILE}" ]; then
        cp "${SRC_DIR}/${ASSOCIATION_FILE}" "${DST_DIR}/"
        echo "  -> [$name] Copied $ASSOCIATION_FILE"
    else
        echo "  -> [$name] [Warning] $ASSOCIATION_FILE not found"
    fi
done

echo "=========================================="
echo "    Preprocessing Completed Successfully!"
echo "=========================================="