#!/bin/bash
set -e

DATA_PATH=${1:-../../data_1080p}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results"
mkdir -p "$OUTPUT_DIR"

KERNEL_FILTER="regex:(project_gaussians|duplicate_gaussians|identify_tile_ranges|render_gaussians)"
FILTER_DESC="All custom Gaussian splatting kernels"

echo "=== NSight Compute Profiling with Real Data ==="
echo "Data path: $DATA_PATH"
echo "Kernel filter: $FILTER_DESC"
echo "Output directory: $OUTPUT_DIR"
echo ""

OUTPUT_FILE="${OUTPUT_DIR}/ncu_real_data_${TIMESTAMP}"
NCU_CMD="ncu --set full --kernel-name $KERNEL_FILTER -o $OUTPUT_FILE"

echo "Running: $NCU_CMD python profile_forward_with_data.py --data-path $DATA_PATH"
echo ""
$NCU_CMD python profile_forward_with_data.py --data-path $DATA_PATH

echo ""
echo "=== Profiling Complete ==="
echo "Report saved to: ${OUTPUT_FILE}.ncu-rep"
echo ""
echo "To view the report:"
echo "  ncu-ui ${OUTPUT_FILE}.ncu-rep"
