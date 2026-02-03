set -e

SCENARIO=${1:-medium}
KERNEL_FILTER=${2:-}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./results"
mkdir -p "$OUTPUT_DIR"

# You can pass the all param to also profile non-custom kernels
if [ -z "$KERNEL_FILTER" ]; then
    KERNEL_FILTER="regex:(project_gaussians|duplicate_gaussians|identify_tile_ranges|render_gaussians)"
    FILTER_DESC="Gaussian splatting kernels only"
elif [ "$KERNEL_FILTER" = "all" ]; then
    KERNEL_FILTER=""
    FILTER_DESC="all kernels (including PyTorch)"
else
    FILTER_DESC="$KERNEL_FILTER"
fi

echo "=== NSight Compute Profiling ==="
echo "Scenario: $SCENARIO"
echo "Kernel filter: $FILTER_DESC"
echo "Output directory: $OUTPUT_DIR"
echo ""

NCU_CMD="ncu --set full"

if [ -n "$KERNEL_FILTER" ]; then
    NCU_CMD="$NCU_CMD --kernel-name $KERNEL_FILTER"
fi

# Output file
OUTPUT_FILE="${OUTPUT_DIR}/ncu_${SCENARIO}_${TIMESTAMP}"
NCU_CMD="$NCU_CMD -o $OUTPUT_FILE"

# Run profiler
echo "Running: $NCU_CMD python profile_forward.py --scenario $SCENARIO"
echo ""
$NCU_CMD python profile_forward.py --scenario $SCENARIO

echo ""
echo "=== Profiling Complete ==="
echo "Report saved to: ${OUTPUT_FILE}.ncu-rep"
echo ""
echo "To view the report:"
echo "  ncu-ui ${OUTPUT_FILE}.ncu-rep"
echo ""
echo "Or export to CSV:"
echo "  ncu --import ${OUTPUT_FILE}.ncu-rep --csv > ${OUTPUT_FILE}.csv"
