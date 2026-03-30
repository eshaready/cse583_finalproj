#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT THESE IF NEEDED ======
BUILD_DIR="$HOME/speccpu2017/benchspec/CPU/600.perlbench_s/build/build_base_myuse-m64.0000"
PROFILE_DATA="/tmp/perlprof/perlbench.profdata"
PASS_SO="$HOME/profile_pass/DumpEdgeFeaturesPass.so"
OUT_DIR="$HOME/profile_pass/output_perlbench"
CLANG="/usr/local/bin/clang"
OPT="/usr/local/bin/opt"
# ==================================

LL_DIR="$OUT_DIR/ll"
CSV_DIR="$OUT_DIR/csv"
LOG_DIR="$OUT_DIR/logs"

rm -rf "$LL_DIR" "$CSV_DIR" "$LOG_DIR"
mkdir -p "$LL_DIR" "$CSV_DIR" "$LOG_DIR"

cd "$BUILD_DIR"

# Common compile flags for perlbench
COMMON_FLAGS=(
  -std=c99
  -m64
  -emit-llvm
  -S
  -DSPEC
  -DNDEBUG
  -DPERL_CORE
  -I.
  -Idist/IO
  -Icpan/Time-HiRes
  -Icpan/HTML-Parser
  -Iext/re
  -Ispecrand
  -DDOUBLE_SLASHES_SPECIAL=0
  -D_LARGE_FILES
  -D_LARGEFILE_SOURCE
  -D_FILE_OFFSET_BITS=64
  -g
  -O2
  -fprofile-use="$PROFILE_DATA"
  -DSPEC_LINUX_X64
  -fno-strict-aliasing
  -DSPEC_LP64
)

echo "Source file,Status" > "$LOG_DIR/summary.csv"

# Find all C files recursively
find . -name "*.c" | sort | while read -r SRC; do
    REL_NO_EXT="${SRC#./}"
    REL_NO_EXT="${REL_NO_EXT%.c}"

    SAFE_NAME=$(echo "$REL_NO_EXT" | sed 's/\//__/g')
    LL_FILE="$LL_DIR/${SAFE_NAME}.ll"
    CSV_FILE="$CSV_DIR/${SAFE_NAME}.csv"
    ERR_FILE="$LOG_DIR/${SAFE_NAME}.err"

    echo "Processing $SRC"

    # create LLVM IR
    if ! "$CLANG" "${COMMON_FLAGS[@]}" "$SRC" -o "$LL_FILE" >"$ERR_FILE" 2>&1; then
        echo "$SRC,IR_FAIL" >> "$LOG_DIR/summary.csv"
        continue
    fi

    # Skip files with no defined functions
    if ! grep -q '^define ' "$LL_FILE"; then
        echo "$SRC,NO_FUNCTIONS" >> "$LOG_DIR/summary.csv"
        continue
    fi

    # run LLVM pass
    if ! "$OPT" -load-pass-plugin "$PASS_SO" \
         -passes=dump-edge-features \
         "$LL_FILE" -disable-output 2>"$CSV_FILE"; then
        echo "$SRC,PASS_FAIL" >> "$LOG_DIR/summary.csv"
        continue
    fi

    # Check whether CSV has content
    if [ ! -s "$CSV_FILE" ]; then
        echo "$SRC,EMPTY_CSV" >> "$LOG_DIR/summary.csv"
    else
        echo "$SRC,OK" >> "$LOG_DIR/summary.csv"
    fi
done

# Merge CSVs into one file
MERGED="$OUT_DIR/perlbench_all_edges_clean.csv"
HEADER='^func,src_bb,dst_bb,succ_idx,prob_num,prob_den,src_prof_count,dst_prof_count,edge_count_est,src_inst_count,src_mem_ops,src_loop_depth,is_back_edge,dst_loop_depth,dst_is_loop_header$'
FIRST_FILE=$(find "$CSV_DIR" -type f -name "*.csv" | sort | head -n 1 || true)

if [ -n "${FIRST_FILE:-}" ]; then
    head -n 1 "$FIRST_FILE" > "$MERGED"
    grep -h -v "$HEADER" "$CSV_DIR"/*.csv >> "$MERGED"
fi

echo "Done."
echo "Merged CSV: $MERGED"
echo "Summary:    $LOG_DIR/summary.csv"
