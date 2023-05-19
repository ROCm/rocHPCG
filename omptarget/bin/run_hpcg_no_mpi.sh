#!/bin/bash
#
# run_hpcg.sh : run hpcg with OpenMP target offload with AOMP compiler
#

HPCG_BIN_DIR=${HPCG_BIN_DIR:-$HOME/git/libs/rocHPCG/omptarget/hpcg/build/bin}

# The hpcg.dat file in directory $HPCG_BIN_DIR will be the input to xhpcg
cd $HPCG_BIN_DIR
echo "================================================================="
echo "=== Input file: $HPCG_BIN_DIR/hpcg.dat ==="
cat  $HPCG_BIN_DIR/hpcg.dat
echo "================================================================="
echo $HPCG_BIN_DIR/xhpcg
$HPCG_BIN_DIR/xhpcg
