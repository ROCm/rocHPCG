#!/bin/bash
#
# run_hpcg.sh : run hpcg built for MPI and OpenMP target offload with AOMP compiler
#

HPCG_BIN_DIR=${HPCG_BIN_DIR:-$HOME/git/libs/rocHPCG/omptarget/hpcg/build/bin}

# Requires AOMP (LLVM) compiler, openmpi, libelf-dev
AOMP=${AOMP:-/usr/lib/aomp}
MPI=${MPI:-$HOME/local/openmpi}
LIBELF=${LIBELF:-/usr/lib/x86_64-linux-gnu}
export LD_LIBRARY_PATH=$LIBELF
export GPURUN_VERBOSE=0

# The hpcg.dat file in directory $HPCG_BIN_DIR will be the input to xhpcg
cd $HPCG_BIN_DIR
echo "================================================================="
echo "=== Input file: $HPCG_BIN_DIR/hpcg.dat ==="
cat  $HPCG_BIN_DIR/hpcg.dat
echo "================================================================="
echo $MPI/bin/mpirun -np 8 --mca btl_openib_warn_no_device_params_found 0 $AOMP/bin/gpurun $HPCG_BIN_DIR/xhpcg
$MPI/bin/mpirun -np 8 --mca btl_openib_warn_no_device_params_found 0 $AOMP/bin/gpurun $HPCG_BIN_DIR/xhpcg
