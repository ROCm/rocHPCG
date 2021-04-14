# rocHPCG
rocHPCG is a benchmark based on the [HPCG][] benchmark application, implemented on top of AMD's Radeon Open eCosystem Platform [ROCm][] runtime and toolchains. rocHPCG is created using the [HIP][] programming language and optimized for AMD's latest discrete GPUs.

## Requirements
* Git
* CMake (3.10 or later)
* MPI
* NUMA library
* AMD [ROCm] platform (4.1 or later)
* [rocPRIM][]
* googletest (for test application only)

## Quickstart rocHPCG build and install

#### Install script
You can build rocHPCG using the *install.sh* script
```
# Clone rocHPCG using git
git clone https://github.com/ROCmSoftwarePlatform/rocHPCG.git

# Go to rocHPCG directory
cd rocHPCG

# Run install.sh script
# Command line options:
#    -h|--help         - prints this help message
#    -i|--install      - install after build
#    -d|--dependencies - install dependencies
#    -r|--reference    - reference mode
#    -g|--debug        - -DCMAKE_BUILD_TYPE=Debug (default: Release)
#    -t|--test         - build single GPU test
#    --with-rocm=<dir> - Path to ROCm install (default: /opt/rocm)
#    --with-mpi=<dir>  - Path to external MPI install (Default: clone+build OpenMPI v4.0.5 in deps/)
#    --with-openmp     - compile with OpenMP support (default: enabled)
#    --with-memmgmt    - compile with smart memory management (default: enabled)
#    --with-memdefrag  - compile with memory defragmentation (defaut: enabled)
./install.sh -di
```
By default, [UCX] v1.10.0 and [OpenMPI] v4.1.0 will be cloned and build in `rocHPCG/deps`.
After build and install, the `rochpcg` executable is placed in `build/release/rochpcg-install`.

#### MPI
You can build rocHPCG using your own MPI installation by specifying the directory, e.g.
```
./install.sh -di --with-mpi=/my/mpiroot/
```
Alternatively, when you do not pass a specific directory, OpenMPI v4.0.5 with UCX will be cloned and built within `rocHPCG/deps` directory.
If you want to disable MPI, you need to run
```
./install.sh -di --with-mpi=off
```

#### ROCm
You can build rocHPCG with specific ROCm versions by passing the directory to the install script, e.g.
```
./install.sh -di --with-rocm=/my/rocm-x.y.z/
```

## Running rocHPCG benchmark application
You can run the rocHPCG benchmark application by either using command line parameters or the `hpcg.dat` input file
```
rochpcg <nx> <ny> <nz> <runtime>
# where
# nx      - is the global problem size in x dimension
# ny      - is the global problem size in y dimension
# nz      - is the global problem size in z dimension
# runtime - is the desired benchmarking time in seconds (> 1800s for official runs)
```

Similarly, these parameters can be entered into an input file `hpcg.dat` in the working directory, e.g. `nx = ny = nz = 280` and `runtime = 1860`.
```
HPCG benchmark input file
Sandia National Laboratories; University of Tennessee, Knoxville
280 280 280
1860
```

## Performance evaluation
For performance evaluation purposes, the number of iterations should be as low as possible (e.g. convergence rate as high as possible), since the final HPCG score is scaled to 50 iterations.
Furthermore, it is observed that high memory occupancy performs better on AMD devices. Problem size suggestion for devices with 16GB is `nx = ny = nz = 280` and `nx = 560, ny = nz = 280` for devices with 32GB or more. Runtime for official runs have to be at least 1800 seconds (use 1860 to be on the safe side), e.g.
```
./rochpcg 560 280 280 1860
```
Please note that convergence rate behaviour might change in a multi-GPU environment and need to be adjusted accordingly.

Additionally, you can specify the device to be used for the application (e.g. device #1):
```
./rochpcg 560 280 280 1860 --dev=1
```

## Support
Please use [the issue tracker][] for bugs and feature requests.

## License
The [license file][] can be found in the main repository.

[HPCG]: https://www.hpcg-benchmark.org/
[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/
[rocPRIM]: https://github.com/ROCmSoftwarePlatform/rocPRIM
[OpenMPI]: https://github.com/open-mpi/ompi
[UCX]: https://github.com/openucx/ucx
[the issue tracker]: https://github.com/ROCmSoftwarePlatform/rocHPCG/issues
[license file]: https://github.com/ROCmSoftwarePlatform/rocHPCG
