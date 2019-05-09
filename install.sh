#!/usr/bin/env bash
# Author: Nico Trost

#set -x #echo on

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "rocHPCG build helper script"
  echo "./install [-h|--help] "
  echo "    [-h|--help] prints this help message"
  echo "    [-i|--install] install after build"
  echo "    [-d|--dependencies] install dependencies"
  echo "    [-r|--reference] reference mode"
  echo "    [-g|--debug] -DCMAKE_BUILD_TYPE=Debug (default: Release)"
  echo "    [-t]--test] build single GPU test"
  echo "    [--with-mpi] compile with MPI support (default: enabled)"
  echo "    [--with-openmp] compile with OpenMP support (default: enabled)"
  echo "    [--with-memmgmt] compile with smart memory management (default: enabled)"
  echo "    [--with-memdefrag] compile with memory defragmentation (defaut: enabled)"
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
# true is a system command that completes successfully, function returns success
# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    ubuntu|centos|rhel|fedora)
        true
        ;;
    *)  printf "This script is currently supported on Ubuntu, CentOS, RHEL and Fedora\n"
        exit 2
        ;;
  esac
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
check_exit_code( )
{
  if (( $? != 0 )); then
    exit $?
  fi
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
    check_exit_code
  else
    $@
    check_exit_code
  fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are not already installed
install_apt_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root apt install -y --no-install-recommends ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root yum -y --nogpgcheck install ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dnf list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root dnf install -y ${package}
    fi
  done
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
install_packages( )
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  # dependencies needed for executable to build
  local library_dependencies_ubuntu=( "make" "hip_hcc" "pkg-config" "libnuma1" "rocrand" )
  local library_dependencies_centos=( "epel-release" "make" "cmake3" "hip_hcc" "gcc-c++" "rpm-build" "numactl-libs" "rocrand" )
  local library_dependencies_fedora=( "make" "cmake" "hip_hcc" "gcc-c++" "libcxx-devel" "rpm-build" "numactl-libs" "rocrand" )

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"

      ;;

    centos|rhel)
#     yum -y update brings *all* installed packages up to date
#     without seeking user approval
#     elevate_if_not_root yum -y update
      install_yum_packages "${library_dependencies_centos[@]}"

      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"

      ;;
    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
else
  echo "This script depends on the /etc/os-release file"
  exit 2
fi

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
install_package=false
install_dependencies=false
install_prefix=rochpcg-install
build_release=true
build_reference=false
build_test=false
with_mpi=ON
with_omp=ON
with_memmgmt=ON
with_memdefrag=ON

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,dependencies,reference,debug,test,with-mpi:,with-openmp:,with-memmgmt:,with-memdefrag: --options hidrgt -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -i|--install)
        install_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -r|--reference)
        build_reference=true
        shift ;;
    -g|--debug)
        build_release=false
        shift ;;
    -t|--test)
        build_test=true
        shift ;;
    --with-mpi)
        with_mpi=${2}
        shift 2 ;;
    --with-openmp)
        with_omp=${2}
        shift 2 ;;
    --with-memmgmt)
        with_memmgmt=${2}
        shift 2 ;;
    --with-memdefrag)
        with_memdefrag=${2}
        shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

build_dir=./build
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
if [[ "${build_release}" == true ]]; then
  rm -rf ${build_dir}/release
else
  rm -rf ${build_dir}/debug
fi

# Default cmake executable is called cmake
cmake_executable=cmake

case "${ID}" in
  centos|rhel)
  cmake_executable=cmake3
  ;;
esac

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then

  install_packages

fi

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
export PATH=${PATH}:/opt/rocm/bin

pushd .
  # #################################################
  # configure & build
  # #################################################
  cmake_common_options="-DHPCG_MPI=${with_mpi} -DHPCG_OPENMP=${with_omp}"
  cmake_common_options="${cmake_common_options} -DOPT_MEMMGMT=${with_memmgmt} -DOPT_DEFRAG=${with_memdefrag}"

  # build type
  if [[ "${build_release}" == true ]]; then
    mkdir -p ${build_dir}/release && cd ${build_dir}/release
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"
  else
    mkdir -p ${build_dir}/debug && cd ${build_dir}/debug
    cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Debug"
  fi

  # reference mode
  if [[ "${build_reference}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DHPCG_REFERENCE=ON"
  fi

  # build test
  if [[ "${build_test}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DBUILD_TEST=ON"
  fi

  # Build library with AMD toolchain because of existense of device kernels
  ${cmake_executable} ${cmake_common_options} -DCMAKE_INSTALL_PREFIX=${install_prefix} ../..
  check_exit_code

  if [[ "${build_test}" == false ]]; then
    make -j$(nproc) install
  else
    make -j$(nproc)
  fi
  check_exit_code

  # #################################################
  # install
  # #################################################
  # installing through package manager, which makes uninstalling easy
  if [[ "${install_package}" == true ]]; then
    if [[ "${build_test}" == false ]]; then
      make package
      check_exit_code

      case "${ID}" in
        ubuntu)
          elevate_if_not_root dpkg -i rochpcg-*.deb
        ;;
        centos|rhel)
          elevate_if_not_root yum -y localinstall rochpcg-*.rpm
        ;;
        fedora)
          elevate_if_not_root dnf install rochpcg-*.rpm
        ;;
      esac
    fi
  fi
popd
