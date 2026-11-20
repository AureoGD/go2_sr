#!/usr/bin/env bash
# Build the C++ RGC-MPC module (_rgc_mpc_cpp) and its pinned dependencies.
#
#   control/self_righting/rgc_mpc_cpp/build.sh            # build (deps only once)
#   BLASFEO_TARGET=ARMV8A_ARM_CORTEX_A57 HPIPM_TARGET=GENERIC ./build.sh   # e.g. robot (ARM)
#   CLEAN=1 ./build.sh                                     # rebuild everything
#
# Output: control/self_righting/rgc_mpc_cpp/_rgc_mpc_cpp*.so (imported by the Python package)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
TP="$HERE/third_party"
SRC="$TP/src"
INST="$TP/install"
BUILD="$HERE/build"

BLASFEO_TAG="0.1.4.3"
HPIPM_TAG="0.1.4"
BLASFEO_TARGET="${BLASFEO_TARGET:-X64_INTEL_HASWELL}"   # AVX2+FMA; see blasfeo/CMakeLists.txt for others
HPIPM_TARGET="${HPIPM_TARGET:-AVX}"                     # AVX | GENERIC
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
JOBS="${JOBS:-$(nproc)}"

# The system pinocchio paths in ~/.bashrc must not leak into the build or the tests
unset PYTHONPATH LD_LIBRARY_PATH

if [[ "${CLEAN:-0}" == "1" ]]; then
    rm -rf "$TP/build" "$INST" "$BUILD"
fi

mkdir -p "$SRC"
[[ -d "$SRC/blasfeo" ]] || git clone -q --depth 1 --branch "$BLASFEO_TAG" https://github.com/giaf/blasfeo.git "$SRC/blasfeo"
[[ -d "$SRC/hpipm" ]] || git clone -q --depth 1 --branch "$HPIPM_TAG" https://github.com/giaf/hpipm.git "$SRC/hpipm"

# ---------------------------------------------------------------- BLASFEO (static, PIC)
if [[ ! -f "$INST/lib/libblasfeo.a" ]]; then
    cmake -S "$SRC/blasfeo" -B "$TP/build/blasfeo" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTARGET="$BLASFEO_TARGET" \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_ASM_FLAGS="-fPIC" \
        -DBLASFEO_EXAMPLES=OFF -DBLASFEO_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX="$INST" > "$TP/blasfeo_configure.log"
    cmake --build "$TP/build/blasfeo" -j "$JOBS" > "$TP/blasfeo_build.log"
    cmake --install "$TP/build/blasfeo" > /dev/null
fi

# ---------------------------------------------------------------- HPIPM (static, PIC)
if [[ ! -f "$INST/lib/libhpipm.a" ]]; then
    cmake -S "$SRC/hpipm" -B "$TP/build/hpipm" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTARGET="$HPIPM_TARGET" \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DHPIPM_TESTING=OFF \
        -DBLASFEO_PATH="$INST" \
        -DCMAKE_INSTALL_PREFIX="$INST" > "$TP/hpipm_configure.log"
    cmake --build "$TP/build/hpipm" -j "$JOBS" > "$TP/hpipm_build.log"
    cmake --install "$TP/build/hpipm" > /dev/null
fi

# ---------------------------------------------------------------- Python module
PYBIND11_DIR="$("$PYTHON" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake -S "$HERE" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$PYTHON" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DRGC_DEPS_PREFIX="$INST" > "$BUILD.configure.log" 2>&1 || { cat "$BUILD.configure.log"; exit 1; }
cmake --build "$BUILD" -j "$JOBS"

echo "[rgc_mpc_cpp] built: $(ls "$HERE"/_rgc_mpc_cpp*.so)"
