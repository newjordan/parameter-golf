#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SETUP_SYS_DEPS="${SETUP_SYS_DEPS:-1}"
INSTALL_RUSTUP="${INSTALL_RUSTUP:-1}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-auto}"

RUSTUP_PROFILE="${RUSTUP_PROFILE:-minimal}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-stable}"

SO_PATH="${SO_PATH:-${REPO_ROOT}/Nitrust/rust/target/release/libnitrust_py.so}"
DATA_GLOB="${DATA_GLOB:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin}"

usage() {
  cat <<USAGE
Usage: bash Nitrust/scripts/setup_nitrust.sh

Environment toggles:
  SETUP_SYS_DEPS=1     Install apt packages (curl/build-essential/pkg-config/libssl-dev/ca-certificates)
  INSTALL_RUSTUP=1     Install rustup/cargo if cargo is missing
  RUN_BUILD=1          Build nitrust rust bridge (nitrust-py)
  RUN_PREFLIGHT=1      Force preflight and fail if dataset shards are missing
  RUN_PREFLIGHT=auto   Run preflight only when DATA_GLOB exists (default)
  RUN_PREFLIGHT=0      Skip preflight

Optional overrides:
  SO_PATH=/abs/path/libnitrust_py.so
  DATA_GLOB=/path/to/fineweb_train_*.bin
  RUSTUP_PROFILE=minimal
  RUST_TOOLCHAIN=stable
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

log() {
  echo "[nitrust-setup] $*"
}

warn() {
  echo "[nitrust-setup] WARN: $*"
}

die() {
  echo "[nitrust-setup] FATAL: $*" >&2
  exit 1
}

run_privileged() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  return 1
}

activate_cargo_env() {
  if [ -f "${HOME}/.cargo/env" ]; then
    # shellcheck disable=SC1090
    source "${HOME}/.cargo/env"
  fi
  if [ -d "${HOME}/.cargo/bin" ] && [[ ":${PATH}:" != *":${HOME}/.cargo/bin:"* ]]; then
    export PATH="${HOME}/.cargo/bin:${PATH}"
  fi
}

install_sys_deps() {
  if [ "${SETUP_SYS_DEPS}" != "1" ]; then
    log "system deps install skipped (SETUP_SYS_DEPS=${SETUP_SYS_DEPS})"
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not found; skipping system package install"
    return
  fi
  log "installing system deps via apt-get"
  if ! run_privileged apt-get update; then
    warn "could not run apt-get update (need root/sudo); continuing"
    return
  fi
  if ! run_privileged apt-get install -y curl build-essential pkg-config libssl-dev ca-certificates; then
    warn "could not install apt packages; continuing"
  fi
}

install_rustup_if_needed() {
  activate_cargo_env
  if command -v cargo >/dev/null 2>&1; then
    log "cargo found: $(cargo --version)"
    return
  fi

  if [ "${INSTALL_RUSTUP}" != "1" ]; then
    die "cargo missing and INSTALL_RUSTUP=${INSTALL_RUSTUP}"
  fi
  if ! command -v curl >/dev/null 2>&1; then
    die "curl missing; set SETUP_SYS_DEPS=1 or install curl manually"
  fi

  log "installing rustup (profile=${RUSTUP_PROFILE}, toolchain=${RUST_TOOLCHAIN})"
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile "${RUSTUP_PROFILE}" --default-toolchain "${RUST_TOOLCHAIN}"

  activate_cargo_env
  command -v cargo >/dev/null 2>&1 || die "cargo still missing after rustup install"
  log "cargo ready: $(cargo --version)"
}

run_build() {
  if [ "${RUN_BUILD}" != "1" ]; then
    log "rust build skipped (RUN_BUILD=${RUN_BUILD})"
    return
  fi

  [ -f "${REPO_ROOT}/Nitrust/rust/Cargo.toml" ] || die "missing Cargo.toml at Nitrust/rust/Cargo.toml"
  [ -f "${REPO_ROOT}/Nitrust/scripts/build_nitrust_py.sh" ] || die "missing build helper script"

  log "building nitrust rust bridge"
  SO_PATH="${SO_PATH}" "${REPO_ROOT}/Nitrust/scripts/build_nitrust_py.sh"

  [ -f "${SO_PATH}" ] || die "build completed but shared object missing at ${SO_PATH}"
}

run_preflight() {
  if [ "${RUN_PREFLIGHT}" = "0" ]; then
    log "preflight skipped (RUN_PREFLIGHT=0)"
    return
  fi

  local preflight="${REPO_ROOT}/Nitrust/scripts/medusa_nitrust_preflight.sh"
  [ -f "${preflight}" ] || die "missing preflight script at ${preflight}"

  if [ "${RUN_PREFLIGHT}" = "auto" ] && ! compgen -G "${DATA_GLOB}" >/dev/null; then
    warn "preflight skipped: no shards match DATA_GLOB=${DATA_GLOB}"
    warn "next: bootstrap data, then rerun setup with RUN_PREFLIGHT=1"
    return
  fi
  if [ "${RUN_PREFLIGHT}" != "1" ] && [ "${RUN_PREFLIGHT}" != "auto" ]; then
    die "invalid RUN_PREFLIGHT=${RUN_PREFLIGHT} (expected 0, auto, or 1)"
  fi

  log "running nitrust preflight"
  SO_PATH="${SO_PATH}" DATA_GLOB="${DATA_GLOB}" "${preflight}"
}

main() {
  log "repo_root=${REPO_ROOT}"
  install_sys_deps
  install_rustup_if_needed
  run_build
  run_preflight

  echo "============================================"
  echo "NITRUST SETUP COMPLETE"
  echo "SO_PATH=${SO_PATH}"
  echo "next: bash Nitrust/scripts/run_ab_fixedstep_1gpu.sh"
  echo "============================================"
}

main "$@"
