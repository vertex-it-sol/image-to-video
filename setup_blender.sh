#!/usr/bin/env bash
# Download and extract Blender 5.2 LTS locally (no sudo, no apt).
#
# We use the official tarball rather than `pip install bpy` (whose wheels require
# Python 3.13 — this project's venv is 3.10) or apt (jammy ships Blender 3.0.1
# from 2022, which predates EEVEE Next and Blackwell OptiX kernels).
#
# Blender bundles its own Python, so blender_turntable.py cannot import anything
# from the venv. The venv only orchestrates via subprocess.

set -euo pipefail

VERSION="5.2.0"
SERIES="Blender5.2"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/blender"
TARBALL="blender-${VERSION}-linux-x64.tar.xz"
URL="https://download.blender.org/release/${SERIES}/${TARBALL}"

if [ -x "${DEST}/blender" ]; then
    echo "Blender already present at ${DEST}/blender"
    "${DEST}/blender" --version | head -1
    exit 0
fi

mkdir -p "$(dirname "${DEST}")"
cd "$(dirname "${DEST}")"

echo "Downloading ${URL}"
curl -fL --retry 3 -o "/tmp/${TARBALL}" "${URL}"

echo "Extracting..."
rm -rf "${DEST}" "blender-${VERSION}-linux-x64"
tar -xJf "/tmp/${TARBALL}"
mv "blender-${VERSION}-linux-x64" "${DEST}"
rm -f "/tmp/${TARBALL}"

echo "Verifying..."
"${DEST}/blender" --version | head -1

# Report any unsatisfied shared libraries. All 16 that Blender needs were already
# present on this box, so this is a guard against a different environment.
if ldd "${DEST}/blender" 2>/dev/null | grep -q "not found"; then
    echo "WARNING: missing shared libraries:"
    ldd "${DEST}/blender" | grep "not found"
    exit 1
fi

echo "Blender ready at ${DEST}/blender"
