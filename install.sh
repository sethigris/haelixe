#!/bin/bash
set -e
REPO="sethigris/haelixe"
INSTALL_DIR="/usr/local/bin"
BINARY_NAME="haelixe"

echo "Haelixe Deep Learning Engine Installer"
echo "---------------------------------------"

OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM="linux";;
    Darwin*)    PLATFORM="macos";;
    *)          echo "Unsupported OS: ${OS}"; exit 1;;
esac

ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64)     ARCH="x86_64";;
    arm64|aarch64) ARCH="aarch64";;
    *)          echo "Unsupported architecture: ${ARCH}"; exit 1;;
esac

RELEASE_URL="https://github.com/${REPO}/releases/latest/download/axiom-${PLATFORM}-${ARCH}.tar.gz"

echo "Downloading latest Haelixe binary for ${PLATFORM}-${ARCH}..."
TMP_DIR=$(mktemp -d)
curl -fsSL "$RELEASE_URL" -o "${TMP_DIR}/axiom.tar.gz" || { echo "Failed to download binary. Did the GitHub Action finish?"; exit 1; }

echo "Extracting..."
tar -xzf "${TMP_DIR}/axiom.tar.gz" -C "${TMP_DIR}"

echo "Installing to ${INSTALL_DIR} (requires sudo)..."
sudo mv "${TMP_DIR}/${BINARY_NAME}" "${INSTALL_DIR}/${BINARY_NAME}"
sudo chmod +x "${INSTALL_DIR}/${BINARY_NAME}"

rm -rf "${TMP_DIR}"

echo ""
echo "Haelixe successfully installed!"
echo "Run 'axiom' anywhere in your terminal to get started."
