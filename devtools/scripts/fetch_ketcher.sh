#!/usr/bin/env bash
set -euo pipefail
VERSION="${1:-3.18.0}"   
DEST="$(dirname "$0")/../openmmdl/openmmdl_setup/static/ketcher"
TMP="$(mktemp -d)"

curl -fL -o "$TMP/k.zip" \
  "https://github.com/epam/ketcher/releases/download/v${VERSION}/ketcher-standalone-${VERSION}.zip"
unzip -q "$TMP/k.zip" -d "$TMP/x"

rm -rf "$DEST"; mkdir -p "$DEST"
SRC="$(dirname "$(find "$TMP/x" -name index.html | head -1)")"
cp -r "$SRC"/. "$DEST"/
rm -rf "$TMP"
echo "Ketcher $VERSION -> $DEST"