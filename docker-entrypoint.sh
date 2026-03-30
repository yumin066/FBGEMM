#!/usr/bin/env bash
set -euo pipefail

# Build-time RUN cannot mount /proc. Do it at container startup.
# Ignore failures so container can still start in restricted environments.
sudo umount -l /proc || true
sudo mount -t proc proc /proc || true

exec "$@"
