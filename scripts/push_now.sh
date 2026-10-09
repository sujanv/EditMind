#!/usr/bin/env bash
# Push all remaining commits immediately to origin main
set -e

echo "[EditMind] Pushing all commits immediately to origin main..."
git push origin main
echo "[EditMind] All commits pushed successfully!"
