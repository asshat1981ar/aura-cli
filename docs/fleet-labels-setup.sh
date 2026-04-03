#!/usr/bin/env bash
# chmod +x docs/fleet-labels-setup.sh
set -euo pipefail

gh label create "fleet:trigger"     --color "0075ca" --description "Trigger fleet dispatcher"          || echo "label fleet:trigger already exists, skipping"
gh label create "fleet:in-progress" --color "e4e669" --description "Fleet run in progress"              || echo "label fleet:in-progress already exists, skipping"
gh label create "fleet:done"        --color "0e8a16" --description "Fleet run completed"                || echo "label fleet:done already exists, skipping"
gh label create "fleet:blocked"     --color "d93f0b" --description "Fleet run blocked; needs human"     || echo "label fleet:blocked already exists, skipping"

echo "✅ All fleet labels created (or already present)."
