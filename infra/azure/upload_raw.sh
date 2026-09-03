#!/usr/bin/env bash
# Upload the raw Home Credit CSVs from data/raw/ to the ADLS Gen2 container
# under the raw/ prefix. Uses your Azure AD login (no account keys).
#
# Usage: bash infra/azure/upload_raw.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/.env.azure"

echo "Uploading $REPO/data/raw/*.csv -> $AZ_STORAGE/$AZ_CONTAINER/raw/"
az storage blob upload-batch \
  --account-name "$AZ_STORAGE" --auth-mode login \
  --destination "$AZ_CONTAINER" --destination-path raw \
  --source "$REPO/data/raw" --pattern "*.csv" \
  --overwrite true --max-connections 8 -o table

echo
az storage blob list --account-name "$AZ_STORAGE" --auth-mode login \
  --container-name "$AZ_CONTAINER" --prefix raw/ \
  --query "[].{name:name, MB:properties.contentLength}" -o table
