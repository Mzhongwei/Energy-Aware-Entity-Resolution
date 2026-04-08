#!/usr/bin/env sh
# Extract mode from config file and output it to stdout.
# Also copies the config file to /tmp/config.yaml.
# Usage: extract-config-mode.sh <fallback_mode>

set -eu

FALLBACK_MODE="${1:-default}"
CONFIG_SRC=""

# Try predefined config paths in order
for p in /app/config/examples/config-embedding.yaml /app/examples/config.yaml; do
  if [ -f "$p" ]; then
    CONFIG_SRC="$p"
    break
  fi
done

# Copy config to standard location
if [ -n "$CONFIG_SRC" ]; then
  cp "$CONFIG_SRC" /tmp/config.yaml
else
  printf 'mode: "%s"\n' "$FALLBACK_MODE" > /tmp/config.yaml
fi

# Extract mode from config file
MODE_FROM_CONFIG="$(sed -n 's/^mode:[[:space:]]*"\?\([^"#]*\)"\?.*/\1/p' /tmp/config.yaml | head -n1 | tr -d '\r')"

# Fall back to provided mode if extraction failed
if [ -z "$MODE_FROM_CONFIG" ]; then
  MODE_FROM_CONFIG="$FALLBACK_MODE"
fi

# Output mode to stdout (Argo will capture this)
printf '%s' "$MODE_FROM_CONFIG"
