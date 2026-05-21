#!/bin/sh
set -eu

BUFFER_PATH="/app/data/buffers/"
BUFFERS_DIRS="raw_data processed_data processed_data_feature candidate_pairs matching_pairs cg_feature_index graph sequences embedding_calculating embedding_decision cg_feature_construction cg_feature_candidate predicted_matching"

# Create buffer directories if they don't exist and clear them if they do
for dir in $BUFFERS_DIRS; do
    full_path="${BUFFER_PATH}${dir}"
    if [ -d "$full_path" ]; then
        rm -rf "${full_path:?}"/*
    else
        mkdir -p "$full_path"
    fi
done

FALLBACK_MODE="${FALLBACK_MODE:-}"
CONFIG_SRC=""

for p in /app/config/examples/config-embedding.yaml /app/examples/config.yaml; do
  if [ -f "$p" ]; then
    CONFIG_SRC="$p"
    break
  fi
done

if [ -n "$CONFIG_SRC" ]; then
  cp "$CONFIG_SRC" /tmp/config.yaml
else
  printf 'mode: "%s"\n' "$FALLBACK_MODE" > /tmp/config.yaml
fi

MODE_FROM_CONFIG="$(sed -n 's/^mode:[[:space:]]*"\?\([^"#]*\)"\?.*/\1/p' /tmp/config.yaml | head -n1 | tr -d '\r')"

if [ -z "$MODE_FROM_CONFIG" ]; then
  MODE_FROM_CONFIG="$FALLBACK_MODE"
fi

printf '%s' "$MODE_FROM_CONFIG" > /tmp/mode
