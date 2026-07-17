#!/bin/sh
set -eu

BUFFER_PATH="/app/data/buffers"
BUFFER_DIRS="raw_data processed_data_graph processed_data_feature candidate_pairs matching_pairs cg_feature graph sequences embedding_calculating predicted_matching"

# Incremental workers are started together and coordinate through these directories.
# Clear stale data and EOS markers before any worker is allowed to start.
for dir in $BUFFER_DIRS; do
    full_path="${BUFFER_PATH}/${dir}"
    if [ -d "$full_path" ]; then
        rm -rf "${full_path:?}"/*
    else
        mkdir -p "$full_path"
    fi
done
