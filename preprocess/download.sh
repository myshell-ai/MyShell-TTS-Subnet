#!/bin/bash

# Base URL for the files
base_url="https://huggingface.co/datasets/ShoukanLabs/AniSpeech/resolve/main/data/ENGLISH-000"

# Directory to save the downloaded files
output_dir="./data"
mkdir -p "$output_dir"

# Download files from 00000 to 00037
for i in $(seq -w 0 37); do
    file_url="${base_url}${i}-of-00038.parquet?download=true"
    output_file="${output_dir}/ENGLISH-000${i}-of-00038.parquet"
    
    echo "Downloading ${output_file}..."
    wget -O "$output_file" "$file_url"
    
    if [ $? -ne 0 ]; then
        echo "Failed to download ${output_file}"
    else
        echo "Successfully downloaded ${output_file}"
    fi
done

echo "Download completed."
