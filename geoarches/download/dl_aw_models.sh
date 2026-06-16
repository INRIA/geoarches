#!/bin/bash

# Ensure we are in the directory where you want the models
mkdir -p modelstore

src="https://huggingface.co/gcouairon/ArchesWeather/resolve/main"
MODELS=("archesweather-m-seed0" "archesweather-m-seed1" "archesweather-m-skip-seed0" "archesweather-m-skip-seed1" "archesweathergen")

for MOD in "${MODELS[@]}"; do
    echo "Processing $MOD..."
    mkdir -p "modelstore/$MOD/checkpoints"
    
    # Download files
    wget -q -O "modelstore/$MOD/checkpoints/checkpoint.ckpt" "$src/${MOD}_checkpoint.ckpt"
    wget -q -O "modelstore/$MOD/config.yaml" "$src/${MOD}_config.yaml"
    
    # Patch the checkpoint using Python
    python3 -c "
import torch
path = 'modelstore/$MOD/checkpoints/checkpoint.ckpt'
ckpt = torch.load(path, map_location='cpu', weights_only=False)
if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
    ckpt = {'state_dict': ckpt}
if 'pytorch-lightning_version' not in ckpt:
    ckpt['pytorch-lightning_version'] = '2.5.0.post0'
torch.save(ckpt, path)
print('✓ $MOD checkpoint downloaded.')
"
done

echo "All models downloaded and patched."
