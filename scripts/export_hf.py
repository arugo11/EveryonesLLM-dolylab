import argparse
import json
import shutil
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/workshop.pt')
    parser.add_argument('--output-dir', default='artifacts/hf_model')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    shutil.copy2(checkpoint_path, output_dir / 'model.pt')

    with open(output_dir / 'config.json', 'w', encoding='utf-8') as file:
        json.dump(checkpoint['config'], file, indent=2)
    with open(output_dir / 'training_metrics.json', 'w', encoding='utf-8') as file:
        json.dump(checkpoint.get('metrics', {}), file, indent=2)

    config = checkpoint['config']
    metrics = checkpoint.get('metrics', {})
    readme = f'''---
language:
- en
pipeline_tag: text-generation
tags:
- pytorch
- from-scratch
- educational
---

# EveryonesLLM workshop model

This model was trained from scratch in the EveryonesLLM GPU workshop.

## Training data

- Dataset: HuggingFaceFW/fineweb-edu
- Sample: sample/10BT/000_00000.parquet
- Tokenizer: tiktoken GPT-2

## Model

- Embedding dimension: {config['embedding_dim']}
- Hidden dimension: {config['hidden_dim']}
- Attention heads: {config['num_attention_heads']}
- Transformer blocks: {config['layer_count']}
- Context length: {config['input_sequence_length']}

## Training

- Batch size: {config['batch_size']}
- Steps: {config['total_steps']}
- Learning rate: {config['learning_rate']}
- BF16: {config['use_bf16']}
- Seen tokens: {metrics.get('seen_tokens')}
- Final validation loss: {metrics.get('final_val_loss')}

## Loading

This is a custom educational PyTorch model, not a Transformers AutoModel package.
Use the EveryonesLLM workshop source code to instantiate `nanoGPT`, then load `model.pt`.

## Notes

The code originates from the MIT-licensed EveryonesLLM project.
FineWeb-Edu is distributed under ODC-By.
Check the dataset and repository terms before redistributing derived artifacts.
'''
    with open(output_dir / 'README.md', 'w', encoding='utf-8') as file:
        file.write(readme)

    print(f'exported: {output_dir}')


if __name__ == '__main__':
    main()
