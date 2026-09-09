import argparse
import json
from pathlib import Path

import tiktoken
import torch

from everyones_llm.model import nanoGPT
from everyones_llm.token_data import TokenDataLoader
from everyones_llm.workshop_config import WorkshopConfig
from everyones_llm.workshop_trainer import WorkshopTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/fineweb')
    parser.add_argument('--checkpoint', default='checkpoints/workshop.pt')
    parser.add_argument('--project', default='everyones-llm-workshop')
    parser.add_argument('--run-name', default='a100-training')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-frequency', type=int, default=100)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    config = WorkshopConfig()
    config.total_steps = args.steps
    config.batch_size = args.batch_size
    config.evaluation_frequency = args.eval_frequency
    config.use_bf16 = args.bf16

    if args.smoke:
        config.batch_size = 2
        config.input_sequence_length = 32
        config.total_steps = min(args.steps, 2)
        config.evaluation_frequency = 1
        config.evaluation_loops = 1
        config.embedding_dim = 64
        config.hidden_dim = 256
        config.num_attention_heads = 4
        config.layer_count = 2
        config.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.use_bf16 = args.bf16 and config.device_type == 'cuda'

    if config.use_bf16 and config.device_type != 'cuda':
        raise RuntimeError('--bf16 requires CUDA.')

    torch.manual_seed(config.random_seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed_value)
        torch.cuda.reset_peak_memory_stats()

    encoder = tiktoken.get_encoding('gpt2')
    vocab_size = encoder.n_vocab
    data_loader = TokenDataLoader(args.data_dir, config)
    model = nanoGPT(vocab_size=vocab_size, config=config).to(config.device_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f'device: {config.device_type}')
    print(f'parameters: {parameter_count:,}')
    print(f'batch size: {config.batch_size}')
    print(f'bf16: {config.use_bf16}')

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                **config.as_dict(),
                'tokenizer': 'tiktoken:gpt2',
                'dataset': 'HuggingFaceFW/fineweb-edu',
                'parameters': parameter_count,
            },
        )

    trainer = WorkshopTrainer(model, optimizer, data_loader, config, wandb_run=run)
    metrics = trainer.train()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.as_dict(),
        'vocab_size': vocab_size,
        'tokenizer': 'gpt2',
        'metrics': metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    with open(checkpoint_path.with_suffix('.json'), 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)

    if run is not None:
        run.summary.update(metrics)
        run.finish()

    print(f'checkpoint: {checkpoint_path}')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
