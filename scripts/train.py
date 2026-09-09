from pathlib import Path

import torch

from everyones_llm import Config, DataLoader, Trainer, nanoGPT


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "input.txt"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "chapter15.pt"


def main():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        text = f.read()

    config = Config()
    torch.manual_seed(config.random_seed_value)

    data_loader = DataLoader(text, config)
    model = nanoGPT(
        vocab_size=data_loader.vocab_size,
        config=config,
    ).to(config.device_type)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    print(f"Device: {config.device_type}")
    print(f"Vocabulary size: {data_loader.vocab_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, optimizer, data_loader, config)
    trainer.train()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": trainer.steps,
            "train_losses": trainer.train_losses,
            "val_losses": trainer.val_losses,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
