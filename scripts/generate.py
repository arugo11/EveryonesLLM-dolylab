import argparse
from pathlib import Path

import torch

from everyones_llm import Config, DataLoader, nanoGPT


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "input.txt"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "chapter15.pt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Let's he")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        text = f.read()

    config = Config()
    data_loader = DataLoader(text, config)
    model = nanoGPT(
        vocab_size=data_loader.vocab_size,
        config=config,
    ).to(config.device_type)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=config.device_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoded_prompt = data_loader.encode(args.prompt)
    encoded_tensor = torch.tensor(
        encoded_prompt,
        dtype=torch.long,
        device=config.device_type,
    ).unsqueeze(0)

    with torch.no_grad():
        generated_tokens = model.generate(
            encoded_tensor,
            max_new_tokens=args.max_new_tokens,
        )

    print(data_loader.decode(generated_tokens[0].tolist()))


if __name__ == "__main__":
    main()
