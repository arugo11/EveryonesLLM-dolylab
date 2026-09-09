import argparse

import tiktoken
import torch

from everyones_llm.model import nanoGPT
from everyones_llm.workshop_config import WorkshopConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/workshop.pt')
    parser.add_argument('--prompt', default='The meaning of life is')
    parser.add_argument('--max-new-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError('--temperature must be greater than zero.')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = WorkshopConfig.from_dict(checkpoint['config'])
    config.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = tiktoken.get_encoding(checkpoint.get('tokenizer', 'gpt2'))
    model = nanoGPT(checkpoint['vocab_size'], config).to(config.device_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    token_ids = encoder.encode(args.prompt, disallowed_special=())
    tokens = torch.tensor(token_ids, dtype=torch.long, device=config.device_type).unsqueeze(0)

    torch.manual_seed(config.random_seed_value)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            model_input = tokens[:, -config.input_sequence_length:]
            logits, _ = model(model_input, None)
            logits = logits[:, -1, :] / args.temperature
            if args.top_k > 0:
                k = min(args.top_k, logits.shape[-1])
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float('-inf')
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)

    print(encoder.decode(tokens[0].tolist()))


if __name__ == '__main__':
    main()
