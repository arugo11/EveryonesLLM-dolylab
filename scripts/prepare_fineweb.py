import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken


def write_tokens(handle, token_ids):
    array = np.asarray(token_ids, dtype=np.uint16)
    array.tofile(handle)
    return len(array)


def smoke_documents():
    samples = [
        'Language models predict the next token from previous tokens.',
        'Transformers use attention to mix information across a sequence.',
        'A training loop updates parameters using gradients from a loss.',
        'Tokenizers convert text into integer token identifiers.',
        'Checkpoints make a trained model reusable in another process.',
    ]
    for _ in range(20):
        for text in samples:
            yield {'text': text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--documents', type=int, default=50000)
    parser.add_argument('--output-dir', default='data/fineweb')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = tiktoken.get_encoding('gpt2')
    if encoder.n_vocab >= np.iinfo(np.uint16).max:
        raise ValueError('The tokenizer vocabulary does not fit into uint16.')

    if args.smoke:
        documents = smoke_documents()
        document_count = 100
    else:
        from datasets import load_dataset
        data_file = (
            'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/'
            'resolve/main/sample/10BT/000_00000.parquet'
        )
        dataset = load_dataset(
            'parquet',
            data_files=data_file,
            split='train',
            streaming=True,
        )
        documents = iter(dataset)
        document_count = args.documents

    train_documents = int(document_count * 0.9)
    train_tokens = 0
    val_tokens = 0
    processed = 0

    with open(output_dir / 'train.bin', 'wb') as train_file, open(output_dir / 'val.bin', 'wb') as val_file:
        for index, document in enumerate(documents):
            if index >= document_count:
                break
            token_ids = encoder.encode(document['text'], disallowed_special=())
            token_ids.append(encoder.eot_token)
            if index < train_documents:
                train_tokens += write_tokens(train_file, token_ids)
            else:
                val_tokens += write_tokens(val_file, token_ids)
            processed += 1
            if processed % 5000 == 0:
                print(f'processed documents: {processed:,}')

    if processed != document_count:
        raise RuntimeError(f'Expected {document_count} documents, but received {processed}.')

    metadata = {
        'dataset': 'HuggingFaceFW/fineweb-edu sample/10BT/000_00000.parquet',
        'documents': processed,
        'train_documents': train_documents,
        'val_documents': processed - train_documents,
        'tokenizer': 'tiktoken:gpt2',
        'vocab_size': encoder.n_vocab,
        'dtype': 'uint16',
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
