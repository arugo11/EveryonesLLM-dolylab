# EveryonesLLM dolylab workshop: Chapter 15 Python project

このディレクトリ構成は、EveryonesLLM Chapter 1〜15をNotebookではなく通常のPython projectとして実行するためのbaselineです。

目的はモデルを書き直すことではありません。Chapter 1〜15で作成したクラスをできるだけそのまま `.py` に移し、Notebookで暗黙になっていたimport関係と実行入口だけを明示しています。

## Chapterとの対応

| Chapter | Notebookで作ったもの | Python project |
|---|---|---|
| 1 | `DataLoader` | `src/everyones_llm/data.py` |
| 2 | `TokenEmbedding` | `src/everyones_llm/embedding.py` |
| 3 | `PositionEmbedding` | `src/everyones_llm/embedding.py` |
| 4 | `EmbeddingModule` | `src/everyones_llm/embedding.py` |
| 5 | LayerNorm | `torch.nn.LayerNorm`をChapter 9/10と同じ位置で利用 |
| 6 | `AttentionHead` | `src/everyones_llm/attention.py` |
| 7 | `MultiHeadAttention` | `src/everyones_llm/attention.py` |
| 8 | `FeedForward` | `src/everyones_llm/feedforward.py` |
| 9 | `TransformerBlock` | `src/everyones_llm/transformer.py` |
| 10 | `VocabularyLogits` | `src/everyones_llm/model.py` |
| 11 | `nanoGPT` | `src/everyones_llm/model.py` |
| 12〜15 | `Trainer`, GPU training, tokens/sec | `src/everyones_llm/trainer.py` |
| 15 | text generation | `scripts/generate.py` |

## Project structure

```text
.
├── input.txt
├── pyproject.toml
├── requirements.txt
├── scripts/
│   ├── train.py
│   └── generate.py
└── src/
    └── everyones_llm/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── embedding.py
        ├── attention.py
        ├── feedforward.py
        ├── transformer.py
        ├── model.py
        └── trainer.py
```

## Setup

リポジトリのrootで仮想環境を作成し、editable installします。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

CUDA対応PyTorchの導入方法は研究室サーバのCUDA環境に合わせて調整してください。

## Train

Chapter 15と同じ文字単位tokenizer、Shakespeare `input.txt`、モデル設定を使います。

```bash
python scripts/train.py
```

学習終了後に次のcheckpointが作成されます。

```text
checkpoints/chapter15.pt
```

Notebookでは学習済みmodelが同じPython process内に存在するためcheckpointは不要でした。Python projectでは学習と生成を別processで実行するため、この保存処理だけを追加しています。

## Generate

Chapter 15で使用している `"Let's he"` をdefault promptにしています。

```bash
python scripts/generate.py
```

promptや生成token数はCLIから変更できます。

```bash
python scripts/generate.py --prompt "ROMEO:" --max-new-tokens 256
```

## このbaselineで意図的に入れていないもの

このbranchではChapter 15までの再構成に限定しています。以下は次のGPU training workshopで差分として導入します。

- Chapter 16のmodel scaling
- Chapter 17のFineWeb-Edu
- Chapter 18の`tiktoken`
- A100向けBF16 / batch size調整
- Weights & Biasesによるexperiment tracking
- Hugging Face Hubへのdataset/model upload

これらを最初から混ぜないことで、Notebook版Chapter 15から研究用training projectへ何を追加したのか追跡しやすくします。
