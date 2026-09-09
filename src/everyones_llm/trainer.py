import time

import torch


class Trainer:
    def __init__(self, model, optimizer, data_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.config = config

        self.steps = []
        self.train_losses = []
        self.val_losses = []

    def train_step(self):
        input_batch, target_batch = self.data_loader.get_batch('train')
        self.optimizer.zero_grad()

        logits, loss = self.model(input_batch, target_batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self):
        self.model.eval()
        losses = {"train": [], "val": []}
        with torch.no_grad():
            for split in ['train', 'val']:
                for _ in range(self.config.evaluation_loops):
                    input_batch, target_batch = self.data_loader.get_batch(split)
                    _, loss = self.model(input_batch, target_batch)
                    losses[split].append(loss.item())
        self.model.train()

        return {split: sum(values) / len(values) for split, values in losses.items()}

    def train(self):
        for step in range(1, self.config.total_steps + 1):
            if step % self.config.evaluation_frequency == 0 or step == self.config.total_steps:
                if step == self.config.evaluation_frequency or step == self.config.total_steps:
                    tokens_per_second = None
                else:
                    current_eval_start_time = time.time()
                    evaluation_interval = current_eval_start_time - last_eval_end_time
                    tokens_per_evaluation_interval = (
                        self.config.batch_size
                        * self.config.input_sequence_length
                        * self.config.evaluation_frequency
                    )
                    tokens_per_second = tokens_per_evaluation_interval / evaluation_interval

                eval_loss = self.evaluate()
                print(
                    f"Step {step}: Train Loss {eval_loss['train']:.4f}, "
                    f"Validation Loss {eval_loss['val']:.4f}"
                )
                print(f"Tokens per second {tokens_per_second}")

                self.steps.append(step)
                self.train_losses.append(eval_loss['train'])
                self.val_losses.append(eval_loss['val'])

                last_eval_end_time = time.time()

            train_loss = self.train_step()
