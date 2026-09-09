import time
from contextlib import nullcontext

import torch


class WorkshopTrainer:
    def __init__(self, model, optimizer, data_loader, config, wandb_run=None):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.config = config
        self.wandb_run = wandb_run
        self.steps = []
        self.train_losses = []
        self.val_losses = []
        self.tokens_per_second = []

    def _autocast(self):
        if self.config.device_type == 'cuda' and self.config.use_bf16:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return nullcontext()

    def train_step(self):
        input_batch, target_batch = self.data_loader.get_batch('train')
        self.optimizer.zero_grad()
        with self._autocast():
            _, loss = self.model(input_batch, target_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self):
        self.model.eval()
        losses = {'train': [], 'val': []}
        with torch.no_grad():
            for split in ['train', 'val']:
                for _ in range(self.config.evaluation_loops):
                    input_batch, target_batch = self.data_loader.get_batch(split)
                    with self._autocast():
                        _, loss = self.model(input_batch, target_batch)
                    losses[split].append(loss.item())
        self.model.train()
        return {split: sum(values) / len(values) for split, values in losses.items()}

    def train(self):
        last_eval_end_time = time.time()
        last_eval_step = 0
        final_train_loss = None
        final_val_loss = None

        for step in range(1, self.config.total_steps + 1):
            self.train_step()

            if step % self.config.evaluation_frequency == 0 or step == self.config.total_steps:
                current_eval_start_time = time.time()
                steps_since_eval = step - last_eval_step
                evaluation_interval = current_eval_start_time - last_eval_end_time
                tokens = (
                    self.config.batch_size
                    * self.config.input_sequence_length
                    * steps_since_eval
                )
                tokens_per_second = tokens / evaluation_interval

                eval_loss = self.evaluate()
                final_train_loss = eval_loss['train']
                final_val_loss = eval_loss['val']

                print(
                    f"Step {step}: Train Loss {final_train_loss:.4f}, "
                    f"Validation Loss {final_val_loss:.4f}, "
                    f"Tokens/s {tokens_per_second:,.0f}"
                )

                self.steps.append(step)
                self.train_losses.append(final_train_loss)
                self.val_losses.append(final_val_loss)
                self.tokens_per_second.append(tokens_per_second)

                if self.wandb_run is not None:
                    metrics = {
                        'train/step': step,
                        'train/loss': final_train_loss,
                        'val/loss': final_val_loss,
                        'performance/tokens_per_sec': tokens_per_second,
                        'performance/seen_tokens': step
                        * self.config.batch_size
                        * self.config.input_sequence_length,
                    }
                    if torch.cuda.is_available():
                        metrics['performance/gpu_memory_gb'] = torch.cuda.memory_allocated() / 1024**3
                        metrics['performance/gpu_peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1024**3
                    self.wandb_run.log(metrics, step=step)

                last_eval_end_time = time.time()
                last_eval_step = step

        return {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'tokens_per_sec': self.tokens_per_second[-1] if self.tokens_per_second else None,
            'seen_tokens': self.config.total_steps
            * self.config.batch_size
            * self.config.input_sequence_length,
        }
