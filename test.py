import matplotlib.pyplot as plt
import torch
import math

import torch.optim.lr_scheduler as lr_scheduler






class WarmUpScheduler(lr_scheduler._LRScheduler):
    """
    Custom learning rate scheduler that implements a warm-up phase.
    """
    def __init__(self, optimizer, warmup_steps, initial_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = initial_lr  # Manually set initial_lr
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr * (self.last_epoch + 1) / self.warmup_steps
                for _ in self.base_lrs
            ]
        else:
            return [base_lr for base_lr in self.base_lrs]


import math

class WarmUpCosineAnnealingScheduler:
    """
    Combined scheduler that first applies warm-up, then cosine annealing.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, initial_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.current_step = 0

        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = initial_lr  # Manually set initial_lr
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]  # Save the initial learning rates
        self.last_lr = self.base_lrs

        self.warmup_scheduler = WarmUpScheduler(optimizer, warmup_steps, initial_lr)
        self.cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr)

    def step(self, epoch=None):
        if self.current_step < self.warmup_steps:
            self.warmup_scheduler.last_epoch = self.current_step
            self.warmup_scheduler.step()
        else:
            # Adjust the epoch count for the cosine annealing scheduler
            cosine_epoch = self.current_step - self.warmup_steps
            self.cosine_scheduler.last_epoch = cosine_epoch
            self.cosine_scheduler.step()
        self.current_step += 1

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.warmup_scheduler.get_lr()
        else:
            return self.cosine_scheduler.get_last_lr()

    def get_last_lr(self):
        return self.get_lr()
    

    
# Define the optimizer and schedulers as before
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
total_epochs = 100
warmup_epochs = 10
batch_size = 32
dataset_size = 50000  # Adjust to the size of your dataset

total_steps = total_epochs * (dataset_size // batch_size)
warmup_steps = warmup_epochs * (dataset_size // batch_size)
print(warmup_steps)
initial_lr = 0.1
min_lr = 1e-5

# Initialize the combined scheduler
scheduler = WarmUpCosineAnnealingScheduler(optimizer, warmup_steps, total_steps, initial_lr, min_lr)

# Track the learning rate for each step
learning_rates = []

for epoch in range(total_epochs):
    for batch in range(dataset_size // batch_size):
        scheduler.step()
        learning_rates.append(scheduler.get_last_lr()[0])

# Plot the learning rate schedule
plt.figure(figsize=(10, 5))
plt.plot(learning_rates)
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('Warm-Up + Cosine Annealing Learning Rate Schedule')
plt.axvline(x=warmup_steps, color='r', linestyle='--', label='End of Warm-Up')
plt.legend()
plt.grid(True)
plt.show()
