import torch.optim.lr_scheduler as lr_scheduler


class WarmUpScheduler(lr_scheduler._LRScheduler):
    """
    To get warmup + cosine anneling to work do we need to define a custom class for the learning rate that inhets from lr_scheduler._LRScheduler

    Fairly certain it works as intended
    """

    def __init__(self, optimizer, warmup_steps, initial_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        super(WarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr * (self.last_epoch + 1) / self.warmup_steps
                for _ in self.base_lrs
            ]

        else:
            return [base_lr for base_lr in self.base_lrs]
