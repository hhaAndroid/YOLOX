from yolox.utils import LRScheduler
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(16, 3, 3, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.conv1(x)
        return x


class Mydataset(Dataset):
    def __init__(self):
        self.img = np.random.random((3, 100, 100))

    def __getitem__(self, item):
        return self.img

    def __len__(self):
        return 118000

    @staticmethod
    def collate_fn(batch):
        return np.array(batch)


def yolox_optimizer_scheduler(model, lr, iters_per_epoch):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = optim.SGD(pg0, lr=0, momentum=0.9, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    scheduler = LRScheduler(
        'yoloxwarmcos',
        lr,
        iters_per_epoch,
        max_epoch,
        warmup_epochs=5,
        warmup_lr_start=0,
        no_aug_epochs=15,
        min_lr_ratio=0.05,
    )
    return optimizer, scheduler


def mmdet_optimizer_scheduler(model, lr, iters_per_epoch):
    optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True,
                     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
    lr_config = dict(
        warmup='exp',
        by_epoch=False,
        warmup_by_epoch=True,
        warmup_iters=5,  # 5 epoch
        no_aug_epochs=15,
        min_lr_ratio=0.05)

    from mmcv.runner import build_optimizer
    from cosinesnnealingwithnoaugiter_lrupdater_hook import CosineAnnealingWithNoAugIterLrUpdaterHook

    optimizer = build_optimizer(model, optimizer)
    scheduler = CosineAnnealingWithNoAugIterLrUpdaterHook(**lr_config)

    return optimizer, scheduler


class Runner():
    def __init__(self, optimizer, iter, epoch, data_loader=None, max_iters=0):
        self.optimizer = optimizer
        self.iter = iter
        self.epoch = epoch
        self.data_loader = data_loader
        self.max_iters = max_iters


if __name__ == '__main__':

    batch_size = 64
    max_epoch = 300

    dataset = Mydataset()
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=Mydataset.collate_fn)

    model = Model()

    basic_lr_per_img = 0.01 / 64
    lr = batch_size * basic_lr_per_img
    iters_per_epoch = len(dataloader)
    max_iter = iters_per_epoch * max_epoch

    yolox_optimizer, yolox_scheduler = yolox_optimizer_scheduler(model, lr, iters_per_epoch)
    mmdet_optimizer, mmdet_scheduler = mmdet_optimizer_scheduler(model, lr, iters_per_epoch)

    mmdet_scheduler.before_run(Runner(mmdet_optimizer, 0, 1, dataloader, max_iter))

    for epoch in range(0, max_epoch):
        mmdet_scheduler.before_train_epoch(Runner(mmdet_optimizer, 0, 1, dataloader, max_iter))
        for iter in range(0, iters_per_epoch):
            progress_in_iter = epoch * iters_per_epoch + iter

            mmdet_scheduler.before_train_iter(Runner(mmdet_optimizer, progress_in_iter, epoch, dataloader, max_iter))

            yolox_optimizer.zero_grad()
            yolox_optimizer.step()

            lr = yolox_scheduler.update_lr(progress_in_iter + 1)
            info_yolox = []
            for param_group in yolox_optimizer.param_groups:
                param_group["lr"] = lr
                info_yolox.append(param_group["lr"])
            print('========================================================')
            print(f'yolox: {iter}/{epoch},lr={info_yolox}')

            info_mmdet = []
            for param_group in mmdet_optimizer.param_groups:
                param_group["lr"] = lr
                info_mmdet.append(param_group["lr"])
            print(f'mmdet: {iter}/{epoch},lr={info_mmdet}')

            for i in range(len(info_yolox)):
                yolox = round(info_yolox[i], 5)
                mmdet = round(info_mmdet[i], 5)
                assert yolox == mmdet
