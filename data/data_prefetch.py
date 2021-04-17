# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 4/17/21 6:56 PM
# @Version	1.0
# --------------------------------------------------------
import torch


class DataPrefetch:
    def __init__(self, loader):
        self.next_input = None
        self.next_target = None
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        targets = self.next_target
        self.preload()
        return inputs, targets


def main():
    """

        training_data_loader = DataLoader(
            dataset=train_dataset,
            num_workers=opts.threads,
            batch_size=opts.batchSize,
            pin_memory=True,
            shuffle=True,
        )
        for iteration, batch in enumerate(training_data_loader, 1):
        =====================improvement================================
        data_loader = DataPrefetch(training_data_loader)
        data, label = data_loader.next()
        step = 0
        while data is not None:
            step += 1
            data, label = data_loader.next()
    """
    pass
