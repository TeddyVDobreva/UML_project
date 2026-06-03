import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from models import WideResNet
import numpy as np
import torchvision.transforms as transforms


EPOCHS = 200
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
NESTEROV = True
DECAY = 5e-4
PRIT_FREQ = 10
LAYERS = 40
WIDE_LAYERS = 2
DROPRATE = 0
NAME = "WideResNet-40-2"
NUM_CLASSES = 22
TRANS =  transform_train = transforms.Compose([transforms.ToTensor()])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = images
        self.labels = labels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """Given an index, return the token ids and label for the corresponding sample."""
        # Turn image from HxWxC to CxHxW
        item = TRANS(self.images[idx])

        label = self.labels[idx] # 0 negative, 1 positive
        return item, label


def train_loop(train_images, train_labels, validation_images, validation_labels, loss = "CE"):
    # Data loading code
    best_prec1 = 0
    kwargs = {"num_workers": 1, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        ImageDataset(train_images, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        ImageDataset(validation_images, validation_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        **kwargs,
    )

    # create model
    model = WideResNet(
        LAYERS,
        NUM_CLASSES,
        WIDE_LAYERS,
        dropRate=DROPRATE,
    )

    # get the number of model parameters
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    if loss == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        # IMPLEMENT LOG NORM LOSS
        criterion = None
    optimizer = torch.optim.SGD(
        model.parameters(),
        LR,
        momentum=MOMENTUM,
        nesterov=NESTEROV,
        weight_decay=DECAY,
    )

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * EPOCHS
    )

    for epoch in range(EPOCHS):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
            },
            is_best,
        )
    print("Best accuracy: ", best_prec1)


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inp, target) in enumerate(train_loader):
        # target = target.cuda(non_blocking=True)
        # input = input.cuda(non_blocking=True)

        # compute output
        output = model(inp)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), inp.size(0))
        top1.update(prec1.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRIT_FREQ == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                )
            )


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inp, target) in enumerate(val_loader):
        # target = target.cuda(non_blocking=True)
        # input = input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(inp)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), inp.size(0))
        top1.update(prec1.item(), inp.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRIT_FREQ == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                )
            )

    print(" * Prec@1 {top1.avg:.3f}".format(top1=top1))
    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (NAME)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "runs/%s/" % (NAME) + "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    train_loop()
