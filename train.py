import os
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from models import WideResNet

TRANS = transform_train = transforms.Compose([transforms.ToTensor()])


class LogitNormLoss(nn.Module):
    """
    Class for Logit Normalization Loss.

    Inherits from nn.Module.
    Implemented with code from https://github.com/hongxin001/logitnorm_ood/blob/main/common/loss_function.py
    """

    def __init__(self, t: float = 1.0) -> None:
        """
        Constructor for the LogitNormLoss class.

        Args:
            t (float): a temperature parameter that modulates the magnitude of the logits.

        Returns:
            None: Initializes the LogitNormLoss class.
        """
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the Logit Normalization Loss between an output and a target.

        Arg:
            x (torch.Tensor): output tensor.
            target (torch.Tensor): target output tensor.

        Returns:
            torch.Tensor: the Logit Normalization Loss between the output and the target.
        """
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)


class ImageDataset(torch.utils.data.Dataset):
    """
    Class for Logit Normalization Loss.

    Inherits from torch.utils.data.Dataset.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        """
        Constructor for an ImageDataset class.

        Args:
            images (np.ndarray): an array with images.
            labels (np.ndarray): an array with labels corresponding to the images.

        Returns:
            None: Initializes the ImageDataset class.
        """
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        """
        Gives the length of the dataset.

        Returns:
            int: the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Gives the token ids and label for a sample at a given index.

        Args:
            idx (int): the index of the wanted sample.

        Returns:
            tuple[torch.Tensor, int]: a tuple with the image and label at index idx in the dataset.
        """
        item = TRANS(self.images[idx])
        label = self.labels[idx]
        return item, label


def train_loop(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    num_classes: int = 22,
    model_name: str = "model",
    num_layers: int = 40,
    num_wide_layers: int = 2,
    droprate: float = 0,
    lr: float = 0.1,
    decay: float = 5e-4,
    optimizer_momentum: float = 0.9,
    nesterov: bool = True,
    loss: str = "cross-entropy",
    lognorm_temperature: float = 1,
    batch_size: int = 128,
    epochs: int = 200,
    print_freq: int = 10,
) -> WideResNet:
    """
    A function for initialization, training, and evaluation on validation dataset.

    Args:
        train_images(np.ndarray): the images in the training dataset.
        train_labels (np.ndarray): the corresponding labels of the images in the training dataset.
        validation_images(np.ndarray): the images in the validation dataset.
        validation_labels (np.ndarray): the corresponding labels of the images in the validation dataset.
        loss (str): the loss to be used in the training of the model.

    Returns:
        WideResNet: The trained model.
    """
    # Load the data.
    best_prec1 = 0
    kwargs = {"num_workers": 1, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        ImageDataset(train_images, train_labels),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        ImageDataset(validation_images, validation_labels),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # Create the model.
    model = WideResNet(
        num_layers,
        num_classes,
        num_wide_layers,
        dropRate=droprate,
    )

    # Get the number of model parameters.
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )

    cudnn.benchmark = True

    # Define loss function (criterion) and optimizer.
    if loss == "cross-entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss == "logit-normalization":
        criterion = LogitNormLoss(lognorm_temperature)
    else:
        raise ValueError(
            f'Loss should be one of "cross-entropy" or "logit-normalization" but {loss} was given'
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=optimizer_momentum,
        nesterov=nesterov,
        weight_decay=decay,
    )

    # Create the learning rate scheduler.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader) * epochs
    )

    for epoch in range(epochs):
        # Train the model for one epoch.
        train(train_loader, model, criterion, optimizer, scheduler, epoch, print_freq)

        # Evaluate the model on the validation set.
        prec1 = validate(val_loader, model, criterion, epoch, print_freq)

        # Remember best prec@1 and save checkpoint.
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
            },
            is_best,
            loss=loss,
            model_name=model_name,
        )
    print("Best accuracy: ", best_prec1)

    return model, best_prec1


def train(
    train_loader: torch.utils.data.DataLoader,
    model: WideResNet,
    criterion: nn.Module,
    optimizer: torch.optimizer.SGD,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    epoch: int,
    print_freq: int,
) -> None:
    """
    Train for one epoch on the training dataset.

    Args:
        train_loader (torch.utils.data.DataLoader): the data loader with the images and true labels.
        model (WideResNet): the model to be trained.
        criterion (nn.Module): the loss to be use for the training.
        optimizer (torch.optimizer.SGD): the optimizer to be use for the training.
        scheduler (torch.optim.lr_scheduler.CosineAnnealingLR): the learnig rate scheduler to be use for the training.
        epoch (int): the number of the current epoch.
        print_freq (int): the number of epochs between printing results.

    Returns:
        None: Trains the model on epoch on the training set.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (inp, target) in enumerate(train_loader):
        # Compute output
        output = model(inp)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), inp.size(0))
        top1.update(prec1.item(), inp.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
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


def validate(
    val_loader: torch.utils.data.DataLoader,
    model: WideResNet,
    criterion: nn.Module,
    epoch: int,
    print_freq: int,
) -> None:
    """
    Perform validation on the validation dataset.

    Args:
        val_loader (torch.utils.data.DataLoader): the data loader with the images and true labels.
        model (WideResNet): the model to be evaluated.
        criterion (nn.Module): the loss to be use for the evaluation.
        epoch (int): the number of the current epoch.
        print_freq (int): the number of epochs between printing results.

    Returns:
        None: Evaluates the model on the validation set.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inp, target) in enumerate(val_loader):
        # Compute output
        with torch.no_grad():
            output = model(inp)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), inp.size(0))
        top1.update(prec1.item(), inp.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
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


def save_checkpoint(
    state: dict,
    is_best: bool,
    filename: str = "checkpoint.pth.tar",
    loss: str = "cross-entropy",
    model_name: str = "model",
) -> None:
    """
    Saves checkpoint to disk.

    Args:
        state (dict): the checkpoint data to save.
        is_best (bool): True if the data is about the epoch with the best perfromance.
        filename (str): The file to which to save the checkpoint.
        loss (str): The name of the loss function.
        model_name (str): the name with which to save the model
    """
    directory = "runs/%s/" % (model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename, "runs/%s/" % (model_name) + "_" + (loss) + "model_best.pth.tar"
        )


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self) -> None:
        """
        Constructor for the AverageMeter class.

        Args:
            None

        Returns:
            None: Initializes AverageMeter class.
        """
        self.reset()

    def reset(self) -> None:
        """
        Resets the average values to 0.

        Args:
            None

        Returns:
            None: Resets values to 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Updates the average and current values

        Args:
            val (float): current value
            n (int): the number of steps with the current value

        Returns:
            None: Updates values.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)
) -> list[float]:
    """
    Computes the precision@k for the specified values of k.

    Args:
        output (torch.Tensor): the outputed predictions of the model.
        target (torch.Tensor): the target labels.
        topk (tuple): tuple of k values

    Returns:
        list[float]: the accuracy for all topk values
    """
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


def get_scores(
    model: WideResNet, images: np.ndarray, batch_size: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gets the predicted scores and predictions for a set of images.

    Args:
        model (WideResNet): the model for which to get the scores.
        images (np.ndarray): the images for which to get the scores.
        batch_size (int): the batch size to use when getting the scores.

    Returns:
        tuple[np.ndarray, np.ndarray]: the predicted scores and predictions for the images.
    """
    model.eval()
    dataset = ImageDataset(images, np.zeros(len(images)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores, all_preds = [], []
    with torch.no_grad():
        for inp, _ in loader:
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)
            all_scores.append(probs.max(dim=1).values.numpy())
            all_preds.append(logits.argmax(dim=1).numpy())

    return np.concatenate(all_scores), np.concatenate(all_preds)
