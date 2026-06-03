import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential


class BasicBlock(nn.Module):
    """
    Class for basic blocks (modules) for the model.

    Inherits from nn.Module.
    """

    def __init__(
        self, in_planes: int, out_planes: int, stride: int, dropRate: float = 0.0
    ) -> None:
        """
        Constructor for BasicBlock class.

        Args:
            in_planes (int): Input channels at the module.
            out_planes (int): Output channels at the module.
            stride (int): Number of times a kernel is moved.
            dropRate (float): Drop rate in the module. Defaults to 0.0.

        Returns:
            None: Initializes the BasicBlock class.
        """
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns the output of an input going through a layer.

        Args:
            x (Tensor): Input to the layer.

        Returns:
            Tensor: Output of the layer.
        """
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """
    Class for layers of the model.

    Inherits from nn.Module.
    """

    def __init__(
        self,
        n_layers: int,
        in_planes: int,
        out_planes: int,
        block,  # FIGURE OUT
        stride: int,
        dropRate: float = 0.0,
    ) -> None:
        """
        Constructor for the NetworkBlock class.

        Args:
            n_layers (int): Number of layers.
            in_planes (int): Input channels at the layer.
            out_planes (int): Output channels at the layer.
            block (BasicBlock): Layer in the network.
            stride (int): Number of times a kernel is moved.
            dropRate (float): Drop rate at the layer. Defaults to 0.0.

        Returns:
            None: Initializes the NetworkBlock class.
        """
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, n_layers, stride, dropRate
        )

    def _make_layer(
        self,
        block: BasicBlock,
        in_planes: int,
        out_planes: int,
        n_layers: int,
        stride: int,
        dropRate: float,
    ) -> Sequential:
        """
        Creates a layer in the network.

        Args:
            block (BasicBlock): Layer in the network.
            in_planes (int): Input channels at the layer.
            out_planes (int): Output channels at the layer.
            n_layers (int): Number of layers.
            stride (int): Number of times a kernel is moved.
            dropRate (float): Drop rate at the layer.

        Returns:
            Sequential: Layer of the network.
        """
        layers = []
        for i in range(int(n_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Sequential) -> Sequential:
        """
        Returns the output of the input going through a layer.

        Args:
            x (Sequential): Input to the layer.

        Returns:
            Sequential: Output of the layer.
        """
        return self.layer(x)


class WideResNet(nn.Module):
    """
    Class for the Wide Residual Network Model.

    Inherits from nn.Module.
    """

    def __init__(
        self, depth: int, num_classes: int, widen_factor: int = 1, dropRate: float = 0.0
    ) -> None:
        """
        Constructor for WideResNet class.

        Args:
            depth (int): Depth of the network.
            num_classes (int): Number of classes.
            widen_factor (int): Widening factor. Defaults to 1.
            dropRate (float): Drop rate of the network. Defaults to 0.0.

        Returns:
            None: Initializes the WideResNet class.
        """
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # Global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x: Sequential) -> Tensor:
        """
        Returns the output from feeding the input to the network.

        Args:
            x (Sequential): The network.

        Returns:
            Tensor: Output data from the network.
        """
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
