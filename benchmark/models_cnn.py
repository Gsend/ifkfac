"""
benchmark/models_cnn.py

ResNet-34 adapted for CIFAR-10 (32×32 inputs).  Used by the comparison
sweep to match the 22M-parameter SmallGPT-medium transformer.

Differences from the standard ImageNet ResNet-34:
  - First conv is 3×3 (stride 1) instead of 7×7 (stride 2)
  - No initial maxpool
  - Average-pool over the final 4×4 feature map → fc(512, 10)

Parameter count: ~21.3M, close to SmallGPT-medium's 22M.

Also provides CIFAR-10 dataloaders matching our K-FAC hook contract:
  - batch=128, 1000 steps ≈ 2.5 epochs
  - Standard CIFAR augmentation (random crop + flip) on training set
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet34_CIFAR(nn.Module):
    """ResNet-34 adapted for CIFAR-10 (3×32×32 inputs, 10 classes).

    Layers (BasicBlock × [3,4,6,3] groups, ~21.3M params):
        stem  : conv(3→64, k=3, s=1) + bn + relu
        layer1: 3 × BasicBlock(64→64, s=1)
        layer2: 4 × BasicBlock(64→128, first s=2)
        layer3: 6 × BasicBlock(128→256, first s=2)
        layer4: 3 × BasicBlock(256→512, first s=2)
        head  : adaptive_avg_pool + linear(512→10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ---- CIFAR-10 data loaders ------------------------------------------------

def cifar10_loaders(batch_size: int = 128, num_workers: int = 2,
                     data_dir: str = None, device=None):
    """Return (train_loader_factory, val_loader, num_classes).

    train_loader_factory is a callable that returns a fresh iter each call —
    mirrors the contract used by the transformer multiseed script.
    """
    from torchvision import datasets, transforms

    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent / "data")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)

    def train_loader_factory():
        return torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader_factory, val_loader, 10


def evaluate_acc(model, val_loader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


if __name__ == "__main__":
    # Sanity check param count
    m = ResNet34_CIFAR()
    n = sum(p.numel() for p in m.parameters())
    print(f"ResNet-34 CIFAR: {n/1e6:.1f}M parameters")
    print(f"  Linear/Conv2d layers tracked by K-FAC: "
          f"{sum(1 for mod in m.modules() if isinstance(mod, (nn.Linear, nn.Conv2d)))}")
