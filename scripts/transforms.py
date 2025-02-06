import torchvision.transforms.v2 as v2
import torch.nn as nn
import torch

class CustomCrop(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        crop_off_top = 110
        height = 240
        width = 320
        return v2.functional.crop(x, crop_off_top, 0, height - crop_off_top, width)

class MyTransforms(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = v2.Compose([
            v2.ToImage(),
            CustomCrop(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((40, 60)),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mirror = v2.RandomHorizontalFlip(p=1.0)

    def forward(self, x, angle=None):
        x = self.transform(x)
        if torch.rand(1).item() < 0.5 and angle is not None:
            # 50% change of horizontal flip
            # only when angle is passed so not during deployment
            x = self.mirror(x)
            angle = -angle
        return (x, angle) if angle is not None else x
