import torchvision as tv
import torch.nn as nn


def fcn_resnet50(in_channels, out_channels, *args, **kwargs):
    fcn = tv.models.segmentation.fcn_resnet50(*args, num_classes=out_channels, **kwargs)
    fcn.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
    return fcn
