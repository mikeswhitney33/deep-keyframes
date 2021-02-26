import argparse
import os
import torch
import torchvision.transforms.functional as tvF
from PIL import Image

import models
import data

from utils import get_device_with_args, predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("frame1")
    parser.add_argument("frame2")
    parser.add_argument("state_dict")
    parser.add_argument("--output", type=str, default="out.png")
    args, device = get_device_with_args(parser)

    net = models.fcn_resnet50(6, 3).to(device)
    with open(args.state_dict, "rb") as file:
        net.load_state_dict(torch.load(file, map_location=device))
    frame1 = Image.open(args.frame1)
    frame2 = Image.open(args.frame2)
    out_image = predict(device, net, frame1, frame2)
    frame1.show()
    frame2.show()
    out_image.save(os.path.join(f"{args.frame1[:-5]}-out.png"))
