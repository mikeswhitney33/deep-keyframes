import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as tvF


def get_device(device_number):
    name = "cpu"
    if torch.cuda.is_available() and 0 >= device_number < torch.cuda.device_count():
        name = f"cuda:{device_number}"
    return torch.device(name)


def get_device_with_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=-1, type=int)
    args = parser.parse_args()
    return args, get_device(args.device)


def predict(device, net, frame1, frame2):
    with torch.no_grad():
        net.eval()
        a = tvF.normalize(tvF.to_tensor(frame1), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        b = tvF.normalize(tvF.to_tensor(frame2), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        inputs = torch.cat([a, b]).unsqueeze(0).to(device)
        outputs = net(inputs)["out"]
        out_image = tvF.to_pil_image(outputs.squeeze() * 0.5 + 0.5)
    return out_image
