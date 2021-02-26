import os

from PIL import Image
import torch
import torchvision.transforms.functional as tvF
import av
import numpy as np


def compose(*transforms):
    def transform(im1, im2):
        for t in transforms:
            im1, im2 = t(im1, im2)
        return im1, im2
    return transform


def resize(im1, im2):
    a, b = im1
    a = a.resize((224, 224))
    b = b.resize((224, 224))
    im2 = im2.resize((224, 224))
    return (a, b), im2


def to_tensor(im1, im2):
    a, b = im1
    a = tvF.to_tensor(a)
    b = tvF.to_tensor(b)
    im2 = tvF.to_tensor(im2)
    return torch.cat([a, b]), im2


def normalize(im1, im2):
    im1 = (im1 - 0.5) / 0.5
    im2 = (im2 - 0.5) / 0.5
    return im1, im2


def gendata(video_path, train=False, transform=None):
    container = av.open(video_path)
    container.streams.video[0].thread_type = "AUTO"

    key = None
    labels = []
    toggle = train
    for packet in container.demux():
        for frame in packet.decode():
            if isinstance(frame, av.audio.frame.AudioFrame):
                continue
            if frame.key_frame:
                if not key:
                    key = frame
                else:
                    x = (key.to_image(), labels[-1].to_image())
                    y = labels[len(labels)//2].to_image()
                    if len(np.unique(np.array(key.to_image()))) > 100 and toggle:
                        if transform:
                            x, y = transform(x, y)
                        yield x, y
                    toggle = not toggle
                    key = frame
                    labels = []
            else:
                if key:
                    labels.append(frame)

class CachedDataset:
    def __init__(self, root, transform=None):
        self.root = root
        filenames = sorted(os.listdir(root))
        self.anames = [f for f in filenames if f[-5] == 'a']
        self.bnames = [f for f in filenames if f[-5] == 'b']
        self.ynames = [f for f in filenames if f[-5] == 'y']
        self.transform = transform

    def __getitem__(self, idx):
        a = Image.open(os.path.join(self.root, self.anames[idx]))
        b = Image.open(os.path.join(self.root, self.bnames[idx]))
        y = Image.open(os.path.join(self.root, self.ynames[idx]))

        features = (a, b)
        labels = y

        if self.transform:
            features, labels = self.transform(features, labels)

        return features, labels
    def __len__(self):
        return len(self.anames)


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("input_movie")
    parser.add_argument("--outdir", default=os.path.join("data", "cache"))
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    dirname = args.outdir
    for i, (x, y) in enumerate(gendata(args.input_movie)):
        a, b = x
        a.save(os.path.join(dirname, f"{i:06}a.png"))
        b.save(os.path.join(dirname, f"{i:06}b.png"))
        y.save(os.path.join(dirname, f"{i:06}y.png"))
        print(f"{i:06} saved")

