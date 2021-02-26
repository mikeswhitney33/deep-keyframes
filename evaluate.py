import argparse
import os

import torch
import tqdm
import numpy as np

import data
import models
from utils import get_device_with_args


if __name__ == "__main__":
    _, device = get_device_with_args()

    net = models.fcn_resnet50(6, 3).to(device)

    criteria = torch.nn.MSELoss()
    transform = data.compose(data.resize, data.to_tensor, data.normalize)
    train_dataset = data.CachedDataset(root="data/cache-test", transform=transform)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=24, shuffle=False)

    min_loss = 1e8
    best_ckpt = None
    checkpoints = os.listdir("checkpoints")
    with torch.no_grad():
        net.eval()
        for ckpt in tqdm.tqdm(checkpoints):
            with open(os.path.join("checkpoints", ckpt), "rb") as file:
                net.load_state_dict(torch.load(file))
            running_loss = []
            for x, y in tqdm.tqdm(loader, leave=False):
                x, y = x.to(device), y.to(device)
                outs = net(x)
                loss = criteria(outs["out"], y)
                running_loss.append(loss.item())
            checkpoint_loss = np.mean(running_loss)
            tqdm.tqdm.write(f"{ckpt} -- Loss: {checkpoint_loss}")
            if checkpoint_loss < min_loss:
                min_loss = checkpoint_loss
                best_ckpt = ckpt
    print(f"Best Checkpoint: {best_ckpt} with loss: {min_loss}")

