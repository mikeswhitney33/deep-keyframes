import os
import torch
import argparse
import data
import models
import matplotlib.pyplot as plt
import tqdm

import numpy as np

from utils import get_device_with_args

if __name__ == "__main__":
    _, device = get_device_with_args()

    net = models.fcn_resnet50(6, 3).to(device)
    optimizer = torch.optim.Adam(net.parameters())
    criteria = torch.nn.MSELoss()
    transform = data.compose(data.resize, data.to_tensor, data.normalize)

    train_dataset = data.CachedDataset(root="data/cache-train", transform=transform)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=24, shuffle=True)
    best_loss = 1e8
    net.train()
    for epoch in tqdm.trange(100):
        running_loss = []
        for x, y in tqdm.tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outs = net(x)
            loss = criteria(outs['out'], y)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        epoch_loss = np.mean(running_loss)
        if epoch_loss < best_loss:
            tqdm.tqdm.write(f"Epoch {epoch} Loss: {epoch_loss} was better than {best_loss}")
            torch.save(net.state_dict(), os.path.join("checkpoints", f"{epoch:06}.pth"))
            best_loss = epoch_loss
