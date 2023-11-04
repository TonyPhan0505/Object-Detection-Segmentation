import os
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from model import UNet
from dataset import CustomDataset
import torch.nn.functional as F
from utils import dice_loss, evaluate
from pathlib import Path

train_images_file = Path("./data/train_X.npy")
train_masks_file = Path("./data/train_seg.npy")
valid_images_file = Path("./data/valid_X.npy")
valid_masks_file = Path("./data/valid_seg.npy")
checkpoint_path = Path("./checkpoints/")

train_images = np.load(train_images_file)
train_masks = np.load(train_masks_file)
valid_images = np.load(valid_images_file)
valid_masks = np.load(valid_masks_file)

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
model = UNet().to(device)
amp = True
epochs = 3
batch_size = 16
lr = 1e-4
save_checkpoint = True
img_scale = 0.5
momentum = 0.999
gradient_clipping = 1.0
n_train = 55000
n_val = 5000

def train_model():
    # datasets
    train_dataset = CustomDataset(n_train, train_images, train_masks)
    val_dataset = CustomDataset(n_val, valid_images, valid_masks)
    
    # data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # logging
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=lr, save_checkpoint=save_checkpoint, img_scale=img_scale, amp = amp)
    )
    print(f'''\nStarting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # optimizer, loss criterion, lr scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.get_in_channels() > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total = n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                assert images.shape[1] == model.get_in_channels()
                images = images.to(device)
                true_masks = true_masks.to(device)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    predicted_masks = model(images)
                    loss = criterion(predicted_masks, true_masks)
                    loss += dice_loss(
                        F.softmax(predicted_masks, dim=1).float(),
                        F.one_hot(true_masks, model.get_out_channels()).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_score, accuracy = evaluate(model, val_loader, device, amp)
                        print(f'\nValidation Dice score: {val_score}. Accuracy: {accuracy}')
        if save_checkpoint:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, str(checkpoint_path / 'checkpoints.pth'))

if __name__ == "__main__":
    train_model()