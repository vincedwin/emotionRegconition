from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from  model import Deep_Emotion
from database import CanvasDataset, eval_data_dataloader
import time
import config
from utils import save_checkpoint, load_checkpoint, epoch_time
from generate_data import Generate_data
a = 3405
x = 1 + a

def train(model, loader, optmizer, loss_fn, device):
    print("===================================Start Training===================================")
    epoch_loss = 0.0
    train_correct = 0
    model.train()

    loop = tqdm(loader, leave = True)

    for idx, (data, labels) in enumerate(loop):
        data, labels = data.to(device), labels.to(device)
        optmizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optmizer.step()
        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss, train_correct

        # _, preds = torch.max(outputs, 1)
        # train_correct += torch.sum(preds == labels.data)
    print("===================================Training Finished===================================")


def evaluate(model, loader, criterion, device):
    epoch_loss = 0.0
    val_correct = 0
    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = model(data)
            val_loss = criterion(val_outputs, labels)
            epoch_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)
        epoch_loss = epoch_loss / len(loader)
    return epoch_loss, val_correct


def main():
        model = Deep_Emotion()
        model.to(config.DEVICE)

        train_dataset       = CanvasDataset(csv_file='dataset/train.csv',
                                            img_dir = config.TRAIN_DIR,
                                            datatype = 'train')
        train_loader        = DataLoader(   train_dataset, batch_size=config.BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=config.NUM_WORKERS)

        validation_dataset  = CanvasDataset(csv_file='dataset/val.csv',
                                            img_dir = config.VAL_DIR,
                                            datatype = 'val')
        val_loader          = DataLoader(   validation_dataset,
                                            batch_size=1,
                                            shuffle = True,
                                            num_workers=config.NUM_WORKERS)

        criterion = nn.CrossEntropyLoss()
        optmizer            = optim.Adam(model.parameters(),lr= config.LEARNING_RATE, betas=(0.5, 0.999))

        best_valid_loss = float("inf")

        if config.LOAD_MODEL:
            load_checkpoint(
                config.CHECKPOINT, model, optmizer, config.LEARNING_RATE,
            )

        print("===================================Start Training===================================")
        for epoch in range(config.NUM_EPOCHS):
            train_loss = 0
            validation_loss = 0
            train_correct = 0
            val_correct = 0

            start_time = time.time()

            """train"""
            train_loss, train_correct = train(model, train_loader, optmizer, criterion, device=config.DEVICE)
            valid_loss, val_correct = evaluate(model, val_loader, criterion, device=config.DEVICE)

            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {config.CHECKPOINT}"
                print(data_str)
                best_valid_loss = valid_loss
                save_checkpoint(model, optmizer, filename=config.CHECKPOINT)

            train_acc = train_correct.double() / len(train_dataset)
            val_acc = val_correct.double() / len(validation_dataset)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            data_str = f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
            data_str += f'\tTrain Loss: {train_loss:.3f}\n'
            data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
            data_str += f'\t Train Acc: {train_acc * 100 : .3f}\n'
            data_str += f'\t Val Acc: {val_acc * 100: .3f}\n'
            print(data_str)

            print("===================================Training Finished===================================")


if __name__ == "__main__":
    main()