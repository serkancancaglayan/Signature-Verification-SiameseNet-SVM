import os
import torch
import Config
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from scipy.io import savemat
from SiameseNet import SiameseNet
from torchvision import transforms
from DataLoader import SignatureData
from torch.utils.data import DataLoader
from ContrastiveLoss import ContrastiveLoss



def train_fn(model, train_data_loader, optimizer, loss_fn, device, epoch_num):
    train_loop = tqdm(train_data_loader)
    epoch_loss = list()
    
    model.train()
    
    for batch_idx, (input1, input2, labels) in enumerate(train_loop):

        input1 = input1.to(device)
        input2 = input2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        x1, x2 = model(input1, input2)
        loss = loss_fn(x1, x2, labels)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        train_loop.set_description_str('Train Epoch ' + str(epoch_num))
    return np.mean(epoch_loss)


@torch.no_grad()
def test_fn(model, test_loader, loss_fn, device, epoch_num):
    test_loop = tqdm(test_loader)
    epoch_loss = list()
    model.eval()

    for batch_idx, (input1, input2, labels) in enumerate(test_loop):

        input1 = input1.to(device)
        input2 = input2.to(device)
        labels = labels.to(device)

        x1, x2 = model(input1, input2)
        loss = loss_fn(x1, x2, labels)

        epoch_loss.append(loss.item())
        test_loop.set_description_str('Test Epoch ' + str(epoch_num))


    return np.mean(epoch_loss)


def main():
    
    model = SiameseNet(Config.ARCHITECTURE_CFG)
    model = model.to(Config.DEVICE)

    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE,Config.IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    train_dataset = SignatureData(Config.TRAIN_FOLDER, Config.TRAIN_ANNOTATION, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    test_dataset = SignatureData(Config.TEST_FOLDER, Config.TEST_ANNOTATION, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)


    contrastive_loss = ContrastiveLoss(Config.ALPHA, Config.BETA, Config.MARGIN)
    optimizer = optim.RMSprop(model.parameters(), lr=Config.LEARNING_RATE, eps=Config.EPS, weight_decay=Config.WEIGHT_DECAY, momentum=Config.MOMENTUM)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, Config.STEP_SIZE, Config.GAMMA)

    train_loss_list = list()
    test_loss_list = list()

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        
        train_loss = train_fn(model, train_loader, optimizer, contrastive_loss, Config.DEVICE, epoch)
        print('Training epoch ' + str(epoch) + ' loss: ', str(train_loss))
        train_loss_list.append(train_loss)

        test_loss = test_fn(model, test_loader, contrastive_loss, Config.DEVICE, epoch)
        print('Testing epoch ' + str(epoch) + ' loss: ', str(test_loss))
        test_loss_list.append(test_loss)

        lr_scheduler.step()

        if epoch % Config.SAVE_CHECKPOINT_INTERVAL == 0:
            model_dict = {
                "model": model.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "optim": optimizer.state_dict()
            }
            checkpoint_name = 'epoch_' + str(epoch) + 'loss_' + str(round(test_loss)) + '.pt'
            torch.save(model_dict, os.path.join(Config.CHECKPOINT_PATH, checkpoint_name))
        

    history_dict = {
        "train_loss": train_loss_list,
        "test_loss": test_loss_list
    }
    savemat("history.mat", history_dict)

if __name__ == "__main__":
    main()