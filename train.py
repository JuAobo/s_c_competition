import torch
import torch.nn as nn
import os
import random
import time
import argparse
import numpy as np
import pandas as pd
import albumentations

from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from util import LabelSmoothCELoss, GradualWarmupScheduler
from dataset import Data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, default='train')
    parser.add_argument('--model', type=str, default='efficientnet-b0')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--bs', type=int, default=16)
    args, _ = parser.parse_known_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(data_set, model):
    data_loader = DataLoader(dataset=data_set, batch_size=16, shuffle=False)
    model.eval()
    m = nn.Softmax(dim=1)
    val_preds = torch.zeros((len(data_set),3), device=device)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            preds = model(inputs)
            preds = m(preds)
            val_preds[i*data_loader.batch_size:i*data_loader.batch_size+inputs.shape[0]]=preds
    val_labels = torch.argmax(val_preds, dim=1)
    return  val_preds, val_labels

if __name__ == '__main__':

    args = parse_args()
    seed_everything(47)
    current_time = time.strftime("%Y_%m_%d_%H.%M", time.localtime())
    classes = ['normal', 'calling', 'smoking', 'smoking_calling']
    train_list = []
    for label, folder in enumerate(classes):
        for img in os.listdir(os.path.join(args.data_folder, folder)):
            train_list.append((os.path.join(args.data_folder, folder, img), label))
    train_df = pd.DataFrame(train_list, columns=['img', 'label'])
    train_transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Resize(224, 224),
        albumentations.Normalize()])
    test_transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = LabelSmoothCELoss(reduction='none')
    model = EfficientNet.from_pretrained(args.model, num_classes=4)
    model = model.to(device)
    train_set = Data(train_df, train_transform)
    train_loader = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=0.0005)
    # optim = SWA(base_optim, swa_start=770, swa_freq=77, swa_lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epoch, eta_min=5e-6)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    for epoch in range(1, args.epoch+6):
        model.train()
        running_loss = 0.0
        scheduler_warmup.step(epoch)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optim.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss, _ = loss.topk(k=loss.shape[0] // 3)
            loss = loss.mean()
            loss.backward()
            optim.step()
            running_loss += loss.item() * len(labels)
        # if epoch>9:
        # optim.swap_swa_sgd()
        train_loss = running_loss / len(train_set)
        print('epoch {}: train_loss {}'.format(epoch, train_loss))
    PATH = args.model + '.pth'
    torch.save(model.state_dict(), PATH)