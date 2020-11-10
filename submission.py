import torch
import torch.nn as nn
import os
import random
import time
import argparse
import numpy as np
import pandas as pd
import albumentations
import json

from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from util import LabelSmoothCELoss, GradualWarmupScheduler
from dataset import Data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, default='testA')
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
    val_preds = torch.zeros((len(data_set),4), device=device)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['normal', 'calling', 'smoking', 'smoking_calling']
    test_transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize()])
    test_transform2 = albumentations.Compose([
        albumentations.HorizontalFlip(p=1),
        albumentations.Resize(224, 224),
        albumentations.Normalize()])
    test_list = []
    for img in os.listdir(args.data_folder):
        test_list.append((os.path.join(args.data_folder, img), 0))
    test_df = pd.DataFrame(test_list, columns=['img', 'label'])
    test_set = Data(test_df, test_transform)
    test_set2 = Data(test_df, test_transform2)
    test_tt = 0
    for net in ['efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5']:
        model = EfficientNet.from_pretrained(net, num_classes=4)
        model = model.to(device)
        model.load_state_dict(torch.load(net + '.pth'), strict=True)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        test_preds, test_labels = evaluate(test_set, model)
        test_preds2, test_labels2 = evaluate(test_set2, model)
        test_preds = (test_preds + test_preds2) / 2
        test_tt += test_preds
    test_preds = test_tt / 3
    test_labels = torch.argmax(test_preds, dim=1)
    test_preds = torch.max(test_preds, dim=1)[0].cpu()
    result = []
    for image_name, category, score in zip(os.listdir(args.data_folder), test_labels, test_preds):
        category = classes[category]
        d = {"image_name": image_name, "category": category, "score": score.item()}
        result.append(d)
    with open("result.json", "w") as f:
        json.dump(result, f)

