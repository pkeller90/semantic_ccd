from __future__ import print_function, division

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import expreport as report

def count_labels(datasets):
    labels = set()
    for _, ds in datasets.items():
        labels.update(ds["func"].unique())
    return len(labels)

def create_label_mapping(datasets):
    labels = set()
    for _, ds in datasets.items():
        labels.update(ds["func"].unique())
    mapping = {}
    indx = 1
    for x in labels:
        mapping[x] = indx
        indx += 1
    return mapping

def load_img(img_path, img_id):
    return Image.open("%s/%s.png" % (img_path, img_id)).convert('RGB')


class PdDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, ds, data_transform, labelmapping):
        self.ds = ds
        self.ds_size = len(self.ds.index)
        self.img_path = img_path
        self.data_transform = data_transform
        self.labelmapping = labelmapping

    def __len__(self):
        return self.ds_size

    def __getitem__(self, index):
        dat = self.ds.iloc[index]
        image = load_img(self.img_path, dat["id"])
        image = self.data_transform(image)
        raw_label = dat["func"]
        label = self.labelmapping[raw_label] if raw_label in self.labelmapping else 0
        return image, label

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        y_test = []
        y_pred = []
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                y_test.extend(labels.data.to("cpu").numpy())
                y_pred.extend(preds.data.to("cpu").numpy())
               # print(preds)
               # print(labels.data)
            #for i in len(preds):
            #    if preds[i] != labels.data[i]:
            #        print(labels.data[i])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))

            print("functionality classification :: ")
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
            print("raw - %s - P: %s; R: %s, F1: %s" % (phase, precision, recall, f1))
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
            print("%s - P: %s; R: %s, F1: %s" % (phase, precision, recall, f1))
            if (phase == "test") and (epoch == num_epochs-1):
                report.report.add_result(epoch_acc.item(), precision, recall, f1, "fc")
                print("CSV-RESULT: fc, %s, %s, %s, %s" % (epoch_acc.item(), precision, recall, f1))
            print()

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#pairs_path is not used but present for simplicity
def learn(data_path, image_path, pairs_path=None, epochs=25):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
    #'train': transforms.Compose([
    #    transforms.RandomResizedCrop(224),
    #    #transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #]),

    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    pddat = pd.read_csv(data_path)
    if "train" in pddat.columns:
        ds = {
            "train": pddat[pddat["train"] == 1],
            "test":  pddat[pddat["train"] == 0]
        }
    else:
        train, test = train_test_split(pddat, test_size=0.2, stratify=pddat[["func"]], random_state=42)
        ds = {
            "train": train,
            "test": test
        }

    label_mapping = create_label_mapping(ds)
    label_count = len(label_mapping.keys()) + 1
    dataset_sizes = {x: len(ds[x]) for x in ['train', 'test']}


    pdds = {x: PdDataset(image_path, ds[x], data_transforms[x], label_mapping)
        for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(pdds[x], batch_size=4, shuffle=True, num_workers=4)
                for x in ['train', 'test']}

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, label_count)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    #print(model_ft)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=epochs)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = "viscode"
    visualization = "kp"
    dat_path = "<path>/datasets/%s/funcs.csv" % (dataset) # Path must be set
    img_path = "<path>/datasets/%s/images/%s" % (dataset, visualization) # Path must be set
    learn(dat_path, img_path)
