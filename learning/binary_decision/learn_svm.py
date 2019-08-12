from __future__ import print_function, division

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import pandas as pd
from sklearn import neighbors, svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import expreport as report





def load_img(img_path, img_id):
    return Image.open("%s/%s.png" % (img_path, img_id)).convert('RGB')


class ImDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, ds, data_transform):
        self.ds = ds
        self.ds_size = len(self.ds.index)
        self.img_path = img_path
        self.data_transform = data_transform

    def __len__(self):
        return self.ds_size

    def __getitem__(self, index):
        dat = self.ds.iloc[index]
        image = load_img(self.img_path, dat["id"])
        image = self.data_transform(image)
        return image, dat["id"]

class VecPairDataset(torch.utils.data.Dataset):
    def __init__(self, vectors, pairs_ds):
        self.ds = pairs_ds
        self.ds_size = len(self.ds.index)
        self.vectors = vectors

    def __len__(self):
        return self.ds_size

    def __getitem__(self, index):
        dat = self.ds.iloc[index]
        id1 = str(dat["id1"])
        id2 = str(dat["id2"])
        if id1 not in self.vectors:
            vec1 = next(iter(self.vectors.values()))
            print("vec1 missing!: ", id1)
        else:
            vec1 = self.vectors[id1]

        if id2 not in self.vectors:
            vec2 = next(iter(self.vectors.values()))
            print("vec2 missing!: ", id2)
        else:
            vec2 = self.vectors[id2]
        label = 0 if dat["type"] <= 0 else 1
        return vec1, vec2, label

class CCDBinClassifier(nn.Module):
    def __init__(self, vec_size):
        super(CCDBinClassifier, self).__init__()
        self.encode_dim = vec_size
        self.hidden_dim = vec_size
        self.num_layers = 1
        self.batch_size = 4
        self.label_size = 1
        self.gpu = True
        # gru
        self.bigru = nn.GRU(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.hidden_dim, 1)

    def init_hidden(self):
        if self.gpu is True:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x1, x2):
        x1 = x1.squeeze(-1).squeeze(-1)
        x2 = x2.squeeze(-1).squeeze(-1)

        #print(x1.size())
        #print(x1)


        y = torch.abs(torch.add(x1, -x2))

        #print(y.size())
        #x = self.avgpool(y)
        #x = x.reshape(x.size(0), -1)
        x = self.fc(y)
        x = torch.sigmoid(x.squeeze(-1))
        return x

def compute_vectors(data_path, image_path, transform, retrained_model = None):
    vectors = {}
    print("compute vectors")

    raw_dataset = pd.read_csv(data_path)
    dataset_size = len(raw_dataset)
    dataset = ImDataset(image_path, raw_dataset, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # create pre-trained but not re-trained image cnn to compute internal features
    if retrained_model is None:
        icnn = models.resnet50(pretrained=True)
        icnn = nn.Sequential(*list(icnn.children())[:-1]) # chop of last layer
    else:
        icnn = retrained_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    icnn = icnn.to(device)
    icnn.eval()  # Set model to evaluate mode
    with torch.no_grad():
        for inputs, ids in dataloader:
            inputs = inputs.to(device)

            outputs = icnn(inputs)
            outputs = outputs.to("cpu")
            ids = ids.to("cpu").numpy()
            for i in range(len(outputs)):
                vectors[str(ids[i])] = outputs[i]
    return vectors


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            trues = []
            predicts = []
            # Iterate over data.
            for inputsA, inputsB, labels in dataloaders[phase]:
                inputsA = inputsA.to(device)
                inputsB = inputsB.to(device)
                labels = labels.type(torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputsA, inputsB)
                    #print(outputs)
                    preds = (outputs.data > 0.50).float()#.cpu().numpy()
                    predicts.extend((outputs.data > 0.50).float().to("cpu").numpy())
                    trues.extend(labels.data.to("cpu").numpy())
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputsA.size(0)
                running_corrects += torch.sum(preds == labels.data)
               # print(preds)
               # print(labels.data)
            #for i in len(preds):
            #    if preds[i] != labels.data[i]:
            #        print(labels.data[i])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts)
            print("raw - %s - P: %s; R: %s, F1: %s" % (phase, precision, recall, f1))
            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            print("%s - P: %s; R: %s, F1: %s" % (phase, precision, recall, f1))
            if (phase == "test") and (epoch == num_epochs-1):
                report.report.add_result(epoch_acc.item(), precision, recall, f1, "bc")
                print("CSV-RESULT: bc, %s, %s, %s, %s, %s" % (phase, epoch_acc.item(), precision, recall, f1))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

# Epochs is not used by this algorithm
def learn(data_path, image_path, pairs_path, epochs=5):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
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

    vectors = {
        "train": compute_vectors(data_path, image_path, data_transforms["train"]),
        "test": compute_vectors(data_path, image_path, data_transforms["test"])
    }

    pddat = pd.read_csv(pairs_path)
    # Use train field if present, else do random splitting
    if "train" in pddat.columns:
        ds = {
            "train": pddat[pddat["train"] == 1],
            "test": pddat[pddat["train"] == 0]
        }
    else:
        train, test = train_test_split(pddat, test_size=0.2, stratify=pddat[["func"]])
        ds = {
            "train": train,
            "test": test
        }
    dataset_sizes = {x: len(ds[x]) for x in ['train', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    pdds = {x: VecPairDataset(vectors[x], ds[x])
        for x in ['train', 'test']}

    #print(vectors["train"].keys())

    x_train = []
    y_train = []
    for idx, r in ds["train"].iterrows():
        id1 = str(r["id1"])
        id2 = str(r["id2"])
        if id1 not in vectors["train"]:
            continue
        if id2 not in vectors["train"]:
            continue
        x1 = vectors["train"][id1].numpy().flatten()
        x2 = vectors["train"][id2].numpy().flatten()
        x_train.append(np.absolute(x1-x2))
        y_train.append(0 if r["type"] <= 0 else 1)
    x_test = []
    y_test = []
    for idx, r in ds["test"].iterrows():
        id1 = str(r["id1"])
        id2 = str(r["id2"])
        if id1 not in vectors["test"]:
            continue
        if id2 not in vectors["test"]:
            continue
        x1 = vectors["train"][id1].numpy().flatten()
        x2 = vectors["train"][id2].numpy().flatten()
        x_test.append(np.absolute(x1-x2))
        y_test.append(0 if r["type"] <= 0 else 1)

    #clf = neighbors.KNeighborsClassifier()
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    confidence = clf.score(x_test, y_test)
    y_pred = clf.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    print("SVM :: ")
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    print("raw - P: %s; R: %s, F1: %s" % (precision, recall, f1))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print("P: %s; R: %s, F1: %s" % (precision, recall, f1))
    report.report.add_result(confidence, precision, recall, f1, "svm")
    print("CSV-RESULT: svm, %s, %s, %s, %s" % (confidence, precision, recall, f1))
    print()

    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    confidence = clf.score(x_test, y_test)
    y_pred = clf.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    print("kNN :: ")
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    print("raw - P: %s; R: %s, F1: %s" % (precision, recall, f1))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print("P: %s; R: %s, F1: %s" % (precision, recall, f1))
    report.report.add_result(confidence, precision, recall, f1, "knn")
    print("CSV-RESULT: knn, %s, %s, %s, %s" % (confidence, precision, recall, f1))
    print()

    dataloaders = {x: torch.utils.data.DataLoader(pdds[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(ds[x].index) for x in ['train', 'test']}

    model = CCDBinClassifier(2048)
    model = model.to(device)

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    train_model(model, dataloaders, dataset_sizes, loss_function, optimizer, exp_lr_scheduler, num_epochs=epochs)


    #example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
    #example_measures = example_measures.reshape(len(example_measures), -1)
    #prediction = clf.predict(example_measures)
    #print(prediction)

if __name__ == "__main__":
    dat_path = "<datasets_path>/<dsname>/funcs.csv" #Path must be set
    pairs_path = "<datasets_path>/<dsname>/pairs.csv" #Path must be set
    img_path = "<datasets_path>/<dsname>/images/kp" #Path must be set
    learn(dat_path, img_path, pairs_path)
