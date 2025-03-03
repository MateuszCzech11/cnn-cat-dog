import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall
from cnn import DogCatClassifier
from dataset import CatDogDataset

ds_train=CatDogDataset('train')
dataloader_train=DataLoader(
    ds_train,
    batch_size=10,
    shuffle=True,
)

def train_model(optimizer,model,num_epoch):
    criterion=nn.BCELoss()
    for epoch in range(num_epoch):
        for features,labels in dataloader_train:
            optimizer.zero_grad()
            labels=labels.view(-1,1).float() # view to ensure labels are (batch_size,1)
            outputs=model(features)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}')


model=DogCatClassifier()
optimizer=optim.Adam(model.parameters(),lr=0.01)

train_model(
    optimizer=optimizer,
    model=model,
    num_epoch=6,
)

torch.save(model.state_dict(),'cat_dog_classifier.pt')