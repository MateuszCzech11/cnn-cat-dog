#  LINK TO DATA        https://huggingface.co/datasets/Bingsu/Cat_and_Dog
import torch
from torchmetrics import Accuracy, Precision, Recall
from torch.utils.data import DataLoader
from dataset import CatDogDataset
from cnn import DogCatClassifier

ds_test=CatDogDataset('test')
dataloader_test=DataLoader(
    ds_test,
    batch_size=10,
    shuffle=True,
)



model=DogCatClassifier()
state_dict=torch.load('cat_dog_classifier.pt')
model.load_state_dict(state_dict=state_dict)

accuracy_metric = Accuracy(task='binary', num_classes=2)
precision_metric = Precision(task='binary', num_classes=2, average=None)
recall_metric = Recall(task='binary', num_classes=2, average=None)

model.eval()
for i,(features,labels) in enumerate(dataloader_test):
    output= model.forward(features)
    predictions = (output > 0.5).int().squeeze()
    accuracy_metric(predictions, labels)
    precision_metric(predictions, labels)
    recall_metric(predictions, labels)

accuracy = accuracy_metric.compute().item()
precision = precision_metric.compute().tolist()
recall = recall_metric.compute().tolist()
print('Accuracy:', accuracy)
print('Precision (per class):', precision)
print('Recall (per class):', recall)