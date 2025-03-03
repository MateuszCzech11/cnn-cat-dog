#  LINK TO DATA        https://huggingface.co/datasets/Bingsu/Cat_and_Dog
import torch
from torchvision import transforms
from datasets import load_dataset

class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self,train_or_test):
        self.dataset=load_dataset("Bingsu/Cat_and_Dog")[train_or_test]

    def __getitem__(self, index):
        sample = self.dataset[index]
        data,label=sample['image'],sample['labels']

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #mean,std taken from ImageNet dataset since img are similiar
            transforms.Resize((300,280))
        ])
        return transform(data),torch.tensor(label)
    def __len__(self):
        return len(self.dataset)