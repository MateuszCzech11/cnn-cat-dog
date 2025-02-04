import torch.nn as nn
#create cnn model
#train it on data 
# validate on data
# save weights and architecture
# create new file where load weights and architecture, and where you put in your own image and it guesses

class DogCatClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(), #128x75x70
            nn.MaxPool2d(kernel_size=2,stride=2,padding=1),#w/o padding dimensions no longer integers - this way its 128x38x36
            nn.Flatten()
        )
        self.fc1=nn.Linear(128*38*36,128)
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(0.5)
        self.fc2=nn.Linear(128,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.conv_block(x)
        x=self.relu(self.fc1(x))
        x=self.drop(x)
        x=self.sigmoid(self.fc2(x))
        return x