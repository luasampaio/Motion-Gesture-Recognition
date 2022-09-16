import numpy as np
import pandas as pd


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data

import matplotlib.pyplot as plt

data_frame = pd.read_excel("Rec_Train.xlsx")

names = ['ID', 'AGX', 'AGY', 'AGZ', 'VX', 'VY', 'VZ', 'MX', 'MY', 'MZ']
data_frame.columns=names

data_frame.head()




class GestureData(Dataset):
    
    def __init__(self, Train=None):
        super().__init__()
        
        
        self.Train = Train
        
        
        if Train:
            data_frame = pd.read_excel("Rec_Train.xlsx")
        else:
            data_frame = pd.read_excel("Rec_Test.xlsx")
        
        names = ['ID', 'AGX', 'AGY', 'AGZ', 'VX', 'VY', 'VZ', 'MX', 'MY', 'MZ']
        data_frame.columns=names
        
        labels = data_frame['ID']
        
        
        x_frame = data_frame.drop(columns=['ID'])
        
        data_set = np.array(x_frame)
        data_set = data_set.astype(float)
        label_set = np.array(labels)
        
        self.data_tensor = torch.tensor(data_set)
        self.label_tensor = torch.tensor(label_set)
        
       
    def __len__(self) -> int:
        
        return len(self.data_tensor)
    
    def __getitem__(self, i) -> torch.Tensor:
        x = self.data_tensor[i]
        y = self.label_tensor[i]
        
        return (x, y)
     
    
train_set = GestureData(Train=True)
test_set = GestureData(Train=False)

train_sampler = data.BatchSampler(data.SequentialSampler(train_set), 32, True)
test_sampler = data.BatchSampler(data.SequentialSampler(test_set), 32, True)

train_loader = DataLoader(train_set, batch_size=16, sampler=train_sampler)
test_loader = DataLoader(test_set, batch_size=16, sampler=test_sampler)


# Building the depth-wise convolution module:

class depthwise_conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        
        super().__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU())
        
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Build The GestureNet Model:

class GestureNet(nn.Module):
    
    def __init__(self, depthwise_conv, image_channels):
        super().__init__()
        
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            depthwise_conv(in_channels=32, out_channels=64, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        
        self.layer3 = nn.Sequential(
            depthwise_conv(in_channels=64, out_channels=128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            depthwise_conv(in_channels=128, out_channels=256, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.avgpool = nn.AvgPool1d(8)
        
        self.fc = nn.Linear(256, 10)
        
    def forward(self, x):
        # Reshape The input
        x = x.reshape(16, 9, 32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.fc(x)
       
        x = self.dropout(x)
        
        return x
      
# Device configuration, model instanciation, setting optimizer and hyperparameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureNet(depthwise_conv, image_channels=9).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=180)

# Run test on noisy data to ensure that your model is working correctly.

def test():
    x = torch.randn(16, 9, 32)
    x = x.to(device)
    y0 = model(x)
    print(y0)
    
test() 

'''
Training Loop...
'''
train_history = {'loss':[], 'accuracy':[]}
test_history = {'validation_loss':[], 'validation_accuracy':[]}

for epoch in range(180):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x_train,y_train) in enumerate(train_loader):
        
        
        labels = y_train[:,0]
        p_train = torch.randperm(16)
        train_samples = x_train[p_train]
        train_labels = labels[p_train]
        
        train_samples , train_labels = train_samples.to(device), train_labels.to(device)
        train_samples = train_samples.float()
        
        optimizer.zero_grad()
        
        
        outputs = model(train_samples)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predictions = outputs.max(1)
        total += train_labels.size(0)
        correct += predictions.eq(train_labels).sum().item()
        
    train_history['loss'].append(train_loss/total)
    train_history['accuracy'].append(100.*correct/total)
        
        
     
    print(
        "Epoch: ", epoch,
        "Correct: ", correct,
        "Loss: ", train_loss,
        "Accuracy: " ,100.*correct /total,
        "Total is: ", total
    ) 
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (x_test,y_test) in enumerate(test_loader):
            
            test_labels = y_test[:,0]
            
            
            test_samples, test_labels = x_test.to(device), test_labels.to(device)
            test_samples = test_samples.float()
            outputs = model(test_samples)
            
            loss = criterion(outputs, test_labels)
            test_loss += loss.item()
            _, predictions = outputs.max(1)
            total += test_labels.size(0)
            correct += predictions.eq(test_labels).sum().item()
            
        test_history['validation_loss'].append(test_loss/total)
        test_history['validation_accuracy'].append(100.*correct/total)
        test_acc = 100.*correct/total
        
    scheduler.step()
    
'''Plotting the Results...'''    

# Training and validation loss:

epochs = len(train_history['loss'])
x_axis = range(0,epochs)
fig, ax = plt.subplots(dpi=100)
ax.plot(x_axis, train_history['loss'], '-', label='Training loss')
ax.plot(x_axis, test_history['validation_loss'], '-', label='Validation loss', color='r')
ax.legend()
plt.ylabel('Loss Function')
ax.grid()
plt.show()

# Training and validation accuracy:

epochs = len(test_history['validation_loss'])
x_axis = range(0,epochs)
fig, ax = plt.subplots(dpi=100)
ax.plot(x_axis, train_history['accuracy'], '-', label='Training accuracy', color='r')
ax.plot(x_axis, test_history['validation_accuracy'], '-', label='Validation accuracy', color='g')
ax.legend()
plt.ylabel('Accuracy')
ax.grid()
plt.show()


