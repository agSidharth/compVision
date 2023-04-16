import numpy as np
import os
import torch, torchvision
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import random
import copy

# Testing on the test data
def calAccuracy(model,dataloader):

    # no need to track the forward computation
    torch.set_grad_enabled(False)
    correct,total = 0,0
    tempAxis = 1
    
    for data in dataloader:
        inputs,labels = data
        total += labels.size(0)
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        _,predicted = torch.max(model(inputs).data,tempAxis)
        correct += (predicted==labels).sum().item()
        
    torch.set_grad_enabled(True)
    finalAcc = (correct*100)/total
    print("The accuracy of the model : "+str(finalAcc))
    return finalAcc

# training model function
def trainModel(model,loss_fn,optimizer,EPOCHS,EPSILON):
    
    last_loss,max_valAcc = ((np.inf)/4),0
    finalModel = copy.deepcopy(model)
    
    for epoch in range(EPOCHS):
        this_loss = 0.0

        for idx,data in enumerate(trainloader,0):
            if(DEBUG and idx%1==0): print("Iteration : "+str(idx))
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            loss = loss_fn(model(inputs),labels)
            loss.backward()
            optimizer.step()

            this_loss += loss
        
        this_loss = this_loss/len(trainloader)
        if(abs(this_loss-last_loss)<EPSILON): break
        last_loss = this_loss
        
        print("Epoch : "+str(epoch)+", Loss ==> "+str(last_loss))
        print("Validation ==>")
        
        this_valAcc = calAccuracy(model,validloader,device)
        
        # Save the model only if validation accuracy is greater than previous max
        if(this_valAcc>max_valAcc):
            max_valAcc = this_valAcc
            finalModel = copy.deepcopy(model)
    
    print("Training finished...Testing ==>")
    testing_Acc = calAccuracy(model,testloader,device)
    
    return finalModel,max_valAcc

class ourDataset(Dataset):
    def __init__(self,dataDir):
        listOfDir = os.listdir(dataDir)
        listOfDir.sort()
        if ".DS_Store" in listOfDir: listOfDir.remove(".DS_Store")
        
        if len(labelsDict.keys())==0:
            index = 0
            for label in listOfDir:
                labelsDict[label] = index
                index += 1
            pass
        
        self.images = []
        self.labels = []

        for dir in listOfDir:
            for file in os.listdir(os.path.join(dataDir,dir)):
                self.images.append(os.path.join(dataDir,dir,file))
                self.labels.append(labelsDict[dir])
            pass
        pass
        
        temp = list(zip(self.images,self.labels))
        random.shuffle(temp)
        temp1,temp2 = zip(*temp)
        self.images,self.labels = list(temp1),list(temp2)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        thisImg = Image.open(self.images[idx])
        thisTransform = transforms.Compose([transforms.PILToTensor()])
        img_tensor = thisTransform(thisImg)/255
        
        return img_tensor,self.labels[idx]

labelsDict = {}

# Initializing seed to maintain consistency
SEED = 661
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

# Using various hyperparameters 
EPOCHS = 4
EPSILON = 1e-3
DEBUG = True 
LR = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 64
REGULARIZATION = True
WEIGHT_DECAY = 0
if REGULARIZATION: WEIGHT_DECAY = 1e-5

# Take cuda if it is available
thisDevice = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(thisDevice)
print("The device in use is : "+thisDevice)


dataLoc = sys.argv[1]
trainDataset = ourDataset(os.path.join(dataLoc,"train"))
validDataset = ourDataset(os.path.join(dataLoc,"valid"))
testDataset  = ourDataset(os.path.join(dataLoc,"test"))

trainloader = DataLoader(trainDataset,batch_size=BATCH_SIZE,shuffle=True)
validloader = DataLoader(validDataset,batch_size=BATCH_SIZE,shuffle=True)
testloader = DataLoader(testDataset,batch_size=BATCH_SIZE,shuffle=True)

"""
print(trainDataset.__len__())
print(validDataset.__len__())
print(testDataset.__len__())
print(trainDataset[0][0].size())
plt.imshow(trainDataset[0][0].numpy().transpose(1, 2, 0))
print(trainDataset[0][1])
plt.show()
"""

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model.classifier[6] = torch.nn.Linear(4096,len(labelsDict))
model = model.to(device)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = LR,momentum = MOMENTUM,weight_decay=WEIGHT_DECAY)

finalModel,testAcccu = trainModel(model,loss_fn,optimizer,EPOCHS,EPSILON)

torch.save(finalModel.state_dict(),"output.pth")
print("Final Training Accuracy ==>")
calAccuracy(finalModel,trainloader)
print("Final Validation Accuracy ==>")
calAccuracy(finalModel,validloader)
print("Final Testing Accuracy ==>")
calAccuracy(finalModel,testloader)
