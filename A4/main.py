import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import sys
import random
import copy
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# Testing on the test data
def calAccuracy(model,dataloader):

    # no need to track the forward computation
    model.eval()
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
    finalAcc = correct/total
    finalAcc = finalAcc*100
    print("The accuracy of the model : "+str(finalAcc))
    return finalAcc

# training model function
def trainModel(model,loss_fn,optimizer,EPOCHS,EPSILON):
    
    loss_list = []
    last_loss,max_valAcc = ((np.inf)/4),0
    model = model.to(device)
    model.train()
    finalModel = copy.deepcopy(model)
    
    for epoch in range(EPOCHS):
        this_loss = 0.0

        for idx,data in enumerate(trainloader,0):
            if(DEBUG and idx%10==0): print("Iteration : "+str(idx))
            inputs,labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            
            optimizer.zero_grad()
            loss = loss_fn(model(inputs),labels)
            loss.backward()
            optimizer.step()

            this_loss = this_loss + loss + 0.0
        
        this_loss = this_loss/len(trainloader)
        loss_list.append(this_loss.cpu().detach().numpy())
        if(abs(this_loss-last_loss)<EPSILON): 
            print("Terminating early")
            break
        last_loss = this_loss
        
        print("Epoch : "+str(epoch)+", Loss ==> "+str(last_loss))
        print("Validation ==>")
        
        this_valAcc = calAccuracy(model,validloader)
        
        # Save the model only if validation accuracy is greater than previous max
        if(this_valAcc>max_valAcc):
            print("Better model trained")
            max_valAcc = this_valAcc
            finalModel = copy.deepcopy(model)
    
    print("\n\nTraining finished...Testing ==>")
    model = copy.deepcopy(finalModel)
    testing_Acc = calAccuracy(model,testloader)
    
    return model,max_valAcc,loss_list

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
        if thisImg.mode=="L": thisImg = thisImg.convert('RGB')
        thisTransform = transforms.Compose([transforms.PILToTensor()])
        img_tensor = thisTransform(thisImg)/255
        
        return img_tensor,self.labels[idx]

labelsDict = {}

# Initializing seed to maintain consistency
SEED = 661
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Using various hyperparameters 
BATCH_SIZE = 64
EPOCHS = 20
EPSILON = 1e-3
REGULARIZATION = True
WEIGHT_DECAY = 0
DEBUG = True 
LR = 0.001
MOMENTUM = 0.9

if REGULARIZATION: 
    WEIGHT_DECAY = 1e-5
    print("Regularization is being used")

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
for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = torch.nn.Linear(4096,len(labelsDict))

for name,param in model.named_parameters():
    if param.requires_grad==True:
        print("\t",name)

loss_fn,model = torch.nn.CrossEntropyLoss(),model.to(device)
optimizer = torch.optim.SGD(model.parameters(),lr = LR,momentum = MOMENTUM,weight_decay=WEIGHT_DECAY)

model,testAcccu,loss_list = trainModel(model,loss_fn,optimizer,EPOCHS,EPSILON)

print("\n\n")
torch.save(model.state_dict(),"output.pth")
print("Final Training Accuracy ==>")
calAccuracy(model,trainloader)
print("Final Validation Accuracy ==>")
calAccuracy(model,validloader)
print("Final Testing Accuracy ==>")
temp = calAccuracy(model,testloader)

plt.plot(loss_list)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("Convergence.png")
