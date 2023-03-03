import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import transforms,datasets,models
from collections import OrderedDict
from PIL import Image
import json


###Load the data
def load_the_data(arguments):
    data_dir = arguments.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms=transforms.Compose([transforms.Resize(225),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    test_transforms=transforms.Compose([transforms.Resize(225),
                                        transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    train_datasets=datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_datasets=datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_datasets=datasets.ImageFolder(test_dir,transform=test_transforms)


    train_dataloaders = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets,batch_size=64)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets,batch_size=64)
    
    return train_datasets,train_dataloaders,valid_dataloaders,test_dataloaders


###Building and training the classifier

##building model
def building_model(arguments):
    architecture={"vgg":25088,"densenet": 1024}

    model={"densenet":models.densenet121(pretrained=True),"vgg": models.vgg16(pretrained=True)}

    model=model[arguments.arch]


    #freeezing parameters of features so we dont backprop them 
    for param in model.parameters():
        param.requires_grad=False


    classifier=nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(architecture[arguments.arch], arguments.hidden_unit)),
                              ('relu', nn.ReLU()),
                               ("drop1",nn.Dropout(0.08)),
                              ('fc2', nn.Linear(arguments.hidden_unit, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier=classifier
    return model


##training 
def training_model(model,arguments,train_dataloaders,valid_dataloaders):
    device= torch.device(arguments.gpu) #use gpu if available
    criterion=nn.NLLLoss()

    optimizer=optim.Adam(model.classifier.parameters(),lr=arguments.learning_rate)

    model.to(device);

    epochs= int(arguments.epochs)
    steps =0
    running_loss=0
    print_every=5


    for epoch in range(epochs):
        for inputs,label in train_dataloaders:
            steps+=1
            inputs,label=inputs.to(device),label.to(device)

            logps=model.forward(inputs)
            loss=criterion(logps,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            if steps%print_every==0:
                test_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for inputs,labels in valid_dataloaders:
                        inputs,labels=inputs.to(device),labels.to(device)
                        logps=model.forward(inputs)
                        batch_loss=criterion(logps,labels)

                        test_loss +=batch_loss.item()

                        #calculating accuracy
                        ps=torch.exp(logps)
                        top_p,top_class=ps.topk(1,dim=1)
                        equals=top_class==labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(valid_dataloaders):.3f}.. "
                      f"Test accuracy: {accuracy/len(valid_dataloaders):.3f}")
                running_loss = 0
                model.train()
    return model
### Testing
def testing_model(model,arguments,test_dataloaders):
    accuracy=0
    device= torch.device(arguments.gpu)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs,labels in test_dataloaders:

            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model.forward(inputs)
            ps=torch.exp(outputs)
            top_p,top_class=ps.topk(1,dim=1)
            equals=(top_class==labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy += equals.mean()
    print("accuracy :{} %".format(100*accuracy/len(test_dataloaders)))


###saving the checkpoint
def saving_checkpoint(model,arguments,train_datasets):
    model.to("cpu")
    model.class_to_idx=train_datasets.class_to_idx

    checkpoint={"classifier":model.classifier,
               "state_dict":model.state_dict(),
               "mapping": model.class_to_idx,
               "architecture":arguments.arch}
    torch.save(checkpoint,arguments.save_dir)
    print("Checkpoint saved at {}".format(arguments.save_dir))
    
    
###main function

def main():
    #parser and all inputs needed 
    parser=argparse.ArgumentParser()

    parser.add_argument("data_dir", help="folder in which flowers are saved ")
    parser.add_argument("--arch",default="densenet",help="torchvision model[vgg,densenet] ")
    parser.add_argument("--gpu",default="cuda" if torch.cuda.is_available() else "cpu",help="torchvision model ")
    parser.add_argument("--learning_rate",default=0.003,help="learning rate ")
    parser.add_argument("--epochs",default=5)
    parser.add_argument("--hidden_unit",default=500)
    parser.add_argument("--dropout",default=0.08)
    parser.add_argument("--save_dir",default="checkpoint.pth",help="folder path in which our checkpoint will be saved ")



    arguments=parser.parse_args()
    train_datasets,train_dataloaders,valid_dataloaders,test_dataloaders=load_the_data(arguments)
    model=building_model(arguments)
    model= training_model(model,arguments,train_dataloaders,valid_dataloaders)
    testing_model(model,arguments,test_dataloaders)
    saving_checkpoint(model,arguments,train_datasets)
    
    
    
    
if __name__== "__main__":
    main()
