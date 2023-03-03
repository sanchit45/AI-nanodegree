import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import transforms,datasets,models
from PIL import Image
import json



def load_checkpoint(arguments):
    checkpoint=torch.load(arguments.checkpoint)
    model_arch=checkpoint['architecture']
    model={"densenet":models.densenet121(pretrained=True),"vgg": models.vgg16(pretrained=True)}
    model=model[model_arch]
    model.classifier=checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx=checkpoint["mapping"]
    
    for param in model.parameters():  #to freeze the parameters
        param.requires_grad=False
        
    return model


###image preprocessing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    p_image=Image.open(image)
    
    preprocess=transforms.Compose ([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    image=preprocess(p_image)
    
    return image

###plotting image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    #image=image/255
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

###Class Prediction
def predict( model, arguments):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval()
    device= arguments.gpu
    model.to(device)
    
    image=process_image(arguments.image)
    image=image.unsqueeze_(0).float() 
    
      
    ##was getting typeerror "Can’t convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first"
    ##adding .cpu() fixed erroe. 
    ##https://stackoverflow.com/questions/53900910/typeerror-can-t-convert-cuda-tensor-to-numpy-use-tensor-cpu-to-copy-the-tens
    
    
    with torch.no_grad():
        output=model.forward(image.to(device))
    ps=torch.exp(output)
    top_probs, top_classes =ps.topk(arguments.top_k, dim=1)
    
    class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
    classes=[]
    for classe in top_classes.cpu().numpy()[0]:
        classes.append(class_to_idx_inv[classe])
    
    return top_probs.cpu().numpy()[0].tolist() ,classes
    


def main():
    #parser 
    parser=argparse.ArgumentParser()

    parser.add_argument("checkpoint", help="file in which checkpoint for model have been saved")
    parser.add_argument("--image",default="flowers/test/64/image_06104.jpg",help="image to predict  ")
    parser.add_argument("--category_names",default='cat_to_name.json',help="file containing mapping of categories to real names")
    parser.add_argument("--top_k",default=5,help="Return top K most likely classes ")
    parser.add_argument("--gpu",default="cuda" if torch.cuda.is_available() else "cpu",help="torchvision model ")
    
    arguments=parser.parse_args()
    model=load_checkpoint(arguments)
    
    
    processed_image=process_image(arguments.image)
    
    probs,classes=predict(model,arguments)
    
    ###label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
        num_output=len(cat_to_name)
    ###printing flowers name with probabilities  
    flowers_name=[]
    for classe in classes :
        flowers_name.append(cat_to_name[classe])
        
    
    
    for i,j in zip(flowers_name,probs):
        print(i+" : ", j)
        
    
    
    


    
    
    
    
if __name__== "__main__":
    main()
