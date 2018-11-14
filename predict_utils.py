import torch 

from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms

# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model_name = checkpoint['model_name']
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model. Did you mean vgg16, densenet121, or alexnet?".format(model_name))
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image_path):
    pil_image = Image.open(image_path)   
    image_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                          std=(0.229, 0.224, 0.225))
                                    ])  
    pil_image = image_transforms(pil_image) 
    return pil_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1) 
    ax.imshow(image) 
    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cpu()
    
    image = process_image(image_path)
    
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        output = model.forward(image)
        probs, labels = torch.topk(output, topk)        
        probs = probs.exp()
        
        class_to_idx_rev = {model.class_to_idx[k]: k for k in model.class_to_idx}
        
        classes = []
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_rev[label])
    
#     print("The category name of the image selected is - {}".format(cat_to_name['100']))

    return probs.numpy()[0], classes
        