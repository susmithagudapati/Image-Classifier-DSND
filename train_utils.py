import json

import torch

from torch import nn
from torchvision import datasets, models, transforms
    
def load_data(data_dir = './flowers/'):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                                std=(0.229, 0.224, 0.225))
                                          ])
    test_transforms = transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])
                                         ])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                               ])

    # Load the datasets with the help of ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # Using the datasets above and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    validate_dataloader = torch.utils.data.DataLoader(validate_data, batch_size=32, shuffle=True)
    
    return train_data, train_dataloader, test_dataloader, validate_dataloader


def load_json_data():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


# setup the model, optimizer, criterion
def create_model(model_name='vgg16', hidden_input=1024, 
                 learning_rate=0.001, mode='gpu'):
    
    cat_to_name = load_json_data()
    
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model. Did you mean vgg16, densenet121, or alexnet?".format(model_name))


    for parameter in model.parameters():
        parameter.requires_grad = False

    #Hyper parameters
    input_size = model.classifier[0].in_features 
    hidden_inputs = [2048, hidden_input]
    output_size = len(cat_to_name)


    # Build a feed-forward network
    classifier = nn.Sequential(nn.Linear(input_size, hidden_inputs[0]),
                          nn.ReLU(),
                          nn.Dropout(p=0.15),
                          nn.Linear(hidden_inputs[0], hidden_inputs[1]),
                          nn.ReLU(),
                          nn.Dropout(p=0.15),
                          nn.Linear(hidden_inputs[1], output_size),
                          nn.LogSoftmax(dim=1))

    model.classifier = classifier
    
    if torch.cuda.is_available() and mode == 'gpu':
        model.cuda()

    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, optimizer, criterion


# validate the model
def validation(model, criterion, mode, validate_dataloader):
    test_loss = 0
    accuracy = 0
    for images, labels in iter(validate_dataloader):
        
        if torch.cuda.is_available() and mode == 'gpu':
            images = images.to('cuda')
            labels = labels.to('cuda')
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


# trains model
def train_model(model, optimizer, criterion, train_dataloader,
                validate_dataloader, epochs=5, mode='gpu'):
    print("Training the model\n")
    
    print_every = 50
    steps = 0

    for e in range(epochs):
        running_loss = 0
        for images, labels in iter(train_dataloader):

            model.train()

            steps += 1
            
            if torch.cuda.is_available() and mode == 'gpu':
                images = images.to('cuda')
                labels = labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, criterion, mode, validate_dataloader)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Test Loss: {:.3f}...".format(test_loss/len(validate_dataloader)),
                      "Accuracy: {:.3f}...".format(accuracy/len(validate_dataloader))
                     )

                running_loss = 0
            model.train()
    print("Finished training the model\n")
            

# Save the checkpoint
def save_checkpoint(model, args, optimizer, train_data):
    print("Our model: \n\n", model, '\n')
    print("The state dict keys: \n\n", model.state_dict().keys())
    
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {'epochs': args.epochs,
              'model_name': args.model_name,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'optimizer_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
             }

    torch.save(checkpoint, args.save_dir)
    
    print("Saved the model as checkpoint\n")
