import argparse
import train_utils

parser = argparse.ArgumentParser(
    description='This script helps in training the model',
)

parser.add_argument('--data_directory', dest='data_directory', action='store', default='./flowers')
parser.add_argument('--model_name', dest='model_name', action='store', default='vgg16')
parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth')
parser.add_argument('--learning_rate', dest='learning_rate', action='store', default=0.001, type=float)
parser.add_argument('--hidden_input', dest='hidden_input',  action='store', default=1024, type=int)
parser.add_argument('--epochs', dest='epochs', action='store', default=5, type=int)
parser.add_argument('--gpu', dest="mode", action="store", default="gpu")

args = parser.parse_args()

# fetch dataloaders
train_data, train_dataloader, test_dataloader, validate_dataloader = train_utils.load_data(args.data_directory)

# setup the classifier, criterion, optimizer model
model, optimizer, criterion = train_utils.create_model(
    args.model_name, args.hidden_input, args.learning_rate, args.mode)

# train model
train_utils.train_model(model, optimizer, criterion, train_dataloader, 
                        validate_dataloader, args.epochs, args.mode)

# save the model as checkpoint
train_utils.save_checkpoint(model, args, optimizer, train_data)