import argparse
import predict_utils

parser = argparse.ArgumentParser(
    description='This script helps in predicting the model',
)

parser.add_argument('--image_path', dest='image_path', action='store', 
                    default='./flowers/valid/100/image_07895.jpg')
parser.add_argument('--checkpoint_path', dest='checkpoint_path', action='store', default='checkpoint.pth')
parser.add_argument('--top_k', dest='top_k', action='store', default=5, type=int)
parser.add_argument('--gpu', dest="mode", action="store", default="gpu")

args = parser.parse_args()


# load the checkpoint
checkpoint_model = predict_utils.load_checkpoint(args.checkpoint_path)

# predict the class of an image
probs, classes = predict_utils.predict(args.image_path, checkpoint_model, args.top_k)

for i in range(args.top_k):
    print("Probability - {} - Class - {}".format(probs[i], classes[i]))