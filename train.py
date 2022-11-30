import argparse
from functions import *




parser = argparse.ArgumentParser(description='get training parameters')

parser.add_argument('data_directory', help='images location')
parser.add_argument('--gpu', action='store_false', help="set the model to train in GPU mode")
parser.add_argument('--arch', help="spesify the model's architecture")
parser.add_argument('--learning_rate', help="spesify the model's learning")
parser.add_argument('--hidden_units', help="spesify the model's hidden layers")
parser.add_argument('--epochs', help="spesify the model's epochs")
parser.add_argument('--save_dir', help='location to save the model')

arg = parser.parse_args()

trainloader, valloader, train_data = data_mani(arg.data_directory)
model = train(trainloader, valloader, arg.learning_rate, arg.hidden_units, arg.epochs, arg.gpu, arg.arch)
save(arg.save_dir, model, train_data)


