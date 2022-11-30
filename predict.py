import argparse
from utility import *


parser = argparse.ArgumentParser(description='get training parameters')

parser.add_argument('img_path', help='image location')
parser.add_argument('--checkpoint', help="spesify the checkpoint's name")
parser.add_argument('--gpu', action='store_false', help="set the model to train in GPU mode")
parser.add_argument('--top_k', help="k most likely classes")
parser.add_argument('--category_names', help="spesify the model's hidden layers")

arg = parser.parse_args()

image = process_image(arg.img_path)
model = load_checkpoint(arg.checkpoint)
ps, classes = predict(image, model, model.class_to_idx, arg.gpu, arg.top_k, arg.category_names)

print('flower name: probability')
for i in range(len(ps)):
    print(f"{classes[i]}: {ps[i]}")









