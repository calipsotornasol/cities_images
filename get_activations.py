import numpy as np
import os, argparse
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.python.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Flatten

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dir',  type=str, help="Source directory for images")
parser.add_argument('-r', '--res',  type=int, default=224, help="Width/height of output square image")
parser.add_argument('-p', '--path', type=str, default='./', help="Destination directory for output image")

args = parser.parse_args()
in_dir = args.dir
out_res = args.res

def build_model():
    base_model = VGG16(weights='imagenet')
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    return Model(inputs=base_model.input, outputs=top_model(base_model.output))
    
def load_img(in_dir):
    pred_img = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    img_collection = []
    for idx, img in enumerate(pred_img):
        img = os.path.join(in_dir, img)
        if os.path.isdir(img): continue
        #Preparing to trim, converting to an array
        try: 
            img_2=image.load_img(img,target_size=(out_res, out_res))    
            x = image.img_to_array(img_2)
            s = int(x.shape[0]*0.05)
            x=x[s:-s,s:-s,:] #Removing borders
            #Converting again
            img_2 = image.array_to_img(x)
            img_collection.append(img_2)
        except:
            print("Error with {}".format(img))
   
    return img_collection, pred_img
    
def get_activations(model, img_collection):
    activations = []
    for idx, img in enumerate(img_collection):
        print("Processing image {}".format(idx+1))
        img = img.resize((224, 224), Image.ANTIALIAS)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        activations.append(np.squeeze(model.predict(x)))
    return activations
    
model = build_model()
img_collection, pred_img = load_img(in_dir)
activations = get_activations(model, img_collection)
f = open('get_activations_out', "wb")
f.write(pickle.dumps(activations))
f.close()
