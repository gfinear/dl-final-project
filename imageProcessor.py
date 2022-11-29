import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm

class ImageProcessor():
    def __init__(self, data_folder):
        self.data_folder = data_folder
        
    def get_image_features(self, image_name):
        '''
    Method used to extract the features from the images in the dataset using ResNet50
    '''
        image_features = []
        vis_images = []
        resnet = tf.keras.applications.ResNet50(False)  ## Produces Bx7x7x2048
        gap = tf.keras.layers.GlobalAveragePooling2D()  ## Produces Bx2048
        img_path = f'{self.data_folder}/{image_name}'
        with Image.open(img_path) as img:
            img_array = np.array(img.resize((224,224)))
        img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]
        image_features += [gap(resnet(img_in))]
        data =  dict(
            image_features    = np.array(image_features),
            images            = np.array(vis_images),
        )

        with open(f'{self.data_folder}/image.p', 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
        print(f'Data has been dumped into {self.data_folder}/image.p!')

        return None
