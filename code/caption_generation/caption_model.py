import os
import argparse
import numpy as np
import pickle
import tensorflow as tf
from typing import Optional
from types import SimpleNamespace


from .model import ImageCaptionModel, accuracy_function, loss_function
from .decoder import TransformerDecoder
from .image_processor import ImageProcessor
from . import transformer

class CaptionModel():
    def __init__(self):
        self.data_file = '../../data/caption_data.p'
        self.model_location = 'transform_model'
    
    def __call__(self, photo):
        with open(self.data_file, 'rb') as data_file:
            data_dict = pickle.load(data_file)

        feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 5, axis=0)
        img_prep  = lambda x: np.repeat(x, 5, axis=0)
        train_captions  = np.array(data_dict['train_captions'])
        test_captions   = np.array(data_dict['test_captions'])
        train_img_feats = feat_prep(data_dict['train_image_features'])
        test_img_feats  = feat_prep(data_dict['test_image_features'])
        word2idx        = data_dict['word2idx']

        model = tf.keras.models.load_model(
        self.model_location,
        custom_objects=dict(
            AttentionHead           = transformer.AttentionHead,
            AttentionMatrix         = transformer.AttentionMatrix,
            MultiHeadedAttention    = transformer.MultiHeadedAttention,
            TransformerBlock        = transformer.TransformerBlock,
            PositionalEncoding      = transformer.PositionalEncoding,
            TransformerDecoder      = TransformerDecoder,
            ImageCaptionModel       = ImageCaptionModel
            ),
        )
        photo_process = ImageProcessor('../../data')
        photo = photo_process.get_image_features(photo)
        return self.gen_caption_temperature(model, photo, word2idx, word2idx['<pad>'], .5, 20)

    def gen_caption_temperature(self, model, image_embedding, wordToIds, padID, temp, window_length):
        idsToWords = {id: word for word, id in wordToIds.items()}
        unk_token = wordToIds['<unk>']
        caption_so_far = [wordToIds['<start>']]
        while len(caption_so_far) < window_length and caption_so_far[-1] != wordToIds['<end>']:
            caption_input = np.array([caption_so_far + ((window_length - len(caption_so_far)) * [padID])])
            logits = model(np.expand_dims(image_embedding, 0), caption_input)
            logits = logits[0][len(caption_so_far) - 1]
            probs = tf.nn.softmax(logits / temp).numpy()
            next_token = unk_token
            attempts = 0
            while next_token == unk_token and attempts < 5:
                next_token = np.random.choice(len(probs), p=probs)
                attempts += 1
            caption_so_far.append(next_token)
        return ' '.join([idsToWords[x] for x in caption_so_far][1:-1])

if __name__ == '__main__':
    
    caption = CaptionModel()
    caption.caption('Unknown.jpeg')
