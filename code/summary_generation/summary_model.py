import numpy as np
import pickle
import tensorflow as tf

from .decoder import Decoder
from .encoder import Encoder

class SummaryModel():

    def __init__(self, temperature=0.05, length=251):
        self.temperature = temperature
        self.length = length

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.caption_style, self.summary_style = self.load_styles()
        self.wordToId = self.load_summary_data()

    def __call__(self, caption):
        if isinstance(caption, str):
            caption = [caption]
        caption_skipthought = self.encoder(caption)
        summary_skipthought = self.transfer_style(caption_skipthought)
        summary = self.generate_summary(summary_skipthought)
        return summary

    def transfer_style(self, embedding):
        return embedding - self.caption_style + self.summary_style

    def generate_summary(self, skipthought):
        idsToWords = {id: word for word, id in self.wordToId.items()}
        unk_token = self.wordToId['<unk>']
        padId = self.wordToId['<pad>']
        summary = [self.wordToId['<start>']]
        while len(summary) < self.length and summary[-1] != self.wordToId['<end>']:
            summary_input = np.array([summary + ((self.length - len(summary)) * [padId])])
            logits = self.decoder((summary_input, skipthought))
            logits = logits[0][len(summary) - 1]
            probs = tf.nn.softmax(logits / self.temperature).numpy()
            next_token = unk_token
            attempts = 0
            while next_token == unk_token and attempts < 5:
                next_token = np.random.choice(len(probs), p=probs)
                attempts += 1
            summary.append(next_token)
        return ' '.join([idsToWords[x] for x in summary][1:-1])

    def load_styles(self):
        embedding_path = 'summary_generation/skipthought_embeddings/'
        caption_path = embedding_path + 'caption_skipthoughts.p'
        summary_path = embedding_path + 'summary_skipthoughts.p'

        with open(caption_path, 'rb') as cf, open(summary_path, 'rb') as sf:
            caption_data = pickle.load(cf)
            summary_data = pickle.load(sf)

        caption_style = caption_data['train_style']
        summary_style = summary_data['train_style']

        return caption_style, summary_style

    def load_summary_data(self):
        summary_data_path = '../data/summary_data.p'
        with open(summary_data_path, 'rb') as f:
            summary_data = pickle.load(f)

        wordToIds = summary_data['word2idx']
        return wordToIds