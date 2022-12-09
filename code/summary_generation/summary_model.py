import numpy as np
import pickle
import tensorflow as tf

from .decoder import Decoder
from .encoder import Encoder

class SummaryModel():

    def __init__(self, temperature=0.5, length=151, k=50):
        self.temperature = temperature
        self.length = length
        self.k = k

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.caption_style, self.summary_style = self.load_styles()
        self.data = self.load_summary_data()
        
        self.wordToId = self.data['wordToId']
        self.summaries = self.data['summaries']
        self.skipthoughts = self.data['skipthoughts']

    def __call__(self, caption):
        if isinstance(caption, str):
            caption = [caption]
        caption_skipthought = self.encoder(caption)
        summary_skipthought = self.transfer_style(caption_skipthought)
        summary = self.generate_summary(summary_skipthought)
        return summary

    def transfer_style(self, embedding):
        return embedding #- self.caption_style + self.summary_style

    def generate_summary(self, skipthought):
        idsToWords = {id: word for word, id in self.wordToId.items()}
        unk_token = self.wordToId['<unk>']
        padId = self.wordToId['<pad>']
        endId = self.wordToId['<end>']
        summary = [self.wordToId['<start>']]
        while len(summary) < self.length and summary[-1] != self.wordToId['<end>']:
            summary_input = np.array([summary + ((self.length - len(summary)) * [padId])])
            logits = self.decoder((summary_input, skipthought))
            logits = logits[0][len(summary) - 1]
            probs = tf.nn.softmax(logits / self.temperature).numpy()
            next_token = unk_token
            attempts = 0
            while next_token in {unk_token, endId, padId} and attempts < 3:
                # next_token = np.random.choice(len(probs), p=probs)
                top = np.argpartition(probs, -self.k)[-self.k:]
                k_prob = probs[top] / np.sum(probs[top])
                next_token = np.random.choice(top, p=k_prob)
                if next_token != endId or len(summary) >= 100:
                    attempts += 1
            summary.append(next_token)
        return ' '.join([idsToWords[x] for x in summary][1:-1])

    def generate_summary_knn(self, skipthought):
        idsToWords = {id: word for word, id in self.wordToId.items()}
        distances = np.sum(np.abs(self.skipthoughts - skipthought), axis=1)
        closest = np.argmin(distances)
        summary = self.summaries[closest]
        summary = [idsToWords[i] for i in summary]
        return " ".join(summary)

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

        train_sum = summary_data['train_summaries']
        test_sum = summary_data['test_summaries']
        summaries = np.concatenate((train_sum, test_sum))

        skipthought_path = 'summary_generation/skipthought_embeddings/summary_skipthoughts.p'
        with open(skipthought_path, 'rb') as f:
            skipthought_data = pickle.load(f)

        train_skip = skipthought_data['train_skipthoughts']
        test_skip = skipthought_data['test_skipthoughts']
        skipthoughts = np.concatenate((train_skip, test_skip))

        data = {
            'wordToId': wordToIds,
            'summaries': summaries,
            'skipthoughts': skipthoughts
        }

        return data
