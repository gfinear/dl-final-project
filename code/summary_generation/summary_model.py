import pickle

from .decoder import Decoder
from .encoder import Encoder

class SummaryModel():

    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.caption_style, self.summary_style = self.load_styles()

    def __call__(self, caption):
        if isinstance(caption, str):
            caption = [caption]
        caption_skipthought = self.encoder(caption)
        summary_skipthought = self.transfer_style(caption_skipthought)
        summary = self.decoder(summary_skipthought)
        return summary

    def transfer_style(self, embedding):
        return embedding - self.caption_style + self.summary_style

    def load_styles(self):
        embedding_path = 'summary_generation/skipthought_embeddings/'
        caption_path = embedding_path + 'caption_skipthoughts.p'
        summary_path = embedding_path + 'summary_skipthoughts.p'

        with open(caption_path, 'rb') as cf, open(summary_path, 'rb') as sf:
            caption_data = pickle.load(cf)
            summary_data = pickle.load(sf)

        caption_style = caption_data['style']
        summary_style = summary_data['style']

        return caption_style, summary_style