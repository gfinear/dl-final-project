import tensorflow as tf

try: from caption_generation.transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.image_embedding = tf.keras.Sequential([tf.keras.layers.Dense(hidden_size, activation = 'relu'), tf.keras.layers.Dense(hidden_size)])

        self.encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        self.decoder = TransformerBlock(self.hidden_size)
        self.classifier = tf.keras.Sequential([tf.keras.layers.Dense(hidden_size, activation = 'relu'), tf.keras.layers.Dense(vocab_size)])

    def call(self, encoded_images, captions):
        reshape_images = self.image_embedding(tf.expand_dims(encoded_images, 1))

        reshape_captions = self.encoding(captions)
        reshape_captions= self.decoder(reshape_captions, reshape_images)
        probs = self.classifier(reshape_captions)
        return probs
    
    def get_config(self):
        return {k:getattr(self, k) for k in "vocab_size hidden_size window_size".split()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
