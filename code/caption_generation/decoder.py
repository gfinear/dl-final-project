import tensorflow as tf

try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        self.image_embedding = tf.keras.Sequential([tf.keras.layers.Dense(hidden_size, activation = 'relu'), tf.keras.layers.Dense(hidden_size)])

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(self.hidden_size)

        # Define classification layer (logits)
        self.classifier = tf.keras.Sequential([tf.keras.layers.Dense(hidden_size, activation = 'relu'), tf.keras.layers.Dense(vocab_size)])

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        reshape_images = self.image_embedding(tf.expand_dims(encoded_images, 1))
        # 2) Pass the captions through your positional encoding layer
        reshape_captions = self.encoding(captions)
        # 3) Pass the english embeddings and the image sequences to the decoder
        reshape_captions= self.decoder(reshape_captions, reshape_images)
        # 4) Apply dense layer(s) to the decoder out to generate logits
        probs = self.classifier(reshape_captions)
        return probs
    
    def get_config(self):
        return {k:getattr(self, k) for k in "vocab_size hidden_size window_size".split()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
