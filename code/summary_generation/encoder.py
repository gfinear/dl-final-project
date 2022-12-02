from .skipthought_ryankiros.skipthoughts import load_model, Encoder as SkipthoughtEncoder

class Encoder():
    def __init__(self):
        self.model = load_model()
        self.skipthought_encoder = SkipthoughtEncoder(self.model)

    def __call__(self, text):
        return self.skipthought_encoder.encode(text)