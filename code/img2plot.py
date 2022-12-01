from caption_generation.captionize import Captionizer
from summary_generation.summarize import Summarizer

class Img2Plot():

    def __init__(self):
        self.caption_model = Captionizer()
        self.summary_model = Summarizer()

    def __call__(self, image_path):
        caption = self.caption_model(image_path)
        summary = self.summary_model(caption)
        return summary
