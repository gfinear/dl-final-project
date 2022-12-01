from caption_generation.caption_model import CaptionModel
from summary_generation.summary_model import SummaryModel

class Img2Plot():

    def __init__(self):
        self.caption_model = CaptionModel()
        self.summary_model = SummaryModel()

    def __call__(self, image_path):
        caption = self.caption_model(image_path)
        summary = self.summary_model(caption)
        return summary
