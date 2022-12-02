from caption_generation import CaptionModel
from summary_generation import SummaryModel

class Img2Plot():

    def __init__(self):
        self.caption_model = CaptionModel()
        self.summary_model = SummaryModel()

    def __call__(self, image_path):
        caption = self.caption_model(image_path)
        summary = self.summary_model(caption)
        return summary
