import pytesseract
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, SkipValidation

class OCRModelInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: SkipValidation[Image]
    timeout: int = 10

class OCRModelOutput(BaseModel):
    text: str

class OCRModel:
    def __init__(self):
        self.ocr = pytesseract

    def __call__(self, input: OCRModelInput) -> OCRModelOutput:
        pred = self.ocr.image_to_string(input.image)
        return OCRModelOutput(text=pred)

    @staticmethod
    def version():
        return str(pytesseract.get_tesseract_version())
