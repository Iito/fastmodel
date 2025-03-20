# FastModelAPI

A lightweight Python package that turns any Ai inference class with a `__call__` method into a web service, as long as its inputs and outputs use Pydantic models. Designed for ML inference, it simplifies exposing models without manually defining request and response schemas.  

## Features  
- Converts a callable class into a FastAPI-based web service  
- Accepts requests as `multipart/form-data`  
- Supports JSON or streaming responses based on output serializability  
- Built on FastAPI, Pydantic, and Uvicorn  

## Installation  

### Using pip

```bash
pip install fastmodel
```

### From sources
```bash
git clone git@github.com:Iito/fastmodel.git
cd fastmodel
pip install .
```

## Usage  

Here's an example of exposing an OCR model using `pytesseract`:  

```python
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
```
You can try this examples as follow:
- tesseract and pytesseract must be installed on the host machine.

```bash
export PYTHONPATH=`pwd`/examples:$PYTHONPATH
fastmodel serve ocr.OCRModel
```
![](demo.gif)

### Request Example  

Send an image for OCR processing using `curl`:  

```bash
curl -X POST 'http://localhost:8000/' \
--form 'image=@"/path/to/image"'
--form 'timeout="20"'
```

### Response Example  

```python
{
  "status": int,
  "message": str,
  "version": str,
  "text": str
}
```

## Limitations  
- Only works with **Uvicorn** (Gunicorn is not supported).  
- **Single worker only** due to the way the server is handled.  
- No customization options yet.  

## Why This Exists  
FastAPI is great with native python types and Pydantic models, but manually defining request and response models is tedious. This package automates that, making it easier to serve ML models without extra boilerplate.  
