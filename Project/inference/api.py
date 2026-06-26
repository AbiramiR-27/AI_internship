import io
import torch
import numpy as np

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

from model import ISNetDIS

# FastAPI App
app = FastAPI(
    title="ISNet Segmentation API",
    description="Inference API for ISNet trained on DIS-5K",
    version="1.0"
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load Model
MODEL_PATH = "isnet_model.pth"

model = ISNetDIS().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("ISNet model loaded successfully.")

# Image Preprocessing

def preprocess_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    original_size = image.size

    image = image.resize(
        (512, 512)
    )

    image = np.array(image).astype(np.float32)

    image = image / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    image = image.unsqueeze(0)

    image = image.to(device)

    return image, original_size


# Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    input_tensor, original_size = preprocess_image(
        image_bytes
    )

    with torch.no_grad():

        outputs = model(input_tensor)

        # Handle ISNet outputs
        if isinstance(outputs, tuple):

            ds = outputs[0]

            if isinstance(ds, list):
                pred = ds[0]
            else:
                pred = ds

        elif isinstance(outputs, list):

            pred = outputs[0]

        else:

            pred = outputs

    pred = pred.squeeze()

    pred = pred.cpu().numpy()

    pred = (
        pred - pred.min()
    ) / (
        pred.max() - pred.min() + 1e-8
    )

    pred = (
        pred * 255
    ).astype(np.uint8)

    mask = Image.fromarray(pred)

    mask = mask.resize(
        original_size
    )

    buffer = io.BytesIO()

    mask.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png"
    )


# Health Check
@app.get("/")
def home():

    return {
        "message": "ISNet Segmentation API is running"
    }