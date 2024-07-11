import io
import tempfile
from typing import List

import numpy as np
import rembg
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

app = FastAPI()

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# adjust the chunk size to balance between speed and memory usage
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()


def fill_background(image):
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def preprocess(input_image, do_remove_background=True, foreground_ratio=0.85):
    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image, mc_resolution=256, formats=["glb"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('png', 'jpg', 'jpeg')):
        raise HTTPException(status_code=400, detail="Invalid image format")

    image = Image.open(io.BytesIO(await file.read()))
    preprocessed = preprocess(image, do_remove_background=True, foreground_ratio=0.85)
    _, mesh_name_glb = generate(preprocessed, mc_resolution=256, formats=["glb"])
    return FileResponse(mesh_name_glb, media_type="application/octet-stream", filename="output.glb")
