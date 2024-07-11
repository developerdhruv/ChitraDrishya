from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from PIL import Image
import torch
import tempfile
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

model.renderer.set_chunk_size(8192)
model.to(device)

def preprocess(input_image: Image.Image, do_remove_background: bool = True, foreground_ratio: float = 0.85):
    # Function to fill background
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        input_image = Image.open(file.file).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error opening image: {e}")

    try:
        # Preprocess the image
        preprocessed_image = preprocess(input_image)

        # Check the shape of the image tensor
        image_tensor = torch.from_numpy(np.array(preprocessed_image)).float()
        print(f"Image tensor shape: {image_tensor.shape}")

        # Generate the GLB file
        scene_codes = model(preprocessed_image, device=device)
        mesh = model.extract_mesh(scene_codes, True, resolution=256)[0]

        # Ensure the mesh is correctly oriented
        mesh = to_gradio_3d_orientation(mesh)
        
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as mesh_file:
            mesh.export(mesh_file.name)
            return FileResponse(mesh_file.name, media_type="application/octet-stream", filename="output.glb")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
