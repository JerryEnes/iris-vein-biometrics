from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from PIL import Image
import io

app = FastAPI(
    title="Iris-Vein Multimodal Biometrics API",
    description="Vision Transformer based multimodal biometric authentication system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Iris-Vein Multimodal Biometrics API",
        "version": "1.0.0",
        "model": "Vision Transformer",
        "status": "operational"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": "cpu"}

@app.post("/test")
async def test_endpoint():
    return {"message": "API is working correctly"}

@app.post("/process")
async def process_images(
    iris_image: UploadFile = File(...),
    vein_image: UploadFile = File(...)
):
    """Process iris and vein images"""
    try:
        # Read images
        iris_content = await iris_image.read()
        vein_content = await vein_image.read()
        
        # Convert to numpy arrays
        iris_np = np.frombuffer(iris_content, np.uint8)
        vein_np = np.frombuffer(vein_content, np.uint8)
        
        iris_img = cv2.imdecode(iris_np, cv2.IMREAD_COLOR)
        vein_img = cv2.imdecode(vein_np, cv2.IMREAD_COLOR)
        
        if iris_img is None or vein_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        return {
            "message": "Images processed successfully",
            "iris_shape": iris_img.shape,
            "vein_shape": vein_img.shape
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
