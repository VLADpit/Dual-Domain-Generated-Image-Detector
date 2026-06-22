import os
import io
import logging
import torch
import torch.nn as nn
import torch.fft as fft
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# 1. Конфигурация и логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "weights\best_model_dual_domain.pth")
IMG_SIZE = int(os.getenv("IMG_SIZE", "256"))

logger.info(f"Запуск на устройстве: {DEVICE}")
logger.info(f"Путь к модели: {MODEL_PATH}")

# 2. Частотные утилиты
def comp_log_spectrum(image_tensor):
    if image_tensor.shape[1] == 3:
        gray = 0.299 * image_tensor[:, 0] + 0.587 * image_tensor[:, 1] + 0.114 * image_tensor[:, 2]
    else:
        gray = image_tensor[:, 0]
    
    fft_result = fft.fft2(gray)
    fft_shifted = fft.fftshift(fft_result)
    magnitude = torch.abs(fft_shifted)
    return torch.log1p(magnitude)

def bihifilter(log_spectrum, freq_cutoff=0.1):
    B, H, W = log_spectrum.shape
    y = torch.linspace(-1, 1, H, device=log_spectrum.device)
    x = torch.linspace(-1, 1, W, device=log_spectrum.device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt(X**2 + Y**2)
    mask = torch.sigmoid((dist - freq_cutoff) * 20) 
    return log_spectrum * mask.unsqueeze(0)


# 3. Архитектура DualDomainCNN
class DualDomainCNN(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1), nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1), nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1), nn.BatchNorm2d(base_channels*8), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1), nn.BatchNorm2d(base_channels*8), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.BatchNorm2d(base_channels*2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1), nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1), nn.BatchNorm2d(base_channels*4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1), nn.BatchNorm2d(base_channels*8), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1), nn.BatchNorm2d(base_channels*8), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels*16, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*8, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 1))

    def forward(self, spatial_input, freq_input):
        spatial_feat = self.spatial_branch(spatial_input)
        freq_feat = self.freq_branch(freq_input)

        combined = torch.cat([spatial_feat, freq_feat], dim=1)

        logits = self.fusion(combined)
        return logits.squeeze(-1)


# 4. Инициализация API
app = FastAPI(title="AI Image Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели
model = DualDomainCNN(base_channels=32).to(DEVICE)
model_loaded = False

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        model_loaded = True
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f" Ошибка загрузки весов модели: {e}")
else:
    logger.warning(f"Файл весов не найден по пути: {MODEL_PATH}!")

transform_img = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 5. Эндпоинты
@app.get("/health")
async def health_check():
    return {
        "status": "ok" if model_loaded else "degraded",
        "device": str(DEVICE),
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена. Проверьте наличие файла весов")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением (JPEG, PNG и т.д.)")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        img_tensor = transform_img(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            log_spectrum = comp_log_spectrum(img_tensor)
            filtered_spectrum = bihifilter(log_spectrum)
            freq_tensor = filtered_spectrum.unsqueeze(1).to(DEVICE)
            
            logits = model(img_tensor, freq_tensor)
            prob = torch.sigmoid(logits).item()
            
        is_fake = prob >= 0.5
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": "FAKE" if is_fake else "REAL",
            "confidence": round(prob if is_fake else 1 - prob, 4),
            "is_generated": bool(is_fake)
        })
        
    except Exception as e:
        logger.error(f"Ошибка инференса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")
    
@app.post("/reload-model")
async def reload_model():
    global model, model_loaded
    try:
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            model_loaded = True
            logger.info("Модель перезагружена!")
            return {"status": "success", "message": "Model reloaded"}
        return {"status": "error", "message": "Weights file not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)