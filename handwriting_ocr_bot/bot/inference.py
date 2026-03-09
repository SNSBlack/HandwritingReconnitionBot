
import os
import sys
import importlib.util
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


def _import_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_BOT_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_BOT_DIR)
_MODEL_DIR   = os.path.join(_PROJECT_DIR, "model")

_dataset = _import_from_file("dataset", os.path.join(_MODEL_DIR, "dataset.py"))
_crnn    = _import_from_file("crnn",    os.path.join(_MODEL_DIR, "crnn.py"))

CRNN       = _crnn.CRNN
NUM_CLASSES = _dataset.NUM_CLASSES
decode_ctc  = _dataset.decode_ctc


def _load_crnn_model(checkpoint_path: str, device: torch.device):
    model = CRNN()
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _preprocess_image(image: Image.Image, img_height=64, img_width=512) -> torch.Tensor:
    """Предобрабатывает изображение для CRNN."""
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((img_height, img_width)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    return transform(image).unsqueeze(0)


def _crnn_predict(model, image_tensor: torch.Tensor, device: torch.device) -> str:
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        log_probs = model(image_tensor)
    preds = log_probs.argmax(dim=2).squeeze(1)
    return decode_ctc(preds.cpu().tolist())


def _load_easyocr(use_gpu: bool):
    import easyocr
    #загрузка модели для русского + английского языка
    reader = easyocr.Reader(
        ["ru", "en"],
        gpu=use_gpu,
        verbose=False,
    )
    return reader


def _easyocr_predict(reader, image: Image.Image) -> str:
    """Распознаёт текст через EasyOCR."""
    img_np = np.array(image)
    results = reader.readtext(img_np, detail=0, paragraph=True)
    return " ".join(results).strip()

def _load_trocr(model_path: str, device: torch.device):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return processor, model


def _trocr_predict(processor, model, image: Image.Image, device: torch.device) -> str:
    pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated = model.generate(pixel_values, max_new_tokens=64, num_beams=4)
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

class OCREngine:
    """
    Единый интерфейс для OCR.

    Args:
        mode: "trocr" | "easyocr" | "crnn"
        checkpoint_path: путь к модели
        device_str: "cuda", "cpu", ...
    """

    def __init__(
        self,
        mode: str = "trocr",
        checkpoint_path: str | None = None,
        device_str: str = "cuda",
    ):
        self.mode = mode

        if device_str.startswith("cuda") and not torch.cuda.is_available():
            print("⚠️  CUDA недоступна, использую CPU")
            device_str = "cpu"
        self.device = torch.device(device_str)
        use_gpu = self.device.type == "cuda"

        print(f"🖥️  OCR устройство: {self.device}")

        if mode == "trocr":
            if checkpoint_path and os.path.exists(checkpoint_path):
                model_path = checkpoint_path
                print(f"🧠 Загружаю TrOCR (дообученная): {model_path}")
            else:
                model_path = "microsoft/trocr-base-handwritten"
                print(f"🧠 Загружаю TrOCR (базовая от Microsoft)")
            self._processor, self._model = _load_trocr(model_path, self.device)
            self._reader = None

        elif mode == "crnn":
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")
            print(f"🧠 Загружаю CRNN модель: {checkpoint_path}")
            self._model = _load_crnn_model(checkpoint_path, self.device)
            self._processor = None
            self._reader = None

        elif mode == "easyocr":
            print("🧠 Загружаю EasyOCR (ru + en)...")
            self._model = None
            self._processor = None
            self._reader = _load_easyocr(use_gpu)

        else:
            raise ValueError(f"Неизвестный режим: {mode}. Используйте 'trocr', 'crnn' или 'easyocr'")

        print("✅ Модель загружена\n")

    def recognize(self, image: Image.Image) -> str:
        if self.mode == "trocr":
            text = _trocr_predict(self._processor, self._model, image, self.device)
        elif self.mode == "crnn":
            tensor = _preprocess_image(image)
            text = _crnn_predict(self._model, tensor, self.device)
        else:
            text = _easyocr_predict(self._reader, image)

        return text if text.strip() else "(текст не распознан)"

    def recognize_from_path(self, path: str) -> str:
        """Удобный метод — принимает путь к файлу."""
        image = Image.open(path).convert("RGB")
        return self.recognize(image)

    def recognize_from_bytes(self, data: bytes) -> str:
        """Удобный метод — принимает байты изображения."""
        import io
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return self.recognize(image)