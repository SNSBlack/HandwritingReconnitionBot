"""
Мой код для добучения следует аккуратно запускать, т.к. я его писал с учётом своих тех осбенностей железа.
Для быстрой работы кода надо 128+ gb оперативной памяти
"""

ARCHIVE_DIR = r"C:\Users\Kirill\PycharmProjects\PythonProject\archive"

import os
import sys
import time
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

def set_cpu_base_clock():
    import platform, subprocess, ctypes
    if platform.system() != "Windows":
        return
    try:
        result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
        guid = result.stdout.strip().split()[3]
        subprocess.run(["powercfg", "/setacvalueindex", guid,
            "54533251-82be-4824-96c1-47b60b740d00",
            "bc5038f7-23e0-4960-96da-33abaf5935ec", "97"],
            check=True, capture_output=True)
        subprocess.run(["powercfg", "/setactive", guid], check=True, capture_output=True)
        print("   🔒 [powercfg] Турбобуст отключён")
    except Exception:
        print("   ⚠️  powercfg недоступен (нет прав администратора)")
    try:
        BELOW_NORMAL = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL)
        print("   🔒 [priority] Приоритет процесса снижен")
    except Exception:
        pass


def restore_cpu_clock():
    import platform, subprocess, ctypes
    if platform.system() != "Windows":
        return
    try:
        result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
        guid = result.stdout.strip().split()[3]
        subprocess.run(["powercfg", "/setacvalueindex", guid,
            "54533251-82be-4824-96c1-47b60b740d00",
            "bc5038f7-23e0-4960-96da-33abaf5935ec", "100"],
            check=True, capture_output=True)
        subprocess.run(["powercfg", "/setactive", guid], check=True, capture_output=True)
        NORMAL = 0x00000020
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, NORMAL)
        print("✅ Настройки CPU восстановлены")
    except Exception:
        pass


class HandwritingDatasetTrOCR(Dataset):
    def __init__(self, tsv_path: str, data_dir: str, processor, augment: bool = False):
        self.data_dir = data_dir
        self.processor = processor
        self.augment = augment
        self.samples = self._load_tsv(tsv_path)
        self.pixel_cache = None
        self.label_cache = None

    def _load_tsv(self, tsv_path):
        samples = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    filename, text = parts
                    img_path = os.path.join(self.data_dir, filename)
                    if os.path.exists(img_path) and text.strip():
                        samples.append((img_path, text.strip()))
        return samples

    def preload(self):
        """Загружает все изображения и токены в RAM заранее (один раз)."""
        print(f"   🗂️  Предзагрузка {len(self.samples)} изображений в RAM...")
        pixels = []
        labels = []
        for img_path, text in tqdm(self.samples, desc="   Загрузка", unit="img", leave=False):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (384, 64), color=255)
            pv = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
            pixels.append(pv)

            lb = self.processor.tokenizer(
                text, return_tensors="pt",
                padding="max_length", max_length=32, truncation=True,
            ).input_ids.squeeze(0)
            lb[lb == self.processor.tokenizer.pad_token_id] = -100
            labels.append(lb)

        self.pixel_cache = pixels
        self.label_cache = labels
        print(f"   ✅ Загружено в RAM")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.pixel_cache is not None:
            return {
                "pixel_values": self.pixel_cache[idx],
                "labels": self.label_cache[idx],
                "text": self.samples[idx][1],
            }
        img_path, text = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (384, 64), color=255)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text, return_tensors="pt",
            padding="max_length", max_length=32, truncation=True,
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels, "text": text}


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    texts = [b["text"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels, "texts": texts}


def cer(pred: str, target: str) -> float:
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dp = list(range(len(target) + 1))
    for i, pc in enumerate(pred):
        new_dp = [i + 1]
        for j, tc in enumerate(target):
            new_dp.append(min(dp[j] + (0 if pc == tc else 1), new_dp[-1] + 1, dp[j + 1] + 1))
        dp = new_dp
    return dp[-1] / len(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_tsv",      default=os.path.join(ARCHIVE_DIR, "train.tsv"))
    parser.add_argument("--test_tsv",       default=os.path.join(ARCHIVE_DIR, "test.tsv"))
    parser.add_argument("--data_dir",       default=os.path.join(ARCHIVE_DIR, "train"))
    parser.add_argument("--test_data_dir",  default=os.path.join(ARCHIVE_DIR, "test"))
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=5e-5)
    parser.add_argument("--workers",        type=int,   default=4)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--checkpoint_dir", default=r"D:\data sets\cheks_trocr")
    parser.add_argument("--resume",         default=None)
    parser.add_argument("--limit-cpu",      action="store_true", default=None)
    args = parser.parse_args()

    if args.limit_cpu is None:
        print("\n🔒 Ограничить CPU на базовой частоте 3.67 GHz?")
        print("   [1] Да   [2] Нет")
        while True:
            choice = input(">>> ").strip()
            if choice == "1":
                args.limit_cpu = True
                break
            elif choice == "2":
                args.limit_cpu = False
                break

    if args.limit_cpu:
        print("🔒 Настройка CPU...")
        set_cpu_base_clock()
    else:
        print("⚡ CPU работает на полной мощности")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA недоступна, использую CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n🖥️  Устройство: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(device)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, 190 * 1000)
            actual = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            print(f"   ⚡ Лимит GPU: {actual // 1000} Вт")
            pynvml.nvmlShutdown()
        except Exception:
            pass

    print("\n🧠 Загрузка TrOCR...")
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    MODEL_NAME = "microsoft/trocr-base-handwritten"

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    local_model = os.path.join(args.checkpoint_dir, "trocr_base")

    if os.path.exists(local_model):
        print(f"   Загружаю из локального кэша: {local_model}")
        processor = TrOCRProcessor.from_pretrained(local_model)
        model = VisionEncoderDecoderModel.from_pretrained(local_model)
    else:
        print(f"   Скачиваю {MODEL_NAME} (первый раз ~1.5 GB)...")
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        processor.save_pretrained(local_model)
        model.save_pretrained(local_model)
        print(f"   Сохранено в {local_model}")

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Параметров: {total_params:,}")

    print("\n📂 Загрузка датасета...")
    train_ds = HandwritingDatasetTrOCR(args.train_tsv, args.data_dir, processor, augment=True)
    test_ds  = HandwritingDatasetTrOCR(args.test_tsv,  args.test_data_dir, processor, augment=False)

    print(f"   Train: {len(train_ds)} образцов")
    print(f"   Test:  {len(test_ds)} образцов")

    if len(train_ds) == 0:
        print("❌ Train датасет пустой — проверь пути к файлам")
        sys.exit(1)

    if len(test_ds) == 0:
        print("⚠️  Test датасет пустой — оценка будет пропущена")

    train_ds.preload()
    if len(test_ds) > 0:
        test_ds.preload()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda")) if len(test_ds) > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 0
    best_cer = float("inf")

    all_checkpoints = sorted([
        f for f in os.listdir(args.checkpoint_dir)
        if f.startswith("epoch_") and f.endswith(".pth")
    ])

    if all_checkpoints:
        print(f"\n📦 Найдены чекпоинты:")
        for i, name in enumerate(all_checkpoints):
            fsize = os.path.getsize(os.path.join(args.checkpoint_dir, name)) / 1e6
            print(f"   [{i+1}] {name}  ({fsize:.1f} MB)")
        print("   Введите номер или 0/Enter для начала с нуля")
        while True:
            choice = input(">>> ").strip()
            if choice == "" or choice == "0":
                print("🆕 Начинаю с нуля\n")
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(all_checkpoints):
                ckpt_path = os.path.join(args.checkpoint_dir, all_checkpoints[int(choice)-1])
                ckpt = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_cer = ckpt.get("best_cer", float("inf"))
                print(f"🔄 Продолжаю с эпохи {start_epoch}, CER: {best_cer:.4f}\n")
                break

    print("\n🚀 Начинаю обучение TrOCR...\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{args.epochs}", unit="batch", leave=False)

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            if torch.isfinite(loss):
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_batches += 1

            pbar.set_postfix(loss=f"{total_loss / max(n_batches, 1):.4f}")

        avg_loss = total_loss / max(n_batches, 1)

        val_cer = float("inf")
        if test_loader is not None:
            model.eval()
            total_cer = 0.0
            n_samples = 0
            samples_shown = 0

            eval_pbar = tqdm(test_loader, desc="  Оценка", unit="batch", leave=False)
            with torch.no_grad():
                for batch in eval_pbar:
                    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                    texts = batch["texts"]

                    generated = model.generate(
                        pixel_values,
                        max_new_tokens=32,
                        num_beams=1,
                    )
                    pred_texts = processor.batch_decode(generated, skip_special_tokens=True)

                    for pred, true in zip(pred_texts, texts):
                        total_cer += cer(pred.strip(), true.strip())
                        n_samples += 1
                        if samples_shown < 5:
                            tqdm.write(f"   📌 Ожидалось: '{true}'  →  Предсказано: '{pred.strip()}'")
                            samples_shown += 1

                    eval_pbar.set_postfix(CER=f"{total_cer / max(n_samples, 1):.4f}")

            val_cer = total_cer / max(n_samples, 1)
        else:
            tqdm.write("   ⚠️  Оценка пропущена (test датасет пустой)")
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | CER: {val_cer:.4f} | LR: {lr:.2e} | ⏱ {elapsed:.0f}s")

        epoch_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_cer": best_cer,
        }, epoch_path)
        print(f"   💾 Сохранён: {epoch_path}")

        if val_cer < best_cer:
            best_cer = val_cer
            best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_cer": best_cer,
            }, best_path)
            model.save_pretrained(os.path.join(args.checkpoint_dir, "best_hf"))
            processor.save_pretrained(os.path.join(args.checkpoint_dir, "best_hf"))
            print(f"   ✅ Лучший CER {best_cer:.4f} → сохранён {best_path}")

    print(f"\n🎉 Обучение завершено. Лучший CER: {best_cer:.4f}")
    if args.limit_cpu:
        restore_cpu_clock()


if __name__ == "__main__":
    main()