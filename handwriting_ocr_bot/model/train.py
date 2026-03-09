"""
train.py — обучение CRNN модели на рукописном русском датасете
Использует CTC Loss, поддерживает GPU.
"""

# Пути к датасету по умолчанию
ARCHIVE_DIR = r"C:\Users\Kirill\PycharmProjects\PythonProject\archive"

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import HandwritingDataset, collate_fn, decode_ctc, NUM_CLASSES, ALPHABET
from crnn import CRNN


# ──────────────────────────────────────────────────────────────────────────────
# CPU ограничение
# ──────────────────────────────────────────────────────────────────────────────

def set_cpu_base_clock():
    """Фиксирует CPU на базовой частоте через powercfg. Требует прав администратора."""
    import platform, subprocess
    if platform.system() != "Windows":
        print("   ⚠️  Ограничение частоты CPU поддерживается только на Windows")
        return
    try:
        result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
        guid = result.stdout.strip().split()[3]
        subprocess.run([
            "powercfg", "/setacvalueindex", guid,
            "54533251-82be-4824-96c1-47b60b740d00",
            "bc5038f7-23e0-4960-96da-33abaf5935ec", "99"
        ], check=True, capture_output=True)
        subprocess.run(["powercfg", "/setactive", guid], check=True, capture_output=True)
        print("   🔒 CPU зафиксирован на базовой частоте (турбобуст отключён)")
        print(f"   ℹ️  Для восстановления вручную:")
        print(f"      powercfg /setacvalueindex {guid} 54533251-82be-4824-96c1-47b60b740d00 bc5038f7-23e0-4960-96da-33abaf5935ec 100")
        print(f"      powercfg /setactive {guid}")
    except subprocess.CalledProcessError:
        print("   ⚠️  Не удалось ограничить частоту CPU — запусти от имени администратора")
    except Exception as e:
        print(f"   ⚠️  Ошибка при ограничении CPU: {e}")


def restore_cpu_clock():
    """Возвращает максимальную частоту CPU на 100% после обучения."""
    import platform, subprocess
    if platform.system() != "Windows":
        return
    try:
        result = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True)
        guid = result.stdout.strip().split()[3]
        subprocess.run([
            "powercfg", "/setacvalueindex", guid,
            "54533251-82be-4824-96c1-47b60b740d00",
            "bc5038f7-23e0-4960-96da-33abaf5935ec", "100"
        ], check=True, capture_output=True)
        subprocess.run(["powercfg", "/setactive", guid], check=True, capture_output=True)
        print("✅ Частота CPU восстановлена до 100%")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Метрика
# ──────────────────────────────────────────────────────────────────────────────

def cer(pred: str, target: str) -> float:
    """Character Error Rate."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dp = list(range(len(target) + 1))
    for i, pc in enumerate(pred):
        new_dp = [i + 1]
        for j, tc in enumerate(target):
            new_dp.append(min(dp[j] + (0 if pc == tc else 1), new_dp[-1] + 1, dp[j + 1] + 1))
        dp = new_dp
    return dp[-1] / len(target)


# ──────────────────────────────────────────────────────────────────────────────
# Обучение / оценка
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None
    pbar = tqdm(loader, desc="  Обучение", unit="batch", leave=False)

    for images, labels, label_lengths, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            log_probs = model(images)
            T, batch_size, _ = log_probs.shape
            input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=device)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

        if torch.isfinite(loss):
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        pbar.set_postfix(loss=f"{total_loss / max(n_batches, 1):.4f}", refresh=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, show_samples=10):
    model.eval()
    total_cer = 0.0
    n_samples = 0
    samples_shown = 0
    pbar = tqdm(loader, desc="  Оценка  ", unit="batch", leave=False)

    for images, _, label_lengths, texts in pbar:
        images = images.to(device, non_blocking=True)
        log_probs = model(images)
        preds = log_probs.argmax(dim=2).permute(1, 0)

        for pred_seq, true_text in zip(preds.cpu().tolist(), texts):
            pred_text = decode_ctc(pred_seq)
            total_cer += cer(pred_text, true_text)
            n_samples += 1
            if samples_shown < show_samples:
                tqdm.write(f"   📌 Ожидалось: '{true_text}'  →  Предсказано: '{pred_text}'")
                samples_shown += 1

        pbar.set_postfix(CER=f"{total_cer / max(n_samples, 1):.4f}", refresh=True)

    return total_cer / max(n_samples, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Обучение CRNN для OCR")
    parser.add_argument("--train_tsv",      default=os.path.join(ARCHIVE_DIR, "train.tsv"))
    parser.add_argument("--test_tsv",       default=os.path.join(ARCHIVE_DIR, "test.tsv"))
    parser.add_argument("--data_dir",       default=os.path.join(ARCHIVE_DIR, "train"))
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--img_height",     type=int,   default=64)
    parser.add_argument("--img_width",      type=int,   default=256)
    parser.add_argument("--lstm_hidden",    type=int,   default=256)
    parser.add_argument("--workers",        type=int,   default=0)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--checkpoint_dir", default=r"D:\data sets\cheks")
    parser.add_argument("--resume",         default=None)
    parser.add_argument("--limit-cpu",      action="store_true", default=None,
                        help="Зафиксировать CPU на базовой частоте (без турбобуста)")
    args = parser.parse_args()

    # ── Вопрос про CPU ──────────────────────────────────────────────────────
    if args.limit_cpu is None:
        print("\n🔒 Ограничить CPU на базовой частоте 3.67 GHz (без турбобуста)?")
        print("   Это снизит нагрев VRM при долгом обучении")
        print("   [1] Да — зафиксировать на 3.67 GHz")
        print("   [2] Нет — работать на полной мощности")
        while True:
            choice = input(">>> ").strip()
            if choice == "1":
                args.limit_cpu = True
                break
            elif choice == "2":
                args.limit_cpu = False
                break
            else:
                print("   Введите 1 или 2")

    # ── Устройство ──────────────────────────────────────────────────────────
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA недоступна, переключаюсь на CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n🖥️  Устройство: {device}")

    # ── CPU ограничение ─────────────────────────────────────────────────────
    if args.limit_cpu:
        print("🔒 Настройка CPU...")
        set_cpu_base_clock()
    else:
        print("⚡ CPU работает на полной мощности (турбобуст включён)")

    # ── GPU настройки ───────────────────────────────────────────────────────
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(device)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml
            pynvml.nvmlInit()
            gpu_index = device.index if device.index is not None else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, 200 * 1000)
            actual = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            print(f"   ⚡ Лимит мощности GPU: {actual // 1000} Вт")
            pynvml.nvmlShutdown()
        except ImportError:
            print("   ⚠️  pynvml не установлен — pip install nvidia-ml-py")
        except Exception as e:
            print(f"   ⚠️  Лимит GPU не задан: {e} (запусти от имени администратора)")

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Проверка путей ──────────────────────────────────────────────────────
    print(f"\n📁 train.tsv : {args.train_tsv}")
    print(f"📁 test.tsv  : {args.test_tsv}")
    print(f"📁 data_dir  : {args.data_dir}")
    for p in [args.train_tsv, args.test_tsv, args.data_dir]:
        if not os.path.exists(p):
            print(f"❌ Путь не найден: {p}")
            sys.exit(1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Датасеты ────────────────────────────────────────────────────────────
    print("\n📂 Загрузка датасета...")
    train_ds = HandwritingDataset(
        args.train_tsv, args.data_dir,
        img_height=args.img_height, img_width=args.img_width, augment=True
    )
    test_ds = HandwritingDataset(
        args.test_tsv, args.data_dir,
        img_height=args.img_height, img_width=args.img_width, augment=False
    )

    found = sum(1 for fname, _ in train_ds.samples
                if os.path.exists(os.path.join(args.data_dir, fname)))
    missing = len(train_ds.samples) - found
    print(f"   Найдено изображений на диске: {found} / {len(train_ds.samples)}")
    if missing > 0:
        print(f"   ⚠️  Не найдено: {missing} файлов")
        shown = 0
        for fname, _ in train_ds.samples:
            if not os.path.exists(os.path.join(args.data_dir, fname)):
                print(f"   Пример ненайденного: {fname}")
                shown += 1
                if shown >= 3:
                    break
        if missing == len(train_ds.samples):
            print("\n❌ Ни одно изображение не найдено. Обучение остановлено.")
            sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn, drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(args.workers > 0),
    )

    print(f"   Train: {len(train_ds)} образцов, {len(train_loader)} батчей")
    print(f"   Test:  {len(test_ds)} образцов")

    # ── Модель ──────────────────────────────────────────────────────────────
    model = CRNN(img_height=args.img_height, lstm_hidden=args.lstm_hidden).to(device)
    print(f"\n🧠 Параметров модели: {sum(p.numel() for p in model.parameters()):,}")

    # ── AMP ─────────────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    print(f"⚡ AMP (mixed precision): {'включён' if scaler else 'выключен'}")

    # ── Оптимизатор ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    warmup_epochs = 3
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda ep: (ep + 1) / warmup_epochs
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # ── Чекпоинт ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_cer = float("inf")

    if not args.resume:
        all_checkpoints = []
        if os.path.exists(args.checkpoint_dir):
            all_checkpoints = sorted([
                f for f in os.listdir(args.checkpoint_dir) if f.endswith(".pth")
            ])

        if all_checkpoints:
            print(f"\n📦 Найдены чекпоинты в папке '{args.checkpoint_dir}':")
            for i, name in enumerate(all_checkpoints):
                fsize = os.path.getsize(os.path.join(args.checkpoint_dir, name)) / 1e6
                print(f"   [{i+1}] {name}  ({fsize:.1f} MB)")

            print("\nЗагрузить чекпоинт и продолжить обучение?")
            print("  Введите номер чекпоинта — загрузить")
            print("  Введите 0 или Enter — начать с нуля")

            while True:
                choice = input(">>> ").strip()
                if choice == "" or choice == "0":
                    print("🆕 Начинаю обучение с нуля\n")
                    break
                elif choice.isdigit() and 1 <= int(choice) <= len(all_checkpoints):
                    args.resume = os.path.join(args.checkpoint_dir, all_checkpoints[int(choice) - 1])
                    break
                else:
                    print(f"   Введите число от 0 до {len(all_checkpoints)}")

    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Загружаю чекпоинт: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_cer = ckpt.get("best_cer", float("inf"))
        print(f"   Продолжаю с эпохи {start_epoch}, лучший CER: {best_cer:.4f}\n")

    # ── Обучение ────────────────────────────────────────────────────────────
    print("\n🚀 Начинаю обучение...\n")
    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Эпохи", unit="ep")

    for epoch in epoch_pbar:
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        val_cer = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        epoch_pbar.set_postfix(loss=f"{train_loss:.4f}", CER=f"{val_cer:.4f}",
                               lr=f"{lr:.2e}", time=f"{elapsed:.0f}s")
        tqdm.write(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | CER: {val_cer:.4f} | "
            f"LR: {lr:.2e} | ⏱ {elapsed:.0f}s"
        )

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_cer)

        epoch_ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_cer": best_cer,
        }, epoch_ckpt_path)
        tqdm.write(f"   💾 Сохранён: {epoch_ckpt_path}")

        if val_cer < best_cer:
            best_cer = val_cer
            best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_cer": best_cer,
                "alphabet": ALPHABET,
            }, best_path)
            tqdm.write(f"   ✅ Лучший CER {best_cer:.4f} → сохранён {best_path}")

    print(f"\n🎉 Обучение завершено. Лучший CER: {best_cer:.4f}")
    if args.limit_cpu:
        restore_cpu_clock()


if __name__ == "__main__":
    main()