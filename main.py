

import random
import numpy as np
import torch
from pathlib import Path
import shutil
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# -------------------- НАСТРОЙКА ПУТЕЙ --------------------
WORKING_DIR = Path(r"C:\lab2")
TRAIN_IMAGES_DIR = WORKING_DIR / "yolo_dataset/yolo_dataset/train/images"
TRAIN_LABELS_DIR = WORKING_DIR / "yolo_dataset/yolo_dataset/train/labels"
TEST_DIR = WORKING_DIR / "test_images/test_images"

# Гиперпараметры обучения
EPOCHS = 30
PATIENCE = 5                     # ранняя остановка
MODEL_ARCHS = ["yolo26n.pt", "yolov8n.pt", "yolov9c.pt", "yolo11n.pt", "yolov8s.pt"]

# Папка для результатов
RUNS_DIR = WORKING_DIR / "runs_cv_ensemble_final"
folds_dir = WORKING_DIR / "folds_final"

# Очистка старых файлов (если нужно начать заново)
if RUNS_DIR.exists():
    shutil.rmtree(RUNS_DIR)
if folds_dir.exists():
    shutil.rmtree(folds_dir)
for f in WORKING_DIR.glob("data_fold_*.yaml"):
    f.unlink()

RUNS_DIR.mkdir(parents=True, exist_ok=True)
folds_dir.mkdir(exist_ok=True)

# -------------------- ПАРАМЕТРЫ ОБУЧЕНИЯ --------------------
TRAIN_PARAMS = {
    "imgsz": 800,                
    "batch": 16,
    "cos_lr": True,
    "lr0": 0.001,
    "lrf": 0.01,
    "optimizer": "AdamW",
    "weight_decay": 0.0005,
    "label_smoothing": 0.1,
    "freeze": 10,
    "mosaic": 1.0,
    "mixup": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "plots": False,
    "verbose": False,
    "patience": PATIENCE,
}

# -------------------- ПАРАМЕТРЫ ИНФЕРЕНСА И АНСАМБЛЯ --------------------
INFERENCE_IOU = 0.5
WBF_IOU_THR = 0.5
WBF_SKIP_BOX_THR = 0.0001
USE_TTA = False                   # TTA отключено (не все модели поддерживают)

# -------------------- ФУНКЦИЯ ДЛЯ СТРАТИФИКАЦИИ ПО НАЛИЧИЮ КЛАССА STAFF --------------------
def get_image_labels_info(image_paths, labels_dir):
    """Для каждого изображения определяем, есть ли объект класса staff (класс 1)."""
    staff_present = []
    for img_path in image_paths:
        label_file = labels_dir / (Path(img_path).stem + ".txt")
        has_staff = False
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) and int(float(parts[0])) == 1:
                        has_staff = True
                        break
        staff_present.append(1 if has_staff else 0)
    return np.array(staff_present)

print("Подготовка данных для кросс-валидации...")
image_paths = sorted(Path(TRAIN_IMAGES_DIR).glob("*.jpg"))
image_paths = [str(p.absolute()) for p in image_paths]
print(f"Всего изображений: {len(image_paths)}")
if len(image_paths) == 0:
    raise FileNotFoundError(f"Нет изображений в {TRAIN_IMAGES_DIR}")

# Получаем метки для стратификации
staff_labels = get_image_labels_info(image_paths, TRAIN_LABELS_DIR)
print(f"Изображений с классом staff: {np.sum(staff_labels)}")

# Стратифицированное разбиение на 5 фолдов
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_data = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_paths, staff_labels)):
    train_files = [image_paths[i] for i in train_idx]
    val_files = [image_paths[i] for i in val_idx]

    train_txt = folds_dir / f"train_fold_{fold_idx}.txt"
    val_txt = folds_dir / f"val_fold_{fold_idx}.txt"

    with open(train_txt, "w") as f:
        f.write("\n".join(train_files))
    with open(val_txt, "w") as f:
        f.write("\n".join(val_files))

    fold_data.append({
        "fold": fold_idx,
        "train_txt": str(train_txt),
        "val_txt": str(val_txt)
    })

yaml_template = """
path: ""
train: {train_txt}
val: {val_txt}
nc: 2
names: ['customer', 'staff']
"""

# -------------------- ФУНКЦИЯ ДЛЯ ПОДБОРА ОПТИМАЛЬНОГО ПОРОГА УВЕРЕННОСТИ --------------------
def find_optimal_conf(model, val_data_yaml, iou_thr=0.5, steps=20):
    """
    На валидационных данных подбирает порог уверенности, максимизирующий F1.
    Возвращает лучший порог.
    """
    best_f1 = 0
    best_conf = 0.2
    for conf in np.linspace(0.05, 0.95, steps):
        results = model.val(data=val_data_yaml, conf=conf, iou=iou_thr, plots=False, verbose=False)
        # Средние precision и recall по всем классам
        p = results.box.mp
        r = results.box.mr
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                best_conf = conf
    return best_conf

# -------------------- ОБУЧЕНИЕ С ЛОГИРОВАНИЕМ ПОЭПОХНОЙ ТОЧНОСТИ --------------------
trained_checkpoints = []          # список словарей с информацией о лучших моделях
epoch_metrics_all = []             # метрики по всем эпохам

print("Начинаем обучение 5 моделей на разных фолдах (до 30 эпох с ранней остановкой)...")

for fold in fold_data:
    fold_idx = fold["fold"]
    arch = MODEL_ARCHS[fold_idx % len(MODEL_ARCHS)]
    print(f"\n--- Фолд {fold_idx}: обучаем {arch} ---")

    # Создаём YAML для фолда
    fold_yaml = WORKING_DIR / f"data_fold_{fold_idx}.yaml"
    with open(fold_yaml, "w") as f:
        f.write(yaml_template.format(
            train_txt=fold["train_txt"],
            val_txt=fold["val_txt"]
        ))

    # Загружаем предобученную модель
    model = YOLO(arch)

    # Callback для сбора метрик после каждой эпохи
    epoch_records = []

    def create_epoch_callback(fold_idx, epoch_records):
        def on_fit_epoch_end(trainer):
            epoch = trainer.epoch
            metrics = trainer.metrics
            if not metrics and hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
                metrics = trainer.validator.metrics
            if not metrics:
                metrics = {}

            mAP50 = metrics.get('mAP50', metrics.get('metrics/mAP50(B)', 0))
            mAP50_95 = metrics.get('mAP50-95', metrics.get('metrics/mAP50-95(B)', 0))
            precision = metrics.get('precision', metrics.get('metrics/precision(B)', 0))
            recall = metrics.get('recall', metrics.get('metrics/recall(B)', 0))

            record = {
                "fold": fold_idx,
                "epoch": epoch,
                "mAP50": mAP50,
                "mAP50-95": mAP50_95,
                "precision": precision,
                "recall": recall,
            }
            epoch_records.append(record)
            print(f"   Эпоха {epoch:2d}: mAP50 = {record['mAP50']:.4f}, "
                  f"mAP50-95 = {record['mAP50-95']:.4f}, "
                  f"P = {record['precision']:.4f}, R = {record['recall']:.4f}")
        return on_fit_epoch_end

    model.add_callback("on_fit_epoch_end", create_epoch_callback(fold_idx, epoch_records))

    # Обучаем
    model.train(
        data=str(fold_yaml),
        epochs=EPOCHS,
        **TRAIN_PARAMS,
        project=str(RUNS_DIR),
        name=f"fold{fold_idx}_{arch.replace('.pt','')}"
    )

    # Путь к лучшим весам
    best_weight = RUNS_DIR / f"fold{fold_idx}_{arch.replace('.pt','')}/weights/best.pt"
    if not best_weight.exists():
        raise FileNotFoundError(f"Лучшие веса не найдены: {best_weight}")

    # Подбираем оптимальный порог для этой модели
    best_model = YOLO(best_weight)
    optimal_conf = find_optimal_conf(best_model, str(fold_yaml))
    print(f"   Оптимальный порог уверенности: {optimal_conf:.3f}")

    # Сохраняем информацию о модели
    trained_checkpoints.append({
        "weight_path": best_weight,
        "fold": fold_idx,
        "arch": arch,
        "optimal_conf": optimal_conf,
        "mAP50": best_model.val(data=str(fold_yaml)).box.map50  # можно сохранить для весов WBF
    })

    # Добавляем записи этой модели в общий список метрик
    epoch_metrics_all.extend(epoch_records)

print(f"Обучено {len(trained_checkpoints)} моделей.")

# -------------------- СОХРАНЕНИЕ ПОЭПОХНЫХ МЕТРИК В CSV --------------------
df_epoch_metrics = pd.DataFrame(epoch_metrics_all)
df_epoch_metrics.to_csv(RUNS_DIR / "epoch_metrics.csv", index=False)
print(f"Поэпошные метрики сохранены в {RUNS_DIR / 'epoch_metrics.csv'}")

# -------------------- ВЫЧИСЛЕНИЕ ВЕСОВ ДЛЯ WBF НА ОСНОВЕ mAP50 --------------------
weights = [ckpt["mAP50"] for ckpt in trained_checkpoints]
if all(w == 0 for w in weights):
    print("⚠️  Все веса равны нулю. Используем равные веса.")
    weights = [1.0] * len(weights)
else:
    weights = np.array(weights) / np.mean(weights)   # нормализация (среднее = 1)
print(f"Веса для WBF (пропорциональны mAP50 на валидации): {weights}")

# -------------------- ЗАГРУЗКА МОДЕЛЕЙ С ИХ ОПТИМАЛЬНЫМИ ПОРОГАМИ --------------------
models_info = []
for ckpt in trained_checkpoints:
    model = YOLO(ckpt["weight_path"])
    models_info.append({
        "model": model,
        "optimal_conf": ckpt["optimal_conf"]
    })
print("Модели загружены.")

# -------------------- ПОИСК ТЕСТОВЫХ ИЗОБРАЖЕНИЙ --------------------
TEST_IMAGES_DIR = Path(TEST_DIR)
if not TEST_IMAGES_DIR.exists():
    raise FileNotFoundError(f"Директория не найдена: {TEST_IMAGES_DIR}")

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
test_images = [f for f in TEST_IMAGES_DIR.iterdir() if f.suffix.lower() in image_extensions]
print(f"Найдено тестовых изображений: {len(test_images)}")

# -------------------- ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ С ОПТИМАЛЬНЫМИ ПОРОГАМИ --------------------
all_predictions = {}   # img_stem -> (list of (boxes, scores, labels), (width, height))

for img_path in tqdm(test_images, desc="Инференс"):
    with Image.open(img_path) as img:
        width, height = img.size

    preds = []
    for m in models_info:
        model = m["model"]
        conf_thr = m["optimal_conf"]
        result = model.predict(
            source=str(img_path),
            conf=conf_thr,
            iou=INFERENCE_IOU,
            augment=USE_TTA,
            verbose=False
        )[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            preds.append((boxes, scores, labels))

    all_predictions[img_path.stem] = (preds, (width, height))

print(f"Предсказания получены для {len(all_predictions)} изображений.")

# -------------------- АНСАМБЛЬ ЧЕРЕЗ WEIGHTED BOXES FUSION --------------------
ensemble_dir = RUNS_DIR / "predict_ensemble/labels"
ensemble_dir.mkdir(parents=True, exist_ok=True)

for img_name, (preds, (width, height)) in tqdm(all_predictions.items(), desc="WBF"):
    boxes_list = []
    scores_list = []
    labels_list = []

    for boxes, scores, labels in preds:
        # Нормализация координат к [0,1]
        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= width
        boxes_norm[:, [1, 3]] /= height
        boxes_norm = np.clip(boxes_norm, 0, 1)
        boxes_list.append(boxes_norm)
        scores_list.append(scores)
        labels_list.append(labels)

    if len(boxes_list) == 0:
        (ensemble_dir / f"{img_name}.txt").touch()
        continue

    # WBF с вычисленными весами
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights.tolist(),
        iou_thr=WBF_IOU_THR,
        skip_box_thr=WBF_SKIP_BOX_THR
    )

    # Сохраняем в формате YOLO: class xc yc w h score
    label_file = ensemble_dir / f"{img_name}.txt"
    with open(label_file, "w") as f:
        for b, s, l in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = b
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            xc = x1 + w / 2
            yc = y1 + h / 2
            f.write(f"{int(l)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {s:.6f}\n")

print("Ensemble завершён. Результаты сохранены в", ensemble_dir)

# -------------------- ФОРМИРОВАНИЕ САБМИШЕНА (только класс 1) --------------------
def build_submission_from_solution_order(
    solution_csv: str,
    preds_dir: str,
    output_csv: str = "submission.csv",
    image_col: str = "image_name",
    boxes_col: str = "boxes",
    row_id_col: str = "id",
    require_score: bool = True,
    clamp_score: bool = True,
    keep_only_class: int | None = 1,
) -> None:
    sol_path = Path(solution_csv)
    pdir = Path(preds_dir)
    if not sol_path.exists():
        raise FileNotFoundError(f"solution_csv not found: {sol_path}")
    if not pdir.exists() or not pdir.is_dir():
        raise FileNotFoundError(f"preds_dir not found or not a dir: {pdir}")

    sol = pd.read_csv(sol_path)
    if image_col not in sol.columns:
        raise ValueError(f"solution.csv must contain column '{image_col}'")

    image_names = sol[image_col].astype(str).tolist()
    rows = []

    for idx, image_name in enumerate(image_names):
        stem = Path(image_name).stem
        pred_file = pdir / f"{stem}.txt"
        boxes = []
        if pred_file.exists():
            content = pred_file.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                for ln in content.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    if require_score and len(parts) < 6:
                        continue
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(float(parts[0]))
                        if keep_only_class is not None and cls != keep_only_class:
                            continue
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        sc = float(parts[5]) if len(parts) >= 6 else 1.0
                    except ValueError:
                        continue
                    if clamp_score:
                        sc = 0.0 if sc < 0.0 else (1.0 if sc > 1.0 else sc)
                    boxes.append([xc, yc, w, h, sc])

        rows.append({
            row_id_col: idx,
            image_col: image_name,
            boxes_col: json.dumps(boxes, separators=(",", ":"))
        })

    sub = pd.DataFrame(rows, columns=[row_id_col, image_col, boxes_col])
    sub.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv} ({len(sub)} rows).")

build_submission_from_solution_order(
    solution_csv=r"C:\lab2\sample_sub.csv",
    preds_dir=str(ensemble_dir),
    output_csv=r"C:\lab2\submission_final.csv",
    keep_only_class=1
)

print("Готово! Файл submission_final.csv создан.")
print(f"Поэпошные метрики сохранены в {RUNS_DIR / 'epoch_metrics.csv'}")
