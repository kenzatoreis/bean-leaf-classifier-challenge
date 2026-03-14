import os
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt

SEED = 42
TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
TEST_DIR = "data/test"
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
PATIENCE = 2
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_confusion(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def evaluate(model, loader, criterion):
    model.eval()
    losses, preds, labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return np.mean(losses), acc, f1, labels, preds

def main():
    set_seed(SEED)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tfms)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfms)

    class_names = train_ds.classes
    print("Classes:", class_names)
    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
    print("Test size:", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_f1 = -1.0
    best_epoch = -1
    best_path = "checkpoints/best_efficientnet_b0.pt"
    no_improve = 0
    log_lines = []

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)

        line = (
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )
        print(line)
        log_lines.append(line)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, class_names, "results/confusion_matrix.png")

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        "seed": SEED,
        "model": "efficientnet_b0",
        "best_epoch": best_epoch,
        "best_val_f1": float(best_f1),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "class_names": class_names,
        "runtime_seconds": float(time.time() - start),
        "classification_report": report,
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open("results/val_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    print("\nFinal test metrics:")
    print(json.dumps({
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "checkpoint": best_path
    }, indent=2))

if __name__ == "__main__":
    main()