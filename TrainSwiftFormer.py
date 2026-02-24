import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import SwiftFormerForImageClassification, AutoImageProcessor


def main():
    # ─── Device Setup ───────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ─── Model & Processor ──────────────────────────────────────────────
    model_name = "MBZUAI/swiftformer-s"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SwiftFormerForImageClassification.from_pretrained(
        model_name,
        num_labels=14,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,          # MPS works best with float32 weights
        attn_implementation="eager",        # Disable SDPA — not fully supported on MPS
    )
    model.to(device)

    # ─── IMPORTANT: Do NOT use torch.compile on MPS ─────────────────────
    # torch.compile uses the inductor backend which barely supports MPS,
    # causes huge cold-start overhead, and often falls back to eager mode anyway.

    # ─── Data Augmentation & Transforms ─────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        transforms.RandomErasing(p=0.1),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # ─── Dataset ────────────────────────────────────────────────────────
    data_dir = "_dataset_"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    print(f"Classes: {train_dataset.classes}")

    # ─── DataLoader (tuned for macOS + MPS) ─────────────────────────────
    # On macOS, num_workers > 0 uses fork() which can cause issues with MPS.
    # Using persistent_workers + prefetch keeps the pipeline fed.
    # Batch size 64 fits comfortably in 18GB unified memory for SwiftFormer-S.
    BATCH_SIZE = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # ─── Training Hyperparameters ───────────────────────────────────────
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 2
    LABEL_SMOOTHING = 0.1

    # AdamW with weight decay (standard for vision transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # Cosine annealing LR schedule with linear warmup
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Label-smoothed cross-entropy (better generalization for small datasets)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ─── Training Loop ──────────────────────────────────────────────────
    best_val_accuracy = 0.0
    save_path = "best_swiftformer_gdtFCF.pth"

    print(f"\n{'='*60}")
    print(f"Training Config:")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Epochs:           {NUM_EPOCHS}")
    print(f"  LR:               {LEARNING_RATE}")
    print(f"  Weight decay:     {WEIGHT_DECAY}")
    print(f"  Label smoothing:  {LABEL_SMOOTHING}")
    print(f"  Warmup epochs:    {WARMUP_EPOCHS}")
    print(f"  Steps/epoch:      {len(train_loader)}")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if (i + 1) % 20 == 0:
                avg_loss = running_loss / (i + 1)
                acc = 100 * train_correct / train_total
                lr_now = scheduler.get_last_lr()[0]
                print(f"  [{epoch+1}/{NUM_EPOCHS}] Step {i+1:3d}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | LR: {lr_now:.2e}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs.logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        # ── Epoch Summary ──
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.1f}%")
        print(f"  Val   Loss: {avg_val_loss:.4f}  |  Val   Acc: {val_accuracy:.1f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best! Saved to {save_path}")
        print(f"{'─'*60}\n")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()