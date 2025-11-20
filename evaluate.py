import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.data_utils import get_emnist_dataloaders, get_num_classes
from src.model import SimpleCNN


def get_class_names(split: str) -> list:
    if split == "letters":
        return [chr(ord("A") + i) for i in range(26)]
    
    return [str(i) for i in range(get_num_classes(split))]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained EMNIST CNN model."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved .pth checkpoint.")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory where EMNIST is stored.")
    parser.add_argument("--split", type=str, default="letters",
                        help="EMNIST split used during training.")
    parser.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = checkpoint.get("num_classes", get_num_classes(args.split))

    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = get_emnist_dataloaders(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        val_ratio=0.1,
        num_workers=2,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    class_names = get_class_names(args.split)

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix (EMNIST)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    fig.savefig("outputs/figures/confusion_matrix.png")
    print("Saved confusion matrix to outputs/figures/confusion_matrix.png")


if __name__ == "__main__":
    main()