# ============================================================================
# File: p2_inference_confident.py
# Description: Load pre-trained model and perform inference on test set.
#              Only records predictions with confidence >= 50%.
#              Outputs JSON: {"filenames": [...], "labels": [...]}
# ============================================================================
import os
import sys
import time
import json
import argparse
import torch
import torch.nn.functional as F

from model import MyNet, ResNet18
from dataset import get_dataloader

CONFIDENCE_THRESHOLD = 0.2


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_datadir",
        help="test dataset directory",
        type=str,
        default="../hw2_data/p2_data/",
    )
    parser.add_argument(
        "--model_type", help="mynet or resnet18", type=str, default="resnet18"
    )
    parser.add_argument(
        "--output_path",
        help="output json file path",
        type=str,
        default="./output/annotations.json",
    )
    args = parser.parse_args()

    model_type = args.model_type
    test_datadir = args.test_datadir
    output_path = args.output_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if model_type == "mynet":
        model = MyNet()
        model.load_state_dict(
            torch.load("./checkpoint/mynet_best.pth", map_location=torch.device("cpu"))
        )
    elif model_type == "resnet18":
        model = ResNet18()
        model.load_state_dict(
            torch.load(
                "./checkpoint/resnet18_best.pth", map_location=torch.device("cpu")
            )
        )
    else:
        raise NameError("Unknown model type")
    model.to(device)

    test_loader = get_dataloader(test_datadir, batch_size=1, split="test")
    image_names = test_loader.dataset.image_names

    filenames = []
    labels = []

    model.eval()
    with torch.no_grad():
        test_start_time = time.time()
        for batch, data in enumerate(test_loader):
            sys.stdout.write(f"\r Test batch: {batch + 1} / {len(test_loader)}")
            sys.stdout.flush()

            images = data["images"].to(device)
            # Forward pass. output: (batch_size, 10)
            logits = model(images)
            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            # Get max confidence and predicted label
            confidence, pred_label = torch.max(probs, dim=1)

            conf_val = confidence.item()
            if conf_val >= CONFIDENCE_THRESHOLD:
                filenames.append(image_names[batch])
                labels.append(pred_label.item())

    test_time = time.time() - test_start_time
    print()
    print(
        f"Finish testing {test_time:.2f} sec(s). "
        f"Kept {len(filenames)} / {len(image_names)} predictions (confidence >= {CONFIDENCE_THRESHOLD * 100:.0f}%)."
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {"filenames": filenames, "labels": labels}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result saved to {output_path}")

    # Per-class distribution of kept predictions
    print("\nPer-class prediction distribution:")
    total = len(labels)
    if total == 0:
        print("  (no predictions kept)")
    else:
        counts = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        for cls in sorted(counts.keys()):
            n = counts[cls]
            print(f"  class {cls}: {n} / {total} ({n / total * 100:.2f}%)")


if __name__ == "__main__":
    main()
