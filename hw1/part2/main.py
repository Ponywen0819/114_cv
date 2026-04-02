import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(
        description="main function of joint bilateral filter"
    )
    parser.add_argument(
        "--image_path", default="part2/testdata/1.png", help="path to input image"
    )
    parser.add_argument(
        "--setting_path",
        default="part2/testdata/2_setting.txt",
        help="path to setting file",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Parse setting file
    with open(args.setting_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Last line: sigma_s,<val>,sigma_r,<val>
    sigma_line = lines[-1].split(",")
    sigma_s = int(sigma_line[1])
    sigma_r = float(sigma_line[3])

    # Middle lines: R,G,B weights (skip header "R,G,B")
    weight_list = []
    for line in lines[1:-1]:
        r, g, b = map(float, line.split(","))
        weight_list.append((r, g, b))

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)

    # Bilateral filter (RGB as guidance) as reference
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.int32)

    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.int32)
    error = np.sum(np.abs(jbf_out - bf_out))
    print(f"R={r}, G={g}, B={b} => cost: {error}")

    # For each grayscale weight, run JBF and compute error vs bf_out
    results = []
    for r, g, b in weight_list:
        gray_guidance = (
            r * img_rgb[:, :, 0] + g * img_rgb[:, :, 1] + b * img_rgb[:, :, 2]
        ).astype(np.uint8)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, gray_guidance).astype(np.int32)
        error = np.sum(np.abs(jbf_out - bf_out))
        results.append((error, r, g, b))
        print(f"R={r}, G={g}, B={b} => cost: {error}")

    best = min(results)
    worst = max(results)
    print(f"\nBest:  R={best[1]}, G={best[2]}, B={best[3]} => cost: {best[0]}")
    print(f"Worst: R={worst[1]}, G={worst[2]}, B={worst[3]} => cost: {worst[0]}")

    # Save best and worst output images
    for label, (error, r, g, b) in [("best", best), ("worst", worst)]:
        gray_guidance = (
            r * img_rgb[:, :, 0] + g * img_rgb[:, :, 1] + b * img_rgb[:, :, 2]
        ).astype(np.uint8)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, gray_guidance)
        out_dir = os.path.dirname(args.image_path)
        cv2.imwrite(
            os.path.join(out_dir, f"{label}_jbf.png"),
            cv2.cvtColor(jbf_out, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(os.path.join(out_dir, f"{label}_gray.png"), gray_guidance)
        print(f"Saved {label} result to {out_dir}/{label}_jbf.png and {label}_gray.png")


if __name__ == "__main__":
    main()
