import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis=2), 3, axis=2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)


def main():
    parser = argparse.ArgumentParser(
        description="main function of Difference of Gaussian"
    )
    parser.add_argument(
        "--threshold",
        default=5.0,
        type=float,
        help="threshold value for feature selection",
    )
    parser.add_argument(
        "--image_path", default="part1/testdata/2.png", help="path to input image"
    )
    args = parser.parse_args()

    print("Processing %s ..." % args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    dog = Difference_of_Gaussian(threshold=args.threshold)

    dog_images = dog.get_octave_images(img)
    for octave in range(dog.num_octaves):
        for i in range(dog.num_DoG_images_per_octave):
            save_path = "./output/octave%d_DoG%d.png" % (octave, i)
            minmax_img = cv2.normalize(
                dog_images[octave][i],
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            )
            cv2.imwrite(save_path, minmax_img)

    keypoints = dog.get_keypoint_from_DoG(dog_images)
    plot_keypoints(img, keypoints, "./output/keypoints_%f.png" % args.threshold)

    ### TODO ###


if __name__ == "__main__":
    main()
