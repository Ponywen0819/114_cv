import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2 ** (1 / 4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_octave_images(self, image):
        octave_images = []
        for octave in range(self.num_octaves):
            base_image = cv2.resize(
                image,
                None,
                fx=1.0 / (2**octave),
                fy=1.0 / (2**octave),
                interpolation=cv2.INTER_NEAREST,
            )
            gaussian_images = [base_image] + [
                cv2.GaussianBlur(base_image, ksize=(0, 0), sigmaX=self.sigma**k)
                for k in range(1, self.num_guassian_images_per_octave + 1)
            ]
            octave_images.append(gaussian_images)
        # Step 2: Build DoG pyramid (2 octaves, 4 DoG images per octave)
        # - For each octave, subtract adjacent Gaussian images:
        #   DoG_i = Gaussian_{i+1} - Gaussian_i.
        # - Use cv2.subtract(second_image, first_image).
        # - DoG image shape is the same as its corresponding Gaussian image.

        dog_images = []
        for octave in range(self.num_octaves):
            octave_dog_images = [
                cv2.subtract(octave_images[octave][i + 1], octave_images[octave][i])
                for i in range(self.num_DoG_images_per_octave)
            ]
            dog_images.append(np.stack(octave_dog_images, axis=0))

        return dog_images

    def get_keypoint_from_DoG(self, dog_images):
        keypoints = []
        for octave in range(self.num_octaves):
            # pad_size = 1
            # padded_data = np.pad(
            #     dog_images[octave], pad_size, mode="constant", constant_values=0
            # )
            windows = np.lib.stride_tricks.sliding_window_view(
                dog_images[octave], (3, 3, 3)
            )  # shape: (H-2, W-2, 3, 3, 3)

            # Exclude center pixel (index 13 in flattened 3x3x3) from neighbor comparison
            flat = windows.reshape(*windows.shape[:3], -1)  # (4, H, W, 27)

            flat_no_center_max = flat.copy().astype(np.float64)
            flat_no_center_min = flat.copy().astype(np.float64)
            flat_no_center_max[..., 13] = -np.inf
            flat_no_center_min[..., 13] = np.inf

            max_values = np.max(flat_no_center_max, axis=-1)  # max of 26 neighbors
            min_values = np.min(flat_no_center_min, axis=-1)  # min of 26 neighbors

            # Center pixels corresponding to windows: dog_images[octave][1:3, 1:-1, 1:-1]
            center = dog_images[octave][1:3, 1:-1, 1:-1]  # shape: (2, H-2, W-2)
            key_points = (
                (center > max_values) | (center < min_values)
            ) & (np.abs(center) > self.threshold)

            # argwhere gives [d, i, j] in (2, H-2, W-2) space; actual coords are [i+1, j+1]
            keypoints.extend(
                (np.argwhere(key_points)[:, 1:] + 1) * (2**octave)
            )  # shape: (N, 2), scale to original image

        # Step 4: Remove duplicate keypoints
        # - Use np.unique(..., axis=0).
        # - Expected shape after this step: (N, 2).
        keypoints = np.unique(np.array(keypoints), axis=0)
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(-1, 2)

        # Step 5: Sort keypoints
        # Sort points using np.lexsort((col, row)) -> primary key col, secondary key row.
        if len(keypoints) > 0:
            keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints

    def get_keypoints(self, image):
        """
        Detect DoG keypoints from a grayscale image.

        Args:
            image (np.ndarray): Input grayscale image with shape (H, W) (np.float64).

        Returns:
            keypoints (np.ndarray): Array with shape (N, 2)
        """
        # TODO:
        # Step 1: Build Gaussian pyramid (2 octaves, 5 Gaussian images per octave)
        # - Use cv2.GaussianBlur(src, ksize=(0, 0), sigmaX=self.sigma**k).
        # - Octave 1 image shape: (H, W).
        # - Octave 2 base image: downsample octave 1's last image (sigma**4) by 2 using
        #   cv2.resize(..., interpolation=cv2.INTER_NEAREST),
        #   so shape becomes (H//2, W//2).

        # octave_images = []
        # for octave in range(self.num_octaves):
        #     base_image = cv2.resize(
        #         image,
        #         None,
        #         fx=1.0 / (2**octave),
        #         fy=1.0 / (2**octave),
        #         interpolation=cv2.INTER_NEAREST,
        #     )
        #     gaussian_images = [base_image] + [
        #         cv2.GaussianBlur(base_image, ksize=(0, 0), sigmaX=self.sigma**k)
        #         for k in range(1, self.num_guassian_images_per_octave + 1)
        #     ]
        #     octave_images.append(gaussian_images)
        # octave_images = np.array(octave_images)
        # Step 2: Build DoG pyramid (2 octaves, 4 DoG images per octave)
        # - For each octave, subtract adjacent Gaussian images:
        #   DoG_i = Gaussian_{i+1} - Gaussian_i.
        # - Use cv2.subtract(second_image, first_image).
        # - DoG image shape is the same as its corresponding Gaussian image.

        # dog_images = []
        # for octave in range(self.num_octaves):
        #     octave_dog_images = [
        #         cv2.subtract(octave_images[octave][i + 1], octave_images[octave][i])
        #         for i in range(self.num_DoG_images_per_octave)
        #     ]
        #     dog_images.append(octave_dog_images)
        # dog_images = np.stack(dog_images, axis=0)

        dog_images = self.get_octave_images(image)

        # Step 3: Threshold and find 3D local extrema in DoG volume
        # - Ignore 1-pixel image border.
        # - For each valid pixel in DoG images 1,2 of each octave, compare against
        #   its 26 neighbors in a 3x3x3 neighborhood.
        # - Keep [y, x] as a keypoint if:
        #   (1) it is a local maximum or minimum (>= max or <= min), and
        #   (2) abs(DoG value) > self.threshold.
        # - Coordinates stored in keypoints must be in original image scale:
        #   octave 1 -> [y, x], octave 2 -> [2*y, 2*x].
        keypoints = self.get_keypoint_from_DoG(dog_images)
        return keypoints
