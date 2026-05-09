import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)


def ransac_homography(src_pts, dst_pts, iterations=1000, threshold=5.0):
    """
    Estimate homography with self-implemented RANSAC.
    src_pts and dst_pts are N-by-2 arrays with dst = H(src).
    """
    num_points = src_pts.shape[0]
    if num_points < 4:
        return None, None

    src_homo = np.hstack([src_pts, np.ones((num_points, 1))])
    best_H = None
    best_mask = None
    best_inlier_count = 0
    best_mean_error = np.inf

    for _ in range(iterations):
        sample_idx = random.sample(range(num_points), 4)
        try:
            H = solve_homography(src_pts[sample_idx], dst_pts[sample_idx])
        except np.linalg.LinAlgError:
            continue

        if H is None or not np.all(np.isfinite(H)):
            continue

        projected = H @ src_homo.T
        valid = np.abs(projected[2]) > 1e-8
        if not np.any(valid):
            continue

        projected_xy = np.empty((num_points, 2), dtype=np.float64)
        projected_xy[:] = np.inf
        projected_xy[valid] = (projected[:2, valid] / projected[2, valid]).T
        errors = np.linalg.norm(projected_xy - dst_pts, axis=1)
        inlier_mask = errors < threshold
        inlier_count = np.count_nonzero(inlier_mask)

        if inlier_count < 4:
            continue

        mean_error = errors[inlier_mask].mean()
        if inlier_count > best_inlier_count or (
            inlier_count == best_inlier_count and mean_error < best_mean_error
        ):
            best_H = H
            best_mask = inlier_mask
            best_inlier_count = inlier_count
            best_mean_error = mean_error

    if best_mask is None:
        return None, None

    best_H = solve_homography(src_pts[best_mask], dst_pts[best_mask])
    if best_H[2, 2] != 0:
        best_H = best_H / best_H[2, 2]

    return best_H, best_mask


def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[: imgs[0].shape[0], : imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None
    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs) - 1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            continue

        raw_matches = bf.knnMatch(des2, des1, k=2)
        matches = []
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                matches.append(m)

        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 4:
            continue

        im2_pts = np.float32([kp2[m.queryIdx].pt for m in matches])
        im1_pts = np.float32([kp1[m.trainIdx].pt for m in matches])

        # Apply self-implemented RANSAC to choose best H.
        H, mask = ransac_homography(im2_pts, im1_pts, iterations=1000, threshold=5.0)
        if H is None:
            continue

        # Chain the homographies.
        last_best_H = np.dot(last_best_H, H)

        # Apply warping.
        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction="b")
        out = dst.copy()

    return out


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [
        cv2.imread("../resource/frame{:d}.jpg".format(x))
        for x in range(1, FRAME_NUM + 1)
    ]
    output4 = panorama(imgs)
    cv2.imwrite("output4.png", output4)
