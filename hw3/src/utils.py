import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] != N:
        print("u and v should have the same size")
        return None
    if N < 4:
        print("At least 4 points should be given")

    dtype = np.result_type(u.dtype, v.dtype, np.float64)
    x = u[:, 0].astype(dtype, copy=False)
    y = u[:, 1].astype(dtype, copy=False)
    xp = v[:, 0].astype(dtype, copy=False)
    yp = v[:, 1].astype(dtype, copy=False)

    A = np.empty((2 * N, 9), dtype=dtype)
    A[0::2, 0] = x
    A[0::2, 1] = y
    A[0::2, 2] = 1
    A[0::2, 3:6] = 0
    A[0::2, 6] = -xp * x
    A[0::2, 7] = -xp * y
    A[0::2, 8] = -xp
    A[1::2, 0:3] = 0
    A[1::2, 3] = x
    A[1::2, 4] = y
    A[1::2, 5] = 1
    A[1::2, 6] = -yp * x
    A[1::2, 7] = -yp * y
    A[1::2, 8] = -yp

    # _, eigvecs = np.linalg.eigh(A.T @ A)
    # H = eigvecs[:, 0].reshape(3, 3)
    # if H[2, 2] != 0:
    #     H /= H[2, 2]

    U, S, Vt = np.linalg.svd(A)

    H = Vt[-1].reshape(3, 3)

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction="b"):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    xs, ys = np.meshgrid(np.arange(w_src), np.arange(h_src))

    ones = np.ones_like(xs)
    points = np.stack([xs, ys, ones], axis=0)  # shape = (3, H_src, W_src)
    points = points.reshape(3, -1)  # shape = (3, N)

    if direction == "b":
        # Apply H_inv to destination pixels and reshape back to the ROI.
        dst_xs, dst_ys = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
        dst_points = np.stack(
            [dst_xs, dst_ys, np.ones_like(dst_xs)], axis=0
        ).reshape(3, -1)

        source_points = H_inv @ dst_points
        source_points /= source_points[2:3, :]
        source_xs = source_points[0].reshape(ymax - ymin, xmax - xmin)
        source_ys = source_points[1].reshape(ymax - ymin, xmax - xmin)

        # Keep only coordinates that are valid in both source and destination.
        mask = (
            (source_xs >= 0)
            & (source_xs < w_src)
            & (source_ys >= 0)
            & (source_ys < h_src)
            & (dst_xs >= 0)
            & (dst_xs < w_dst)
            & (dst_ys >= 0)
            & (dst_ys < h_dst)
        )

        sampled_pixels = src[
            source_ys[mask].astype(int), source_xs[mask].astype(int)
        ]

        dst[dst_ys[mask], dst_xs[mask]] = sampled_pixels

    elif direction == "f":
        target_points = H @ points
        target_points /= target_points[2:3, :]

        target_xs = target_points[0]
        target_ys = target_points[1]
        source_pixels = src[
            points[1].astype(int), points[0].astype(int)
        ].astype(np.float64)

        finite_mask = np.isfinite(target_xs) & np.isfinite(target_ys)
        target_xs = target_xs[finite_mask]
        target_ys = target_ys[finite_mask]
        source_pixels = source_pixels[finite_mask]

        if target_xs.size == 0:
            return dst

        x0 = np.floor(target_xs).astype(int)
        y0 = np.floor(target_ys).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = target_xs - x0
        wy1 = target_ys - y0
        wx0 = 1 - wx1
        wy0 = 1 - wy1

        neighbor_xs = np.concatenate([x0, x1, x0, x1])
        neighbor_ys = np.concatenate([y0, y0, y1, y1])
        weights = np.concatenate(
            [wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1]
        )
        neighbor_pixels = np.tile(source_pixels, (4, 1))

        mask = (
            (neighbor_xs >= 0)
            & (neighbor_xs < w_dst)
            & (neighbor_ys >= 0)
            & (neighbor_ys < h_dst)
            & (weights > 0)
        )

        if not np.any(mask):
            return dst

        accum = np.zeros((h_dst, w_dst, ch), dtype=np.float64)
        weight_sum = np.zeros((h_dst, w_dst), dtype=np.float64)
        valid_xs = neighbor_xs[mask]
        valid_ys = neighbor_ys[mask]
        valid_weights = weights[mask]

        np.add.at(
            accum,
            (valid_ys, valid_xs),
            neighbor_pixels[mask] * valid_weights[:, None],
        )
        np.add.at(weight_sum, (valid_ys, valid_xs), valid_weights)

        filled = weight_sum > 0
        interpolated = accum[filled] / weight_sum[filled][:, None]
        if np.issubdtype(dst.dtype, np.integer):
            dtype_info = np.iinfo(dst.dtype)
            interpolated = np.clip(
                np.rint(interpolated), dtype_info.min, dtype_info.max
            )
        dst[filled] = interpolated.astype(dst.dtype, copy=False)

    return dst
