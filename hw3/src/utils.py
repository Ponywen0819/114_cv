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

    if v.shape[0] is not N:
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
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates

        # TODO: 6. assign to destination image with proper masking

        pass

    elif direction == "f":
        target_points = H @ points
        target_points /= target_points[2:3, :]

        mask = (
            (target_points[0] >= 0)
            & (target_points[0] < w_dst)
            & (target_points[1] >= 0)
            & (target_points[1] < h_dst)
        )

        target_points = target_points[:, mask]
        source_points = points[:, mask]

        dst[target_points[1].astype(int), target_points[0].astype(int)] = src[
            source_points[1].astype(int), source_points[0].astype(int)
        ]

    return dst
