import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        """
        Parameters
        ----------
        img : ndarray
            The image to be filtered.

        guidance : ndarray
            The guidance image used to compute range weights.
            It can be either:
              - grayscale: shape (H, W)
              - color:     shape (H, W, C)

        Returns
        -------
        output : ndarray
            The filtered result with the same shape as img.
        """
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        H, W = img.shape[:2]

        # Precompute spatial Gaussian kernel
        ky = np.arange(self.wndw_size) - self.pad_w
        kx = np.arange(self.wndw_size) - self.pad_w
        ky_grid, kx_grid = np.meshgrid(ky, kx, indexing='ij')
        spatial_kernel = np.exp(-(ky_grid**2 + kx_grid**2) / (2 * self.sigma_s**2))

        # Normalize guidance to [0, 1] for range kernel
        padded_guidance_norm = padded_guidance / 255.0

        # Build sliding window views: (H, W, wndw, wndw) or (H, W, wndw, wndw, C)
        wndw = self.wndw_size
        if padded_guidance_norm.ndim == 2:
            g_windows = np.lib.stride_tricks.sliding_window_view(padded_guidance_norm, (wndw, wndw))
            # (H, W, wndw, wndw)
            center = padded_guidance_norm[self.pad_w:self.pad_w+H, self.pad_w:self.pad_w+W, np.newaxis, np.newaxis]
            diff = center - g_windows  # (H, W, wndw, wndw)
            range_kernel = np.exp(-diff**2 / (2 * self.sigma_r**2))
        else:
            C_g = padded_guidance_norm.shape[2]
            g_windows = np.lib.stride_tricks.sliding_window_view(padded_guidance_norm, (wndw, wndw, C_g))[:, :, 0]
            # (H, W, wndw, wndw, C_g)
            center = padded_guidance_norm[self.pad_w:self.pad_w+H, self.pad_w:self.pad_w+W, np.newaxis, np.newaxis, :]
            diff = center - g_windows  # (H, W, wndw, wndw, C_g)
            range_kernel = np.exp(-np.sum(diff**2, axis=-1) / (2 * self.sigma_r**2))
            # (H, W, wndw, wndw)

        # Combined weight: (H, W, wndw, wndw)
        weight = spatial_kernel * range_kernel
        weight_sum = np.sum(weight, axis=(-2, -1))  # (H, W)

        # Weighted sum over image windows
        if padded_img.ndim == 2:
            img_windows = np.lib.stride_tricks.sliding_window_view(padded_img, (wndw, wndw))
            # (H, W, wndw, wndw)
            output = np.sum(weight * img_windows, axis=(-2, -1)) / weight_sum
        else:
            C_i = img.shape[2]
            img_windows = np.lib.stride_tricks.sliding_window_view(padded_img, (wndw, wndw, C_i))[:, :, 0]
            # (H, W, wndw, wndw, C_i)
            output = np.sum(weight[:, :, :, :, np.newaxis] * img_windows, axis=(2, 3)) / weight_sum[:, :, np.newaxis]
            # (H, W, C_i)

        return np.clip(output, 0, 255).astype(np.uint8)