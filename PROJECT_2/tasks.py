import cv2
import numpy as np
from widgets import *

def detector1(img, threshold, gradient_filter, mode, metric, **kwargs):
    """
        gradient magnitudes + thresholding
    """
    grad = image_gradient(img, gradient_filter, mode, **kwargs)

    norm = image_magnitudes(grad, metric)

    edge = thresholding(norm, threshold)

    return edge

def detector2(img, kernel_size, sigma, mode, channel_merging, **kwargs):
    """
        Laplacian of Gaussian filter + zero crossing
    """
    result_log = log_filter(img, kernel_size, sigma, mode, **kwargs)
    result_zero_crossing = zero_crossing(result_log, channel_merging)

    return result_zero_crossing