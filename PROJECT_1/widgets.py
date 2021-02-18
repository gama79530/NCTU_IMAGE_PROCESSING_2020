import cv2
import numpy as np
import IPython

def _cvt_2uint8(array, clip=True):
    output = np.clip(array, 0, 255) if clip else array
    output = output.astype('uint8')
    return output

def jupyter_img_show(img):
    _, i_img = cv2.imencode('.png', img)
    IPython.display.display(IPython.display.Image(data=i_img))

"""
********************************
    Intensity transformation
********************************
"""

def gamma_correction(img, gamma, c=1):
    """
    Apply gamma correction on input image.
    s = 255 * c * (r / 255) ^ gamma

    Args:
        img: input image array
        gamma: the gamma value of gamma value, which is between 0 and 1.
        c: the constant of gamma_correction, which is between 0 and 1.

    Returns:
        a transformed image array.
    """
    trans_img = 255 * (c * (img / 255) ** gamma)
    # clip
    trans_img = _cvt_2uint8(trans_img)
    return trans_img

def histogram_equalization(img) :
    """
    Apply histogram equalization on input image.
    s = (L - 1) \sum_{j=0}^{k} p_{j}(r_{j})

    Args:
        img: input image array
    
    Returns:
        a transformed image array.
    """
    # tuning dimension 
    is_grayscale = len(img.shape) == 2
    img_3dim = np.expand_dims(img, axis=2) if is_grayscale else img

    # count
    count = np.zeros((256, img_3dim.shape[2]), dtype='int32')
    for c in range(img_3dim.shape[2]):
        for x in range(img_3dim.shape[0]):
            for y in range(img_3dim.shape[1]):
                    count[img_3dim[x][y][c]][c] += 1

    # Build lookup table
    lookup_table = _cvt_2uint8(255 * np.cumsum(count, axis=0) / (img_3dim.shape[0] * img_3dim.shape[1]), False)
    # apply transform
    trans_img_3dim = np.zeros(img_3dim.shape, dtype='float32')
    for x in range(img_3dim.shape[0]):
        for y in range(img_3dim.shape[1]):
            for c in range(img_3dim.shape[2]):
                trans_img_3dim[x][y][c] = lookup_table[img_3dim[x][y][c]][c]

    # tuning dimension 
    trans_img = np.squeeze(trans_img_3dim, axis=2) if is_grayscale else trans_img_3dim
    # clip
    trans_img = _cvt_2uint8(trans_img)
    return trans_img

def piecewise_linear_transformation(img, funcs, break_points):
    """
    Apply piecewise linear transformation on input image.
    The following conditions should be satisfied
        1. each function is an increasing linear function
        2. len(funcs) - len(break_points) = 1
        3. for each element b in break_points, 0 < b < 255
        4. 2 neighbor function must have same value at their common break point.

    Args:
        img: input image array
        funcs: a list of functions those are used on transformation
        break_points: a list of break point.

    Returns:
        a transformed image array.
    """
    def binary_search(array, target):
        start = 0
        end = len(array)
        while end - start > 1:
            mid = (start + end) // 2
            if array[mid] == target:
                return mid
            elif array[mid] > target:
                end = mid
            else:
                start = mid
        return start
    
    # tuning dimension 
    is_grayscale = len(img.shape) == 2
    img_3dim = np.expand_dims(img, axis=2) if is_grayscale else img
    # apply transformation
    trans_img_3dim = np.zeros(img_3dim.shape, dtype='float32')
    for x in range(trans_img_3dim.shape[0]):
        for y in range(trans_img_3dim.shape[1]):
            for c in range(trans_img_3dim.shape[2]):
                func = funcs[binary_search([0] + break_points, img_3dim[x][y][c])]
                trans_img_3dim[x][y][c] = func(img_3dim[x][y][c])

    # tuning dimension 
    trans_img = np.squeeze(trans_img_3dim, axis=2) if is_grayscale else trans_img_3dim
    # clip
    trans_img = _cvt_2uint8(trans_img)

    return trans_img

def negative_transformation(img):
    """
    Apply negative transformation on input image.

    Args:
        img: input image array
        
    Returns:
        a transformed image array.
    """
    trans_img = 255 - img
    return trans_img


"""
**************
    filter 
**************
"""
def custom_filter(img, is_order_statistics, mode='constant', clip=True, **kwargs):
    """
    Apply custom filter on input image.
    
    Args:
        img: input image array
        is_order_statistics: whether this filter is order statistics filter 
        mode: str or function, optional (used for numpy.pad)
        clip: whether clip value so that the range of value is between 0 and 255

    kwargs:
        kernel: custom filter whose shape should be (odd integer, odd integer), only used when is_order_statistics=False
        
        k_size: a tuple which contains kernel sizes, only used when is_order_statistics=True
        filtering: a function which acts on each filtering step, only used when is_order_statistics=True
        
    Returns:
        a filtered image array.
    """

    # tuning dimension 
    is_grayscale = len(img.shape) == 2
    img_3dim = np.expand_dims(img, axis=2) if is_grayscale else img
    # create output skeleton
    filter_img_3dim = np.zeros(img_3dim.shape, dtype='float32')
    # create kwargs for np.pad
    pad_kwargs = dict()
    for kwarg in ['stat_lengthsequence', 'constant_valuessequence', 'end_valuessequence', 'reflect_type']:
        if kwarg in kwargs:
            pad_kwargs[kwarg] = pad_kwargs[kwarg]

    if is_order_statistics:
        # extract parameters
        k_size = kwargs['k_size']
        if isinstance(k_size, int):
            k_dim_0 = k_size
            k_dim_1 = k_size
        else:
            k_dim_0 = k_size[0]
            k_dim_1 = k_size[1]
        filtering = kwargs['filtering']
        # pad img
        pad_dim_0, pad_dim_1 = k_dim_0//2, k_dim_1//2        
        pad_img = np.pad(img_3dim, ((pad_dim_0, pad_dim_0), (pad_dim_1, pad_dim_1), (0, 0)), mode, **pad_kwargs)
        # apply filter
        for c in range(filter_img_3dim.shape[2]):
            for x in range(filter_img_3dim.shape[0]):
                for y in range(filter_img_3dim.shape[1]):
                    sliding_window = pad_img[x:x+k_dim_0, y:y+k_dim_1, c]
                    filter_img_3dim[x][y][c] = filtering(sliding_window)
    else:
        # extract parameters
        kernel = kwargs['kernel']
        # pad img
        pad_dim_0, pad_dim_1 = kernel.shape[0]//2, kernel.shape[1]//2
        pad_img = np.pad(img_3dim, ((pad_dim_0, pad_dim_0), (pad_dim_1, pad_dim_1), (0, 0)), mode, **pad_kwargs)
        # apply filter
        for c in range(filter_img_3dim.shape[2]):
            for x in range(filter_img_3dim.shape[0]):
                for y in range(filter_img_3dim.shape[1]):
                    sliding_window = pad_img[x:x+kernel.shape[0], y:y+kernel.shape[1], c]
                    filter_img_3dim[x][y][c] = np.sum(sliding_window * kernel)

    # tuning dimension 
    filter_img = np.squeeze(filter_img_3dim, axis=2) if is_grayscale else filter_img_3dim
    # clip
    filter_img = _cvt_2uint8(filter_img) if clip else filter_img

    return filter_img

def laplacian_filter(img, add_original=True, mode='constant', **kwargs):
    """
    Apply Laplacian filter on input image.
    
    Args:
        img: input image array
        add_original: whether adding original image after applying laplacian filter
        p_mode: str or function, optional (used for numpy.pad)
        
    kwargs:
        constant_values: Used in 'constant'. The values to set the padded values for each axis. 

    Returns:
        a filtered image array.
    """
    # create kernel
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    # apply filter
    kwargs['kernel'] = kernel
    if add_original:
        filter_img = img + custom_filter(img, False, mode, False, **kwargs)
        filter_img = _cvt_2uint8(filter_img)
    else:
        filter_img = custom_filter(img, False, mode, **kwargs)
    
    return filter_img

def gaussian_filter(img, k_size, sigma=1, mode='constant', **kwargs):
    """
    Apply gaussian filter on input image.
    
    Args:
        img: input image array
        k_size: kernel size, must be an odd integer 
        sigma: variance of gaussian distribution
        mode: str or function, optional (used for numpy.pad)
    
    kwargs:
        constant_values: Used in 'constant'. The values to set the padded values for each axis. 
    
    Returns:
        a filtered image array.
    """
    # create kernel
    start, end = -1 * (k_size // 2), k_size // 2 + 1
    x, y = np.mgrid[start:end, start:end]
    kernel = np.exp((x**2 + y**2)/(-2 * sigma**2)) / (2 * np.pi * sigma**2)

    # apply filter
    kwargs['kernel'] = kernel
    filter_img = custom_filter(img, False, mode, **kwargs)

    return filter_img

def median_filter(img, k_size, mode='constant', **kwargs):
    """
    Apply median filter on input image.
    
    Args:
        img: input image array
        k_size: kernel size tuple, must be odd integers
        mode: str or function, optional (used for numpy.pad)

    kwargs:
        constant_values: Used in 'constant'. The values to set the padded values for each axis. 

    Returns:
        a filtered image array.
    """
    # define single step filtering
    def filtering(sliding_window):
        return np.median(sliding_window)
    
    kwargs['k_size'] = k_size
    kwargs['filtering'] = filtering
    filter_img = custom_filter(img, True, mode, **kwargs)

    return filter_img

def avg_filter(img, k_size, crop_extreme=False, mode='constant', **kwargs):
    """
    Apply average filter on input image.
    
    Args:
        img: input image array
        k_size: kernel size tuple, must be odd integers
        crop_extreme: drop extreme value of sliding window when this value is true
        mode: str or function, optional (used for numpy.pad)

    kwargs:
        constant_values: Used in 'constant'. The values to set the padded values for each axis. 

    Returns:
        a filtered image array.
    """
    # define single step filtering
    def filtering(sliding_window):
        sliding_array = np.array(sliding_window.flatten())
        if crop_extreme:
            sliding_array = np.sort(sliding_array)
            sliding_array = sliding_array[1:-1]
        return np.mean(sliding_array)

    kwargs['k_size'] = k_size
    kwargs['filtering'] = filtering
    filter_img = custom_filter(img, True, mode, **kwargs)
        
    return filter_img

def mid_point_filter(img, k_size, mode='constant', **kwargs):
    """
    Apply mid point filter on input image.
    
    Args:
        img: input image array
        k_size: kernel size tuple, must be odd integers
        mode: str or function, optional (used for numpy.pad)

    kwargs:
        constant_values: Used in 'constant'. The values to set the padded values for each axis. 

    Returns:
        a filtered image array.
    """
    # define single step filtering
    def filtering(sliding_window):
        return 0.5 * (0. + np.amax(sliding_window) + np.amin(sliding_window))
    
    kwargs['k_size'] = k_size
    kwargs['filtering'] = filtering
    filter_img = custom_filter(img, True, mode, **kwargs)

    return filter_img

def alpha_trimmed_filter(img, k_size, trim_len=0, mode='constant', **kwargs):
    """
    Apply mid point filter on input image.
    
    Args:
        img: input image array
        k_size: kernel size tuple, must be odd integers
        trim_len: trimming length, trim_len <= k_size^2 // 2
        mode: str or function, optional (used for numpy.pad)

    kwargs:
        constant_values: Used in 'constant'. The values to set the padded values for each axis. 

    Returns:
        a filtered image array.
    """
    # define single step filtering
    def filtering(sliding_window):
        sliding_array = np.array(sliding_window.flatten())
        sliding_array = np.sort(sliding_array)
        return np.mean(sliding_array[trim_len:-1*trim_len])

    kwargs['k_size'] = k_size
    kwargs['filtering'] = filtering
    filter_img = custom_filter(img, True, mode, **kwargs)

    return filter_img

"""
************************
    Color correction
************************
"""
def shift_HLS(img, axis, value):
    """
        Shift value of HLS model.

    Args:
        img: BGR image array
        axis: 0 -> hue 
              1 -> light
              2 -> saturation
              "all" -> all axis
        value: shift value
 
    Returns:
        A shifted image with BGR format

    """
    # convert color space
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)

    # shifting
    shift_img = np.array(hls_img, dtype='float32')
    if axis == 'all':
        hls_img = hls_img + value
    else:
        shift_img[:, :, axis] = shift_img[:, :, axis] + value

    # convert color space
    shift_img = _cvt_2uint8(shift_img)
    shift_img = cv2.cvtColor(shift_img, cv2.COLOR_HLS2BGR_FULL)

    return shift_img

def amplify_HLS(img, axis, value):
    """
        Amplify value of HLS model.

    Args:
        img: BGR image array
        axis: 0 -> hue 
              1 -> light
              2 -> saturation
              "all" -> all axis
        value: amplify value
 
    Returns:
        A amplified image with BGR format

    """
    # convert color space
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)

    # amplify
    amplify_img = np.array(hls_img, dtype='float32')
    if axis == 'all':
        amplify_img = amplify_img * value
    else:
        amplify_img[:, :, axis] = amplify_img[:, :, axis] * value
    
    # convert color space
    shift_img = _cvt_2uint8(amplify_img)
    amplify_img = cv2.cvtColor(amplify_img, cv2.COLOR_HLS2BGR_FULL)

    return amplify_img

def shift_HSV(img, axis, value):
    """
        Shift value of HSV model.

    Args:
        img: BGR image array
        axis: 0 -> hue 
              1 -> saturation
              2 -> value
              "all" -> all axis
        value: shift value
 
    Returns:
        A shifted image with BGR format

    """
    # convert color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    # shifting
    shift_img = np.array(hsv_img, dtype='float32')
    if axis == 'all':
        shift_img = shift_img + value
    else:
        shift_img[:, :, axis] = shift_img[:, :, axis] + value

    # convert color space
    shift_img = _cvt_2uint8(shift_img)
    shift_img = cv2.cvtColor(shift_img, cv2.COLOR_HSV2BGR_FULL)

    return shift_img

def amplify_HSV(img, axis, value):
    """
        Amplify value of HSV model.

    Args:
        img: BGR image array
        axis: 0 -> hue 
              1 -> saturation
              2 -> value
              "all" -> all axis
        value: amplify value
 
    Returns:
        A amplified image with BGR format

    """
    # convert color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    # amplify
    amplify_img = np.array(hsv_img, dtype='float32')
    if axis == 'all':
        amplify_img = amplify_img * value
    else:
        amplify_img[:, :, axis] = amplify_img[:, :, axis] * value

    # convert color space
    shift_img = _cvt_2uint8(amplify_img)
    amplify_img = cv2.cvtColor(amplify_img, cv2.COLOR_HSV2BGR_FULL)

    return amplify_img


def shift_XYZ(img, axis, value):
    """
        Shift value of XYZ model.

    Args:
        img: BGR image array
        axis: 0 -> X 
              1 -> Y
              2 -> Z
              "all" -> XYZ
        value: shift value
 
    Returns:
        A shifted image with BGR format

    """
    # convert color space
    xyz_img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

    # shifting
    shift_img = np.array(xyz_img, dtype='float32')
    if axis == 'all':
        shift_img = xyz_img + value
    else:
        shift_img[:, :, axis] = shift_img[:, :, axis] + value

    # convert color space
    shift_img = _cvt_2uint8(shift_img)
    shift_img = cv2.cvtColor(shift_img, cv2.COLOR_XYZ2BGR)

    return shift_img

def amplify_XYZ(img, axis, value):
    """
        Amplify value of XYZ model.

    Args:
        img: BGR image array
        axis: 0 -> X 
              1 -> Y
              2 -> Z
              'all' -> XYZ
        value: amplify value
 
    Returns:
        A amplified image with BGR format

    """
    # convert color space
    xyz_img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

    # amplify
    amplify_img = np.array(xyz_img, dtype='float32')
    if axis == 'all':
        amplify_img = amplify_img * value
    else:
        amplify_img[:, :, axis] = amplify_img[:, :, axis] * value

    # convert color space
    amplify_img = _cvt_2uint8(amplify_img)
    amplify_img = cv2.cvtColor(amplify_img, cv2.COLOR_XYZ2BGR)

    return amplify_img

