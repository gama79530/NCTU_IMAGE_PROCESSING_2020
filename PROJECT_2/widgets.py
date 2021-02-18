import cv2
import numpy as np
import IPython
import bisect

def cvt_uint8(array):
    """
        Convert dtype of input array into an uint8 array.
    """
    ret_array = np.clip(array, 0, 255)
    ret_array = ret_array.astype('uint8')
    return ret_array

def jupyter_img_show(img):
    """
        Show image on jupyter notebook.
    """
    _, i_img = cv2.imencode('.png', img)
    IPython.display.display(IPython.display.Image(data=i_img))

def filtering2D(img, channel='all', cvt_2uint8=True, mode='constant', **kwargs):
    """
    Applying 2D filter on the input image.
    
    Args:
        img: An input image ndarray
        channel: Applying filter on which channel. This parameter will be ignored when input image is a grayscale image.
        cvt_2uint8: Whether convert the output to a uint8 ndarray
        mode: Used on np.pad

        kernel: 2D filter array which is used on weight filtering. 
        kernel_size: Filter size tuple which is used on statistics filtering. Be ignored when kwarg 'kernel' exists.
        filter_func: Filtering function which is used on statistics filtering. Be ignored when kwarg 'kernel' exists.

        Others: Other **kwargs which is used on np.pad

    Returns:
        A filtered image array.
    """
    def filtering(img_slice):
        filtered_slice = np.array(img_slice, dtype='float32')
        pad_img = np.pad(img_slice, pad_width, mode, **p_kwargs)
        for x in range(img_slice.shape[0]):
            for y in range(img_slice.shape[1]):
                sliding_window = pad_img[x:x+kernel_size[0], y:y+kernel_size[1]]
                if is_statistics:
                    filtered_slice[x][y] = filter_func(sliding_window)
                else:
                    filtered_slice[x][y] = np.sum(sliding_window * kernel)
        return filtered_slice
    
    # extract parameters
    is_grayscale = len(img.shape) == 2
    is_statistics = not 'kernel' in kwargs
    if is_statistics:
        kernel_size, filter_func = kwargs['kernel_size'], kwargs['filter_func']
    else:
        kernel = kwargs['kernel']
        kernel_size = kernel.shape
    
    pad_x, pad_y = kernel_size[0]//2, kernel_size[1]//2
    pad_width = [(pad_x, pad_x), (pad_y, pad_y)]
    p_kwargs = dict()
    for kwarg in ['stat_length', 'constant_values', 'end_values', 'reflect_type']:
        if kwarg in kwargs:
            p_kwargs[kwarg] = kwargs[kwarg]
    # filtering
    if is_grayscale:
        filtered_img = filtering(img)
    else:
        filtered_img = np.array(img, dtype='float32')
        channels = range(img.shape[2]) if channel == 'all' else [channel]
        for c in channels:
            filtered_img[:, :, c] = filtering(filtered_img[:, :, c])
    # clip and convert dtype
    if cvt_2uint8:
        filtered_img = cvt_uint8(filtered_img)

    return filtered_img
    
def image_gradient(img, gradient_filter='sobel', mode='constant', **kwargs):
    """
    Image gradient calculation.
    
    Args:
        img: An input image ndarray
        gradient_filter: 'prewitt' or 'sobel'
        mode: Used on np.pad

        Others: Other **kwargs which is used on np.pad

    Returns:
        Image gradient (dx, dy)
    """
    # create filter
    if gradient_filter == 'prewitt':
        filter_x = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        filter_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    else:
        filter_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        filter_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # apply filter
    kwargs_x = dict(kwargs)
    kwargs_x['kernel'] = filter_x
    gradient_x = filtering2D(img, cvt_2uint8=False, mode=mode, **kwargs_x)
    kwargs_y = dict(kwargs)
    kwargs_y['kernel'] = filter_y
    gradient_y = filtering2D(img, cvt_2uint8=False, mode=mode, **kwargs_y)
    
    return gradient_x, gradient_y

def thresholding(array, threshold, cvt_2uint8=True, channel='all'):
    """
    Applying thresholding on an ndarray.
    
    Args:
        array: An input image ndarray
        threshold: Threshold value
        cvt_2uint8: Output uint8 dtype or not
        channel: Thresholding on which channel. This parameter will be ignored when input image is a grayscale image.
    Returns:
        ndarray after thresholding
    """
    def apply_thresholding(array_slice):
        return np.array(array_slice >= threshold)

    # extract parameters
    is_grayscale = len(array.shape) == 2
    # thresholding
    if is_grayscale:
        ret_array = apply_thresholding(array)
    else:
        channels = range(array.shape[2]) if channel == 'all' else [channel]
        ret_array = np.array(array, dtype='float32')
        for c in channels:
            ret_array[:, :, c] = apply_thresholding(ret_array[:, :, c])
    
    if cvt_2uint8:
        ret_array = cvt_uint8(255 * ret_array)

    return ret_array

def image_magnitudes(image_gradient, metric='L1'):
    """
    Image magnitudes calculation.
    
    Args:
        image_gradient: A tuple which is consist with image gradient (dx, dy)
        metric: Use which metric to calculate norm.

    Returns:
        ndarray magnitudes array
    """
    # extract parameters
    is_grayscale = len(image_gradient[0].shape) == 2
    if metric == 'L2':
        norm_ord = 2
    else:
        norm_ord = 1
    gradient = np.stack(image_gradient, axis=2) if is_grayscale else np.concatenate(image_gradient, axis=2)
    # calculation
    norm = np.linalg.norm(gradient, ord=norm_ord, axis=2)

    return norm

def log_filter(img, kernel_size, sigma=1, mode='constant', **kwargs):
    """
    Apply Laplacian of Gaussian (LoG) filter on input image.

    LoG(x, y) = [x^2 + y^2 - 2 * sigma^2] * exp[(x^2 + y^2) / (-2 * sigma^2)] / [sigma^4]
    
    Args:
        img: An input image ndarray
        kernel_size: kernel size, must be an odd integer 
        sigma: standard deviation of gaussian distribution
        mode: Used on np.pad

        Others: Other **kwargs which is used on np.pad

    Returns:
        A filtered image array.
    """
    # create kernel
    start, end = -1 * (kernel_size // 2), kernel_size // 2 + 1
    x, y = np.mgrid[start:end, start:end]
    ## for readability
    f1 = 2 * sigma**2 - (x**2 + y**2)
    f2 = np.exp((x**2 + y**2)/(-2 * sigma**2))
    f3 = sigma**4
    kernel = f1 * f2 / f3
    # apply filter
    kwargs['kernel'] = kernel
    filtered_img = filtering2D(img, cvt_2uint8=False, mode=mode, **kwargs)

    return filtered_img

def zero_crossing(array, channel_merging=1):
    """
    Apply zero crossing on a ndarray
    
    Args:
        array: A ndarray
        channel_merging: How to merge multiple channel 
            0: or
            1: and

    Returns:
        A zero crossed ndarray
    """
    def filter_func(sliding_window):
        pos = np.sum(sliding_window > 0)
        neg = np.sum(sliding_window < 0)
        ret = True if pos > 0 and neg > 0 else False
        
        return ret

    # extract parameters
    is_grayscale = len(array.shape) == 2
    # do zero crossing
    kwargs = {'kernel_size' : (3,3), 'filter_func' : filter_func}
    filter_img = filtering2D(array, mode='edge', cvt_2uint8=False, **kwargs)

    if not is_grayscale:
        if channel_merging == 0:
            filter_img = np.any(filter_img, axis=2)
        else:
            filter_img = np.all(filter_img, axis=2)
        
    filter_img = cvt_uint8(255 * filter_img)

    return filter_img

def range_transform(array, channel='all'):
    """
    Transform the range of array such that all values are between 0 and 255
    
    Args:
        array: A ndarray
        channel: How to handel multiple channel 
            'all' : normilize by the global maximum
            'channelwise' : normilize by maximum of each channel
            int : specified channel only

    Returns:
        A transformed ndarray
    """
    if channel == 'all' or len(array.shape) == 2:
        max_intensity = np.max(np.absolute(array))
        trans_array = 255 * array / max_intensity 
    else:
        channels = range(len(array.shape[2])) if channel == 'channelwise' else [channel]
        trans_array = np.array(array, dtype=np.float)
        for c in channels:
            max_intensity = np.max(np.absolute(array[:, :, c]))
            trans_array[:, :, c] = 255 * trans_array[:, :, c] / max_intensity

    return cvt_uint8(trans_array)

class Hough_Line_Detection:
    def __init__(self, img, resolution, line_stretch=False, edge_weights=None):
        """
        Args:
            img: image ndarray
            resolution: a resolution tuple (theta, rho) in Hough space.
            line_stretch: draw line with infinite length
            edge_weights: use external algorithm to perform edge detection.
        """
        self.img = img
        self.resolution = resolution
        self.line_stretch = line_stretch
        self.edge_weights = edge_weights
        
        self.hough_weights = np.zeros(resolution, dtype=np.float)     
        self.thetas = np.mgrid[0:self.resolution[0]] * np.pi / (self.resolution[0] - 1)
        self.cos_thetas = np.cos(self.thetas)
        self.sin_thetas = np.sin(self.thetas)
        self.rho_range = (img.shape[0]**2 + img.shape[1]**2)**(0.5)
        self.rho_unit = 2 * self.rho_range / (resolution[1] - 1)
        self.line_head = None if line_stretch else max(img.shape) * np.ones(resolution, dtype=np.int)
        self.line_end = None if line_stretch else -1 * np.ones(resolution, dtype=np.int)

        self.edge_candidate = None
        self.line_img = None 
    def prepare(self, **kwargs):
        """
        Args:
            edge_threshold: threshold which is used on edge detection.
            gradient_filter: 'prewitt' or 'sobel'
            mode: Used on np.pad
            metric: Use which metric to calculate norm.
            
            Others: Other **kwargs which is used on np.pad
        """
        msg = ""
        # perform edge detect if needed
        if not self.edge_weights:
            edge_kwargs = dict()
            for kwarg in ['stat_length', 'constant_values', 'end_values', 'reflect_type']:
                if kwarg in kwargs:
                    edge_kwargs[kwarg] = self.kwargs[kwarg]
            edge_threshold = kwargs['edge_threshold'] if 'edge_threshold' in kwargs else None
            gradient_filter = kwargs['gradient_filter']
            mode = kwargs['mode']
            metric = kwargs['metric']
            msg += self._edge_detect(edge_threshold, gradient_filter, mode, metric, **edge_kwargs)
        msg += self._hough_transform()

        return msg
    
    def line_detect(self, threshold, line_width=1):
        def rho_idx_to_rho(rho_idx):
            return self.rho_unit * rho_idx - self.rho_range

        self.line_img = np.array(self.img)
        self.edge_candidate = np.argwhere(self.hough_weights >= threshold)

        for theta_idx, rho_idx in self.edge_candidate:
            theta = self.thetas[theta_idx]
            cos_theta = self.cos_thetas[theta_idx]
            sin_theta = self.sin_thetas[theta_idx]
            rho = rho_idx_to_rho(rho_idx)
            if self.line_stretch:
                x0 = rho * cos_theta
                y0 = rho * sin_theta
                x1 = int(x0 - sin_theta * self.rho_range)
                y1 = int(y0 + cos_theta * self.rho_range)
                x2 = int(x0 + sin_theta * self.rho_range)
                y2 = int(y0 - cos_theta * self.rho_range)
            else:
                if theta == 0 or theta == np.pi:
                    x1 = x2 = int(rho/cos_theta)
                    y1 = self.line_head[theta_idx][rho_idx]
                    y2 = self.line_end[theta_idx][rho_idx]
                else:
                    x1 = self.line_head[theta_idx][rho_idx]
                    y1 = int((rho - x1 * cos_theta) / sin_theta)
                    x2 = self.line_end[theta_idx][rho_idx]
                    y2 = int((rho - x2 * cos_theta) / sin_theta)

            self.line_img = cv2.line(self.line_img, (y1, x1), (y2, x2), (0, 0, 255), line_width) # cv2 coordinate (x, y) is (c, r) in numpy
        return self.line_img

    def _edge_detect(self, threshold, gradient_filter, mode, metric, **kwargs):
        """
            gradient magnitudes [+ thresholding (*optional)]
        """
        img_grad = image_gradient(self.img, gradient_filter, mode, **kwargs)
        img_norm = image_magnitudes(img_grad, metric)
        if threshold:
            self.edge_weights = thresholding(img_norm, threshold, False)
            msg = "====== Edge weights with 1: {} and 0: {} ======\n".format(str(np.sum(self.edge_weights == True)), str(np.sum(self.edge_weights == False)))
        else:
            self.edge_weights = img_norm
            msg = "====== Edge weights are between {} and {} ======\n".format(str(np.min(self.edge_weights)), str(np.max(self.edge_weights)))
        return msg

    def _hough_transform(self):
        for x, y in np.argwhere(self.edge_weights > 0):
            rhos = x * self.cos_thetas + y * self.sin_thetas
            # rho to rho_idx
            rho_idxs = (rhos + self.rho_range) / self.rho_unit
            rho_idxs = np.rint(rho_idxs)
            rho_idxs = np.array(rho_idxs, dtype=np.int)
            for theta_idx, rho_idx in enumerate(rho_idxs):
                self.hough_weights[theta_idx][rho_idx] += self.edge_weights[x][y]
                # save posible line end point index
                if self.line_stretch:
                    continue
                if theta_idx == 0 or theta_idx == self.resolution[0] - 1:
                    if self.line_head[theta_idx][rho_idx] > y:
                        self.line_head[theta_idx][rho_idx] = y
                    if self.line_end[theta_idx][rho_idx] < y:
                        self.line_end[theta_idx][rho_idx] = y
                else:
                    if self.line_head[theta_idx][rho_idx] > x:
                        self.line_head[theta_idx][rho_idx] = x
                    if self.line_end[theta_idx][rho_idx] < x:
                        self.line_end[theta_idx][rho_idx] = x

        return "====== Hough weights are between {} and {} ======\n".format(str(np.min(self.hough_weights)), str(np.max(self.hough_weights)))
