import numpy as np
import cv2, IPython
import math

def jupyter_img_show(img):
    """
        Show image on jupyter notebook.
    """
    _, i_img = cv2.imencode('.png', img)
    IPython.display.display(IPython.display.Image(data=i_img))

class Block_Codec:
    def __init__(self) -> None:
        # quantization table
        self._quant_lumin_table2 = np.array([[16, 155],
                                             [59, 132]])
        self._quant_lumin_table4 = np.array([[13, 15, 137, 157],
                                            [15, 23, 159, 167],
                                            [25, 53, 141, 146],
                                            [69, 90, 109, 131]])
        self._quant_lumin_table8 = np.array([[16, 11, 10, 16, 124, 140, 151, 161],
                                            [12, 12, 14, 19, 126, 158, 160, 155],
                                            [14, 13, 16, 24, 140, 157, 169, 156],
                                            [14, 17, 22, 29, 151, 187, 180, 162],
                                            [18, 22, 37, 56, 168, 109, 103, 177],
                                            [24, 35, 55, 64, 181, 104, 113, 192],
                                            [49, 64, 78, 87, 103, 121, 120, 101],
                                            [72, 92, 95, 98, 112, 100, 103, 199]])
        self._quant_chrom_table2 = np.array([[47, 99],
                                             [99, 99]])
        self._quant_chrom_table4 = np.array([[19, 41, 99, 99],
                                             [41, 88, 99, 99],
                                             [99, 99, 99, 99],
                                             [99, 99, 99, 99]])
        self._quant_chrom_table8 = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                             [18, 21, 26, 66, 99, 99, 99, 99],
                                             [24, 26, 56, 99, 99, 99, 99, 99],
                                             [47, 66, 99, 99, 99, 99, 99, 99],
                                             [99, 99, 99, 99, 99, 99, 99, 99],
                                             [99, 99, 99, 99, 99, 99, 99, 99],
                                             [99, 99, 99, 99, 99, 99, 99, 99],
                                             [99, 99, 99, 99, 99, 99, 99, 99]])
        # img_metadata
        self._block_size = None
        self._img_size = None
        self._is_grayscale = None

        self.pad_width_0 = None
        self.pad_width_1 = None
        self.pad_img_shape_0 = None
        self.pad_img_shape_1 = None
        self.block_num_0 = None
        self.block_num_1 = None
        self.zigzag_idx = None
        # mid products
        self.img = None
        self.subsamp_img = None
        self.pad_img = None
        self.img_blocks = None
        self.dct_img_blocks = None
        self.quant_img_blocks = None
        self.delta_img_blocks = None
        self.rlc_img_blocks = None

        self.compressed_img = None

    def calculate_param(self):
        self.pad_width_0 = self._block_size - self._img_size[0]%self._block_size if self._img_size[0]%self._block_size else 0 
        self.pad_width_1 = self._block_size - self._img_size[1]%self._block_size if self._img_size[1]%self._block_size else 0
        self.pad_img_shape_0 = self._img_size[0] + self.pad_width_0
        self.pad_img_shape_1 = self._img_size[1] + self.pad_width_1
        self.block_num_0 = self.pad_img_shape_0 // self._block_size
        self.block_num_1 = self.pad_img_shape_1 // self._block_size
        self.zigzag_idx = list()
        for d, num in enumerate(list(range(2, self._block_size)) + list(range(self._block_size, 0, -1))):
            if d%2 == 0:
                i = 0
                j = d + 1 - i
                while j >= self._block_size:
                    i += 1
                    j -= 1
                for count in range(num):
                    self.zigzag_idx.append((i+count, j-count))
            else:
                j = 0
                i = d + 1 - j
                while i >= self._block_size:
                    i -= 1
                    j += 1
                for count in range(num):
                    self.zigzag_idx.append((i-count, j+count))
            
    def compress(self, img, block_size=3, mode='4:2:0'):
        """
            Args:
                img: image array
                block_size: divide image by 2**block_size
                mode: chroma sub-sampling mode, one of '4:4:4', '4:4:0', '4:2:2', '4:2:0', '4:1:1'
        """
        self._block_size = max(min(2**block_size, 8), 2)
        self._img_size = img.shape[:2]
        self._is_grayscale = len(img.shape) == 2
        self.calculate_param()
        # compression
        self.img = img
        if not self._is_grayscale:
            self.chroma_subsample(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), mode)
        self.pad_image()
        self.decompose()
        self.DCT()
        self.quantize()
        self.delta_encoder()
        self.RL_encoder()

        # self.compressed_img = self.pad_img # pad_image
        # self.compressed_img = self.img_blocks # decompose
        # self.compressed_img = self.dct_img_blocks # DCT
        # self.compressed_img = self.quant_img_blocks # quantize
        # self.compressed_img = self.delta_img_blocks # delta_encoder
        self.compressed_img = self.rlc_img_blocks # RL_encoder
        return self.compressed_img, (self.img.shape[:2], self._block_size, self._is_grayscale)

    def decompress(self, compressed_img, img_metadata):
        """
            Args:
                img: image array
                img_metadata: metadata of image (img_size, block_size)
        """
        self.compressed_img = compressed_img
        self._img_size, self._block_size, self._is_grayscale= img_metadata
        self.calculate_param()

        # decompression
        self.rlc_img_blocks = compressed_img # RL_decoder
        # self.delta_img_blocks = compressed_img # delta_decoder
        # self.quant_img_blocks = compressed_img # dequantize
        # self.dct_img_blocks = compressed_img # IDCT
        # self.img_blocks = compressed_img # compose
        # self.pad_img = compressed_img # trim_img

        self.RL_decoder()
        self.delta_decoder()
        self.dequantize()
        self.IDCT()
        self.compose()
        self.trim_img()
        if not self._is_grayscale:
            self.img = cv2.cvtColor(self.subsamp_img, cv2.COLOR_YCrCb2BGR)

        return self.img

    def chroma_subsample(self, YCrCb_img, mode):
        def subsample(array):
            sub_samp = array[::mode_map[mode][0], ::mode_map[mode][1]]
            up_samp = np.repeat(np.repeat(sub_samp, mode_map[mode][0],axis=0), mode_map[mode][1],axis=1)
            return up_samp

        mode_map = { '4:4:4': [1, 1],'4:4:0': [1,2], '4:2:2': [2,1],  '4:2:0': [2,2], '4:1:1': [4,1]}
        YCrCb_img[:, :, 1] = subsample(YCrCb_img[:, :, 1])
        YCrCb_img[:, :, 2] = subsample(YCrCb_img[:, :, 2])

        self.subsamp_img = YCrCb_img

    def pad_image(self):
        # pad image
        if self._is_grayscale:
            self.pad_img = np.pad(self.img, ((0, self.pad_width_0), (0, self.pad_width_1)))
        else:
            self.pad_img = np.pad(self.subsamp_img, ((0, self.pad_width_0), (0, self.pad_width_1), (0, 0)))
        self.pad_img = np.array(self.pad_img - 128, dtype=np.int8)
        
    def trim_img(self):
        if self._is_grayscale:
            self.img = np.array(self.pad_img[:self._img_size[0], :self._img_size[1]] + 128, dtype=np.uint8)
        else:
            self.subsamp_img = np.array(self.pad_img[:self._img_size[0], :self._img_size[1], :] + 128, dtype=np.uint8)

    def decompose(self):     
        self.img_blocks = list()
        if self._is_grayscale:
            for i in range(0, self.pad_img.shape[0], self._block_size):
                for j in range(0, self.pad_img.shape[1], self._block_size):
                    self.img_blocks.append(np.array(self.pad_img[i:i+self._block_size, j:j+self._block_size]))
        else:
            for c in range(3):
                for i in range(0, self.pad_img.shape[0], self._block_size):
                    for j in range(0, self.pad_img.shape[1], self._block_size):
                        self.img_blocks.append(np.array(self.pad_img[i:i+self._block_size, j:j+self._block_size, c]))
        self.img_blocks = np.array(self.img_blocks)

    def compose(self):
        count = 0
        if self._is_grayscale:            
            self.pad_img = np.zeros((self.pad_img_shape_0, self.pad_img_shape_1), dtype=np.int8)
            for i in range(self.block_num_0):
                for j in range(self.block_num_1):
                    self.pad_img[i*self._block_size:(i+1)*self._block_size, j*self._block_size:(j+1)*self._block_size] = self.img_blocks[count]
                    count += 1
        else:
            self.pad_img = np.zeros((self.pad_img_shape_0, self.pad_img_shape_1, 3), dtype=np.int8)
            for c in range(3):
                for i in range(self.block_num_0):
                    for j in range(self.block_num_1):
                        self.pad_img[i*self._block_size:(i+1)*self._block_size, j*self._block_size:(j+1)*self._block_size, c] = self.img_blocks[count]
                        count += 1

    def DCT(self):
        def dct2D(array):
            trans_array = 2 / self._block_size * np.ones(array.shape, dtype=np.float32)
            for u in range(self._block_size):
                for v in range(self._block_size):
                    C_u = 2 ** (-0.5) if u == 0 else 1.0
                    C_v = 2 ** (-0.5) if v == 0 else 1.0
                    cos_x = np.cos((np.array(range(self._block_size), dtype=np.float32) + 0.5) * u * np.pi / self._block_size)
                    cos_x = cos_x.reshape((array.shape[0], 1))
                    cos_y = np.cos((np.array(range(self._block_size), dtype=np.float32) + 0.5) * v * np.pi / self._block_size)
                    cos_y = cos_y.reshape((1, array.shape[1]))
                    trans_array[u][v] *= np.sum(array * C_u * C_v * cos_x * cos_y)

            return trans_array

        self.dct_img_blocks = np.zeros(self.img_blocks.shape, dtype=np.float32)
        for count in range(self.img_blocks.shape[0]):
            self.dct_img_blocks[count, :, :] = dct2D(self.img_blocks[count, :, :])

    def IDCT(self):
        def idct2D(array):
            trans_array = 2 / self._block_size * np.ones(array.shape, dtype=np.float32)
            for x in range(self._block_size):
                for y in range(self._block_size):
                    C_u = np.ones((self._block_size, 1), dtype=np.float32)
                    C_u[0][0] = 2 ** (-0.5)
                    C_v = np.ones((1, self._block_size), dtype=np.float32)
                    C_v[0][0] = 2 ** (-0.5)
                    cos_u = np.cos((x + 0.5) * np.array(range(self._block_size), dtype=np.float32) * np.pi / self._block_size)
                    cos_u = cos_u.reshape((self._block_size, 1))
                    cos_v = np.cos((y + 0.5) * np.array(range(self._block_size), dtype=np.float32) * np.pi / self._block_size)
                    cos_v = cos_v.reshape((1, self._block_size))
                    trans_array[x][y] *= np.sum(array * C_u * C_v * cos_u * cos_v)

            return trans_array

        self.img_blocks = np.zeros(self.dct_img_blocks.shape, dtype=np.float32)
        for count in range(self.dct_img_blocks.shape[0]):
            self.img_blocks[count, :, :] = idct2D(self.dct_img_blocks[count, :, :])

    def quantize(self):
        if self._block_size == 2:
            quant_lumin_table = np.expand_dims(self._quant_lumin_table2, axis=0)
            quant_chrom_table = np.expand_dims(self._quant_chrom_table2, axis=0)
        elif self._block_size == 4:
            quant_lumin_table = np.expand_dims(self._quant_lumin_table4, axis=0)
            quant_chrom_table = np.expand_dims(self._quant_chrom_table4, axis=0)
        else:
            quant_lumin_table = np.expand_dims(self._quant_lumin_table8, axis=0)
            quant_chrom_table = np.expand_dims(self._quant_chrom_table8, axis=0)
        
        self.quant_img_blocks = np.zeros(self.dct_img_blocks.shape, dtype=np.int16)
        self.quant_img_blocks[0:self.block_num_0*self.block_num_1, :, :] = np.around(self.dct_img_blocks[0:self.block_num_0*self.block_num_1, :, :] / quant_lumin_table)
        if not self._is_grayscale:
            self.quant_img_blocks[self.block_num_0*self.block_num_1:, :, :] = np.around(self.dct_img_blocks[self.block_num_0*self.block_num_1:, :, :] / quant_chrom_table)
        
    def dequantize(self):
        if self._block_size == 2:
            quant_lumin_table = self._quant_lumin_table2
            quant_chrom_table = self._quant_chrom_table2
        elif self._block_size == 4:
            quant_lumin_table = self._quant_lumin_table4
            quant_chrom_table = self._quant_chrom_table4
        else:
            quant_lumin_table = self._quant_lumin_table8
            quant_chrom_table = self._quant_chrom_table8
        self.dct_img_blocks = np.zeros(self.quant_img_blocks.shape, dtype=np.int16)
        self.dct_img_blocks[0:self.block_num_0*self.block_num_1, :, :] = self.quant_img_blocks[0:self.block_num_0*self.block_num_1, :, :] * quant_lumin_table
        if not self._is_grayscale:
            self.dct_img_blocks[self.block_num_0*self.block_num_1:, :, :] = self.quant_img_blocks[self.block_num_0*self.block_num_1:, :, :] * quant_chrom_table
        
    def delta_encoder(self):
        self.delta_img_blocks = np.array(self.quant_img_blocks)
        self.delta_img_blocks[1:, 0, 0] = self.delta_img_blocks[1:, 0, 0] - self.delta_img_blocks[:-1, 0, 0]

    def delta_decoder(self):
        self.quant_img_blocks = np.array(self.delta_img_blocks, dtype=np.int16)
        for count in range(1, self.quant_img_blocks.shape[0]):
            self.quant_img_blocks[count][0][0] = self.quant_img_blocks[count][0][0] + self.quant_img_blocks[count-1][0][0]

    def RL_encoder(self):
        def encoder(block):
            block_container = list()
            get_size = lambda num: math.ceil(math.log2(abs(num) + 1))
            # DC term
            block_container.append(get_size(block[0][0]))
            block_container.append(block[0][0])
            # AC terms
            run = 0
            for k in range(self._block_size**2 - 1):
                i, j = self.zigzag_idx[k]
                if block[i][j] == 0:
                    run += 1
                else:
                    size = get_size(block[i][j])
                    while run >= 16:
                        block_container.append(15)
                        block_container.append(0)
                        run -= 16
                    block_container.append(run)
                    block_container.append(size)
                    block_container.append(block[i][j])
                    run = 0
            block_container.append(0)
            block_container.append(0)
            return block_container
            
        self.rlc_img_blocks = list()
        for i in range(self.delta_img_blocks.shape[0]):
            self.rlc_img_blocks.append(encoder(self.delta_img_blocks[i, :, :]))

    def RL_decoder(self):
        def decoder(block_container):
            block = np.zeros((self._block_size, self._block_size), dtype=np.int16)
            # DC term
            _ = block_container.pop(0)
            block[0][0] = block_container.pop(0)
            # AC terms
            count = 0
            while count < self._block_size**2 - 1:
                run = block_container.pop(0)
                size = block_container.pop(0)
                if run == 15 and size == 0:
                    count += 16
                elif run == 0 and size == 0:
                    break
                else:
                    count += run
                    i, j = self.zigzag_idx[count]
                    block[i][j] = block_container.pop(0)
            return block

        self.delta_img_blocks = list()
        for container in self.rlc_img_blocks:
            self.delta_img_blocks.append(decoder(container))

        self.delta_img_blocks = np.array(self.delta_img_blocks, dtype=np.uint16)


def RMS(img, lossy_img):
    term_num = np.prod(img.shape)
    result = (np.sum((img - lossy_img)**2) / term_num)**(0.5)
    return result

def SNR(img, lossy_img):
    numerator = np.sum(lossy_img**2)
    denominator = np.sum((img - lossy_img)**2)
    return numerator / denominator

# def chroma_subsampling(img, mode='4:2:0', cvt_bg=False):
#     """
#         Perform chroma subsampling

#         Args:
#             img: BGR image array
#             mode: one of the following str - '4:4:4', '4:4:0', '4:2:2', '4:2:0', '4:1:1'
#             cvt_bg: convert to BGR color space
#     """
#     def samp(array):
#         sub_samp = array[::mode_map[mode][0], ::mode_map[mode][1]]
#         up_samp = np.repeat(np.repeat(sub_samp, mode_map[mode][0],axis=0), mode_map[mode][1],axis=1)
#         return up_samp

#     mode_map = { '4:4:4': [1, 1],'4:4:0': [1,2], '4:2:2': [2,1],  '4:2:0': [2,2], '4:1:1': [4,1]}
#     YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     YCrCb_img[:, :, 1] = samp(YCrCb_img[:, :, 1])
#     YCrCb_img[:, :, 2] = samp(YCrCb_img[:, :, 2])
    
#     sub_samp_img = cv2.cvtColor(YCrCb_img, cv2.COLOR_YCrCb2BGR) if cvt_bg else YCrCb_img

#     return sub_samp_img