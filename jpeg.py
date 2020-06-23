import cv2 as cv
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import click
import cv2

class JEPG(object):
    T_YUV = np.array([[0.299, 0.587, 0.114],
                      [-0.1687, -0.3313, 0.5],
                      [0.5, -0.4187, -0.0813]])
    T_RGB = np.array([[1, 0, 1.402],
                      [1, -0.34414, -0.71414],
                      [1, 1.772, 0]])
    Y_POND = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])
    UV_POND = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                        [18, 21, 26, 66, 99, 99, 99, 99],
                        [24, 26, 56, 99, 99, 99, 99, 99],
                        [47, 66, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99],
                        [99, 99, 99, 99, 99, 99, 99, 99]])

    @classmethod
    def rgb2yuv(cls, rgb_input: np.array) -> np.array:
        """
        Transforma de RGB a YUV.
        """
        yuv = np.tensordot(rgb_input, cls.T_YUV, axes=[-1, 1])
        yuv[:, :, 1:] += 0.5
        return yuv

    @classmethod
    def dct2(cls, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    @classmethod
    def idct2(cls, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    @classmethod
    def yuv2rgb(cls, yuv_input: np.array) -> np.array:
        yuv_input[:, :, 1] -= 0.5
        yuv_input[:, :, 2] -= 0.5
        rgb = np.tensordot(yuv_input, cls.T_RGB, axes=[-1, 1])
        return np.clip(rgb, 0, 255).astype('uint8')

    @classmethod
    def dct_block(cls, yuv_channel: np.array, pond: np.array):
        """
        Aplica DCT, Discretización y aproximación entera por bloques a canal YUV.
        """
        i = 0
        j = 0
        blocks = []
        while i < yuv_channel.shape[0]:
            j_blocks = []
            while j < yuv_channel.shape[1]:
                block = yuv_channel[i:i + 8, j:j + 8]
                dct_b = cls.dct2(block)
                dct_b = np.divide(dct_b, pond)
                dct_b = np.round(dct_b)
                j_blocks.append(dct_b)
                j += 8
            blocks.append(j_blocks)
            j = 0
            i += 8
        return blocks

    @classmethod
    def dct_op(cls, yuv_img: np.array):
        """
        Aplica transformación DCT por bloque, canal por canal.
        """
        y = cls.dct_block(yuv_img[:, :, 0], cls.Y_POND)
        u = cls.dct_block(yuv_img[:, :, 1], cls.UV_POND)
        v = cls.dct_block(yuv_img[:, :, 2], cls.UV_POND)
        return [y, u, v]

    @classmethod
    def zigzag_flatten(cls, block: np.array, n_drop: int) -> np.array:
        """
        Aplana array usando patron zig-zag y elimina ultimos {n_drop} valores.
        """
        rows = block.shape[0]
        columns = block.shape[1]
        zig_zag_vals = [[] for _ in range(rows + columns - 1)]
        for i in range(rows):
            for j in range(columns):
                sum = i + j
                if sum % 2 == 0:
                    # add at beginning
                    zig_zag_vals[sum].insert(0, block[i][j])
                else:
                    # add at end of the list
                    zig_zag_vals[sum].append(block[i][j])

        # flattening sol
        flatten_vector = []
        for i in zig_zag_vals:
            for j in i:
                flatten_vector.append(j)

        flatten_vector = np.array(flatten_vector)
        flatten_vector[-n_drop:] = 0
        return flatten_vector

    @classmethod
    def compress(cls, rgb_img: np.array, n_drop: int = 32):
        """
        Comprime una imagen RGB usando algoritmo JPEG.
        """
        yuv = cls.rgb2yuv(rgb_img)
        dct_ = cls.dct_op(yuv)
        for channel_blocks in dct_:
            for i, i_blocks in enumerate(channel_blocks):
                for j, j_blocks in enumerate(i_blocks):
                    channel_blocks[i][j] = cls.zigzag_flatten(channel_blocks[i][j], n_drop)
        return dct_

    @classmethod
    def decompress(cls, img_compressed) -> np.array:
        channels_decompressed = []
        for channel, tag in zip(img_compressed, ['Y', 'U', 'V']):
            channels_decompressed.append(cls.decompress_channel(channel, tag))
        yuv_decompressed = np.stack(channels_decompressed, axis=-1)
        return cls.yuv2rgb(yuv_decompressed)

    @classmethod
    def decompress_channel(cls, channel: list, tag: str):
        output = []
        for i_blocks in channel:
            i_row = []
            for j_block in i_blocks:
                org_block = cls.reverse_zigzag_flatten(j_block)
                if tag == 'Y':
                    org_block = np.multiply(org_block, cls.Y_POND)
                else:
                    org_block = np.multiply(org_block, cls.UV_POND)
                org_block = cls.idct2(org_block)
                i_row.append(org_block)
            output.append(np.hstack(i_row))
        return np.vstack(output)

    @classmethod
    def reverse_zigzag_flatten(cls, arr: np.array) -> np.array:
        """
        Transforma un array en una matriz iterando en forma de zigzag.
        """

        # infers matrix shape assuming that output array is squared
        output_shape = int(np.sqrt(arr.shape[0]))
        # speed up memory allocation
        output = [[None] * output_shape for _ in range(output_shape)]

        zig_zag_idxs = [[] for _ in range(2 * output_shape - 1)]
        for i in range(output_shape):
            for j in range(output_shape):
                sum = i + j
                if sum % 2 == 0:
                    # add at beginning
                    zig_zag_idxs[sum].insert(0, (i, j))
                else:
                    # add at end of the list
                    zig_zag_idxs[sum].append((i, j))

        flatten_idxs = []
        for i in zig_zag_idxs:
            for j in i:
                flatten_idxs.append(j)

        for val, (i, j) in zip(arr, flatten_idxs):
            output[i][j] = val

        return np.array(output)


@click.command()
@click.option('--ndrop', help='')
@click.option('--ipath', help='')
@click.option('--opath', help='')
def execute(ndrop, ipath, opath):
    image = cv.imread(ipath)

    print("Compressing...")
    image_compressed = JEPG.compress(image, int(ndrop))
    print("Decompressing...")
    image_recovered = JEPG.decompress(image_compressed)
    print("Saving...")
    im = Image.fromarray(cv2.cvtColor(image_recovered, cv2.COLOR_RGB2BGR))
    im.save(opath)
    print("Ready!")


if __name__ == '__main__':
    execute()

