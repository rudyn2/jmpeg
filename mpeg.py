from jpeg import JEPG
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import cv2
import h5py
import numpy as np
import click


class MPEG(object):
    def __init__(self, vid: np.array):
        self.vid = vid.transpose(0, 3, 2, 1)

    @classmethod
    def reduction_fn(cls, frame_1: np.array, frame_2: np.array, offset: np.array):
        result = (frame_1 + frame_2) / 2.0 - offset
        return result

    def forward(self, fn):

        result = []
        frame_idx = 0
        while frame_idx + 8 < self.vid.shape[0]:

            actual_gop_i_frame = self.vid[frame_idx, :, :, :]
            next_gop_i_frame = self.vid[frame_idx + 8, :, :, :]

            b_4 = fn(actual_gop_i_frame, next_gop_i_frame, self.vid[frame_idx + 4, :, :, :])
            b_2 = fn(actual_gop_i_frame,  b_4, self.vid[frame_idx + 2, :, :, :])
            b_1 = fn(actual_gop_i_frame, b_2, self.vid[frame_idx + 1, :, :, :])
            b_3 = fn(b_2, b_4, self.vid[frame_idx + 3, :, :, :])
            b_6 = fn(b_4, next_gop_i_frame, self.vid[frame_idx + 6, :, :, :])
            b_5 = fn(b_4, b_6, self.vid[frame_idx + 5, :, :, :])
            b_7 = fn(b_6, next_gop_i_frame, self.vid[frame_idx + 7, :, :, :])

            result.extend([actual_gop_i_frame, b_1, b_2, b_3, b_4, b_5, b_6, b_7])
            frame_idx += 8

        # add last frames as it
        if frame_idx < self.vid.shape[0]:
            result.extend([self.vid[idx, :, :, :] for idx in range(frame_idx, self.vid.shape[0])])

        return result

    def compress(self, n_drop: int):
        gops = self.forward(fn=self.reduction_fn)
        num_cores = multiprocessing.cpu_count()
        parallel = Parallel(n_jobs=num_cores)
        print(f"Compression using {num_cores} cores")

        # 1 frame per core
        inputs = tqdm(gops, "Compressing")
        outputs = parallel(delayed(JEPG.compress)(i, int(n_drop)) for i in inputs)
        print("Compression ready")
        return outputs

    @classmethod
    def backward(cls, compressed_video, fn):
        """
        Forma las reconstrucciones usando reglas definidas para GOP IBx7.

        :param compressed_video:
            Lista de frames comprimidos usando JPEG.
        :param fn:
            Función de reducción de redundancia.
        :return:
            Numpy array con frames de video descomprimido con dimensiones [n_frames, n_channels, width, height].
        """

        org_vid = []
        frame_idx = 0
        while frame_idx + 8 < len(compressed_video):
            actual_gop_i_frame = compressed_video[frame_idx]
            next_gop_i_frame = compressed_video[frame_idx + 8]
            b_1, b_2, b_3, b_4, b_5, b_6, b_7 = compressed_video[frame_idx:frame_idx+7]

            c_4 = fn(actual_gop_i_frame, next_gop_i_frame, b_4)
            c_2 = fn(actual_gop_i_frame, b_4,  b_2)
            c_1 = fn(actual_gop_i_frame, b_4, b_1)
            c_3 = fn(b_2, b_4, b_3)
            c_6 = fn(b_4, next_gop_i_frame, b_6)
            c_5 = fn(b_4, b_6, b_5)
            c_7 = fn(b_6, next_gop_i_frame, b_7)

            org_vid.extend([actual_gop_i_frame, c_1, c_2, c_3, c_4, c_5, c_6, c_7])
            frame_idx += 8

        # add last frames as it
        if frame_idx < len(compressed_video):
            org_vid.extend(compressed_video[frame_idx:])

        org_vid = np.stack(org_vid)
        return org_vid.transpose((0, 3, 2, 1))

    @classmethod
    def decompress(cls, compressed_video):
        """
        Descomprime video comprimido usando MPEG-2.

        :param compressed_video:
            Lista de Frames comprimidos usando patrón IBx7.
        :return:
            Lista de Frames descomprimidos.
        """

        num_cores = multiprocessing.cpu_count()
        parallel = Parallel(n_jobs=num_cores)
        print(f"Decompression using {num_cores} cores")

        # 1 frame per core
        inputs = tqdm(compressed_video, "Decompression")
        outputs = parallel(delayed(JEPG.decompress)(i) for i in inputs)
        print("Decompression ready")
        outputs = cls.backward(outputs, fn=cls.reduction_fn)
        return outputs

    @classmethod
    def visualize(cls, name: str, vid: np.array):
        """
        Guarda el video en formato avi usando un array de frames.

        :param name:                Nombre del video de salida con extensión .avi.
        :param vid:                 Array de entrada de dimensiones [n_frames, n_channels, width, height].
        """

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(name, fourcc, 20.0, (1920, 1080))

        for frame_idx in tqdm(range(vid.shape[0]), "Writing video"):
            frame = vid[frame_idx, :, :, :]
            frame = frame.transpose(2, 1, 0)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()


@click.command()
@click.option('--ndrop', help='Número de bytes por bloque a eliminar.')
@click.option('--ipath', help='Dirección local a archivo (.mat o .npy).')
@click.option('--opath', help='Dirección de almacenamiento de video procesado.')
def execute(ndrop, ipath, opath):
    """
    Compress and decompress.
    """

    if ipath.split(".")[-1] == "mat":
        arrays = {}
        f = h5py.File(ipath, 'r')
        for k, v in f.items():
            arrays[k] = np.array(v)
        video = arrays['lago']
    elif ipath.split(".")[-1] == "npy":
        video = np.load(str(ipath))
    else:
        raise ValueError("Solo arrays de videos en formato .mat o .npy.")

    half = 120
    v = MPEG(video[:half, :, :, :])
    compressed_v = v.compress(ndrop)
    decompressed_v = v.decompress(compressed_v)
    MPEG.visualize(str(opath), decompressed_v)


if __name__ == '__main__':
    execute()
    # video = np.load('full_video.npy')
    # ndrop = 32
    # opath = 'full.avi'
    #
    # short_video = video[:120, :, :, :]
    # v = MPEG(short_video)
    # compressed_v = v.compress(ndrop)
    # decompressed_v = v.decompress(compressed_v)
    # MPEG.visualize(str(opath), short_video)