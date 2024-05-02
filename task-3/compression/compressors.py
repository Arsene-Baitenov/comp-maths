import math
import os
import time
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from compression.svd_types import Svd, ImgCompressed, METADATA_SIZE, FLOAT_SIZE


class SvdCompressor(ABC):

    @abstractmethod
    def _compress_channel(self, channel: np.ndarray, k: int) -> Svd:
        ...

    def _compress(self, img_array: np.ndarray, k: int) -> tuple[Svd, ...]:
        return tuple(self._compress_channel(img_array[..., i], k) for i in range(3))

    def to_svd(self, file_path: str, ratio: float) -> ImgCompressed:
        img_size = os.path.getsize(file_path)

        img = Image.open(file_path)
        height = img.height
        width = img.width
        k = math.floor(((img_size / ratio) - METADATA_SIZE) / (FLOAT_SIZE * 3 * (height + width + 1)))

        img_arrays = np.asarray(img)
        r, g, b = self._compress(img_arrays, k)

        return ImgCompressed(height, width, k, r, g, b)

    @staticmethod
    def from_svd(img: ImgCompressed) -> Image:
        unpacked_arrays = [img.red.to_matrix(), img.green.to_matrix(), img.blue.to_matrix()]
        image_matrix = np.dstack(unpacked_arrays).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image_matrix)


class WithNumpySvdCompressor(SvdCompressor, ABC):
    def _compress_channel(self, channel: np.ndarray, k: int):
        u, s, vh = np.linalg.svd(channel, full_matrices=False)
        return Svd(u[:, :k], s[:k], vh[:k, :])


class PowerSvdCompressor(SvdCompressor, ABC):
    def __init__(self, duration: float):
        self.duration = duration
        self.epsilon = 1e-8

    def _power_svd(self, channel: np.ndarray, duration: float):
        time_end = time.time() * 1000 + duration
        mu, sigma = 0, 1
        x = np.random.normal(mu, sigma, size=channel.shape[1])
        b = channel.T.dot(channel)
        while True:
            new_x = b.dot(x)
            if np.allclose(new_x, x, self.epsilon):
                break
            x = new_x / np.linalg.norm(new_x)
            if time.time() * 1000 >= time_end:
                break

        v = x / np.linalg.norm(x)
        sigma = np.linalg.norm(channel.dot(v))
        u = channel.dot(v) / sigma
        return np.reshape(u, (channel.shape[0], 1)), sigma, np.reshape(v, (channel.shape[1], 1))

    def _compress_channel(self, channel: np.ndarray, k: int):
        rank = np.linalg.matrix_rank(channel)
        ut = np.zeros((channel.shape[0], 1))
        st = []
        vht = np.zeros((channel.shape[1], 1))

        single_duration = self.duration / rank

        for i in range(rank):
            u, sigma, v = self._power_svd(channel, single_duration)
            ut = np.hstack((ut, u))
            st.append(sigma)
            vht = np.hstack((vht, v))
            channel = channel - u.dot(v.T).dot(sigma)

        ut = ut[:, 1:]
        vht = vht[:, 1:]
        return Svd(ut[:, :k], np.array(st)[:k], vht.T[:k, :])


class BlockPowerSvdCompressor(SvdCompressor, ABC):
    def __init__(self, duration: float):
        self.duration = duration
        self.epsilon = 1e-8

    def _compress_channel(self, channel: np.ndarray, k: int):
        u = np.zeros((channel.shape[0], k))
        s = np.zeros(k)
        vh = np.zeros((channel.shape[1], k))

        time_end = time.time() * 1000 + self.duration
        while time.time() * 1000 < time_end:
            q, _ = np.linalg.qr(np.dot(channel, vh))
            u = q[:, :k]

            q, r = np.linalg.qr(np.dot(channel.T, u))
            vh = q[:, :k]
            s = r[:k, :k]
            if np.allclose(np.dot(channel, vh), np.dot(u, r[:k, :k]), self.epsilon):
                break

        return Svd(u, np.diag(s).astype(np.float32), vh.T)
