import numpy as np
import cv2
from PIL import Image
from io import BytesIO

class DefocusBlur(object):
  def __init__(self, severity=1):
    self.name = "Defocus Blur"
    self.severity = severity

  def __call__(self, x):
    # print(self.name)
    return DefocusBlur.defocus_blur(x, self.severity)

  @staticmethod
  def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
      L = np.arange(-8, 8 + 1)
      ksize = (3, 3)
    else:
      L = np.arange(-radius, radius + 1)
      ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

  @staticmethod
  def defocus_blur(x, severity=1):
    c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

    x = np.array(x) / 255.
    kernel = DefocusBlur.disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1)

class GaussianNoise(object):
  def __init__(self, severity=1):
    self.name = "Gaussian Noise"
    self.severity = severity

  def __call__(self, x):
    # print(self.name)
    return GaussianNoise.gaussian_noise(x, self.severity)

  @staticmethod
  def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)

class JpegCompression(object):
  def __init__(self, severity=1):
    self.name = "JPEG Compression"
    self.severity = severity

  def __call__(self, x):
    # print(self.name)
    return JpegCompression.jpeg_compression(x, self.severity)

  @staticmethod
  def jpeg_compression(x, severity=1):
    c = [80, 65, 58, 50, 40][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)

    return x

class ShotNoise(object):
    def __init__(self, severity):
        self.name = "Shot Noise"
        self.severity = severity

    def __call__(self, x):
        # print(self.name)
        return ShotNoise.shot_noise(x, self.severity)

    @staticmethod
    def shot_noise(x, severity=1):
        c = [500, 250, 100, 75, 50][severity - 1]

        x = np.array(x) / 255.
        return np.clip(np.random.poisson(x * c) / c, 0, 1)

class SpeckleNoise(object):
  def __init__(self, severity):
    self.name = "Speckle Noise"
    self.severity = severity

  def __call__(self, x):
    # print(self.name)
    return SpeckleNoise.speckle_noise(x, self.severity)

  @staticmethod
  def speckle_noise(x, severity=1):
    c = [.06, .1, .12, .16, .2][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1)
