from dataclasses import dataclass
import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA = 'data'
SENSOR = 'sensor'
MAPPING = 'mapping'
IMAGE_WIDTH = 'image_width'
IMAGE_HEIGHT = 'image_height'
EOP = 2  # End-of-pixel


@dataclass
class LaserBeam:
    sensor: np.array
    mapping: np.array
    image_width: int
    image_height: int
    file_name: str = 'file_name'

    @staticmethod
    def validate_laser_beam_obj(lb):
        assert lb.image_height > 0
        assert lb.image_height > 0
        assert np.all(lb.sensor >= 0)
        assert np.all(lb.mapping >= 0)

    @classmethod
    def from_hdf(cls, path):
        with h5py.File(path, 'r') as f:
            sensor = f[DATA][SENSOR][:]
            mapping = f[DATA][MAPPING][:]
            image_width = f[DATA].attrs[IMAGE_WIDTH]
            image_height = f[DATA].attrs[IMAGE_HEIGHT]

        obj = cls(sensor, mapping, image_width, image_height, path)
        LaserBeam.validate_laser_beam_obj(obj)

        return obj

    @property
    def clean_sensor(self):
        return self.sensor[self.mapping > 0]

    @property
    def clean_mapping(self):
        return self.mapping[self.mapping > 0]

    @property
    def pixels(self):
        cumsum = self.clean_sensor.cumsum()[self.clean_mapping == EOP]
        return np.diff(cumsum, prepend=0)

    @property
    def pixel_matrix(self):
        assert len(self.pixels) % self.image_width == 0

        return self.pixels.reshape([self.image_height, self.image_width])

    def draw_image(self):
        plt.imshow(self.pixel_matrix)
        plt.colorbar()
        plt.show()

    def save_image(self, file_name=None):
        file_name = file_name or self.file_name+'.png'
        plt.imsave(file_name, self.pixel_matrix)

    @staticmethod
    def fit_gaussian(y):
        x = np.arange(len(y))
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

        return mean, sigma


if __name__ == '__main__':
    path1 = 'data/data.hdf5'
    path2 = 'data/data2.hdf5'

    x = LaserBeam.from_hdf(path1)
    y = LaserBeam.from_hdf(path2)

    x.save_image()
    y.draw_image()

    row = len(x.pixel_matrix[10]) // 2 + 1
    mean, sigma = LaserBeam.fit_gaussian(x.pixel_matrix[row])
    print(mean, sigma)
