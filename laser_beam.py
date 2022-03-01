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

    @staticmethod
    def validate_laser_beam_obj(lb):
        assert lb.image_height > 0
        assert lb.image_height > 0
        assert np.all(lb.sensor >= 0)
        assert np.all(lb.mapping >= 0)

    @classmethod
    def from_hdf(cls, path, pixel_size=3):
        with h5py.File(path, 'r') as f:
            sensor = f[DATA][SENSOR][:]
            mapping = f[DATA][MAPPING][:]
            image_width = f[DATA].attrs[IMAGE_WIDTH]
            image_height = f[DATA].attrs[IMAGE_HEIGHT]

        obj = cls(sensor, mapping, image_width, image_height)
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
        return np.diff(cumsum, prepend=cumsum[0])

    @property
    def pixel_matrix(self):
        assert len(self.pixels) % self.image_width == 0

        return self.pixels.reshape([self.image_width, self.image_height])

    def draw_image(self):
        plt.imshow(self.pixel_matrix)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    path1 = 'data/data.hdf5'
    path2 = 'data/data2.hdf5'

    x = LaserBeam.from_hdf(path1)
    y = LaserBeam.from_hdf(path2)

    x.draw_image()
    y.draw_image()
