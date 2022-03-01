import unittest
import sys
sys.path.append('../')

import numpy as np
import laser_beam as lb


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.lb = lb.LaserBeam(
            sensor=np.array([1, 2, 2, 3,
                             3, 0, 1, 5,
                             3,
                             2, 1, 1, 1]),
            mapping=np.array([1, 2, 1, 2,
                              1, 2, 1, 2,
                              0,
                              1, 2, 1, 2]),
            image_width=2,
            image_height=3
        )

    def test_clean_sensor(self):
        self.assertTrue(np.array_equal(self.lb.clean_sensor, [1, 2, 2, 3,
                                                              3, 0, 1, 5,
                                                              2, 1, 1, 1]))

    def test_clean_mapping(self):
        self.assertTrue(np.array_equal(self.lb.clean_mapping, [1, 2, 1, 2,
                                                               1, 2, 1, 2,
                                                               1, 2, 1, 2]))

    def test_pixels(self):
        self.assertTrue(np.array_equal(self.lb.pixels, [3, 5, 3, 6, 3, 2]),
                        self.lb.pixels)

    def test_pixel_matrix(self):
        self.assertTrue(np.array_equal(self.lb.pixel_matrix, [[3, 5],
                                                              [3, 6],
                                                              [3, 2]]))


if __name__ == '__main__':
    unittest.main()
