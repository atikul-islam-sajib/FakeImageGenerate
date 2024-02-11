import sys
import unittest
import joblib as pickle

sys.path.append("src/")

from utils import total_params
from generator import Generator
from discriminator import Discriminator


class DataLoaderTest(unittest.TestCase):
    def setUp(self):
        # This method will be called before each test method
        self.dataloader = pickle.load("./data/processed/dataloader.pkl")
        self.total_data = 0

    def test_dataloader(self):
        # Iterate through the dataloader and count the total number of samples
        for data, label in self.dataloader:
            self.total_data += data.shape[0]

        # Assert that the total number of samples is what you expect
        self.assertEqual(self.total_data, 63565)


if __name__ == "__main__":
    unittest.main()

# =======================================================================================#


class DisTest(unittest.TestCase):

    def setUp(self):
        self.image_size = 64
        self.discriminator = Discriminator(image_size=self.image_size)

    def test_total_params(self):
        self.assertEqual(total_params(model=self.discriminator), 2765568)


if __name__ == "__main__":
    unittest.main()


# =======================================================================================#


class GenTest(unittest.TestCase):

    def setUp(self):
        self.image_size = 64
        self.generator = Generator(image_size=self.image_size)

    def test_total_params(self):
        self.assertEqual(total_params(model=self.generator), 3576704)


if __name__ == "__main__":
    unittest.main()
