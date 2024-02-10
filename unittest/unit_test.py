import unittest
import joblib as pickle

# Assuming your utils and the necessary imports are correctly set up.


class DataLoaderTest(
    unittest.TestCase
):  # Renamed class to avoid confusion with PyTorch DataLoader
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
