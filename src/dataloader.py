import sys
import argparse
import logging
import os
import zipfile
import joblib as pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append("src/")

from utils import create_pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s",
    filemode="w",
    filename="./logs/dataloader.log",
)


class Loader:
    """
    A Loader class for processing image datasets. This class is designed to handle the loading,
    extracting, and preprocessing of image data from a zip archive, preparing it for machine learning models.

    Attributes:
        image_path (str, optional): The path to the zip file containing the dataset. Default is None.
        batch_size (int): The number of images to process in each batch. Default is 64.
        image_height (int): The height to which each image will be resized. Default is 64 pixels.
        image_width (int): The width to which each image will be resized. Default is 64 pixels.
        normalized (bool): Flag to determine whether the images should be normalized. Default is True.
        raw_image_path (str): The path where extracted images are stored. Initially empty.
    """

    def __init__(
        self,
        image_path=None,
        batch_size=64,
        image_height=64,
        image_width=64,
        normalized=True,
    ):
        """
        Initializes the Loader with the dataset path, batch size, image dimensions, and normalization flag.
        """
        self.image_path = image_path
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.normalized = normalized
        self.raw_image_path = ""

    def unzip_dataset(self, extract_to=None):
        """
        Extracts the dataset from a zip archive to a specified directory.

        Parameters:
            extract_to (str): The directory where the zip file contents will be extracted. If None, an exception is raised.

        Raises:
            Exception: If `extract_to` is None, indicating the path is not properly defined.
        """
        if extract_to is not None:
            with zipfile.ZipFile(file=self.image_path, mode="r") as zip_ref:
                zip_ref.extractall(path=extract_to)
        else:
            raise Exception(
                "Path is not defined properly in unzip_dataset method".capitalize()
            )

    def extract_features(
        self,
    ):
        """
        Prepares the dataset by checking for or creating the necessary directories and unzipping the dataset.

        This method checks if a raw folder exists for the dataset; if not, it creates one and extracts the dataset there.
        """
        dataset_folder_name = "./data"
        extract_to = os.path.join(dataset_folder_name, "raw/")
        if os.path.exists(path=os.path.join(dataset_folder_name, "raw/")):
            logging.info("raw folder already exists".title())
            try:
                self.unzip_dataset(extract_to=extract_to)
            except Exception as e:
                print("Error - {}".format(e))
            else:
                self.raw_image_path = os.path.join(dataset_folder_name, "raw/")
        else:
            logging.info("raw folder does not exists and is about to create".title())
            try:
                os.makedirs(os.path.join(dataset_folder_name, "raw/"))
            except Exception as e:
                print("Error - {}".format(e))
            else:
                self.unzip_dataset(extract_to=extract_to)
                self.raw_image_path = os.path.join(dataset_folder_name, "raw/")

    def saved_dataloader(self, dataloader=None):
        """
        Saves the processed dataloader object to disk.

        Parameters:
            dataloader: The dataloader object to be saved. If None, an exception is raised.

        Raises:
            Exception: If `dataloader` is None, indicating it is not properly defined.
        """
        if dataloader is not None:
            processed_data_path = "./data"
            if os.path.exists(os.path.join(processed_data_path, "processed")):
                try:
                    create_pickle(
                        value=dataloader,
                        filename=os.path.join(
                            processed_data_path, "processed/dataloader.pkl"
                        ),
                    )
                    logging.info("done to create pickle file".title())
                except Exception as e:
                    print("Error - {}".format(e))
            else:
                logging.info(
                    "Processed data folder is not exists and is about to create".capitalize()
                )
                os.makedirs(os.path.join(processed_data_path, "processed"))
                try:
                    create_pickle(
                        value=dataloader,
                        filename=os.path.join(
                            processed_data_path, "processed/dataloader.pkl"
                        ),
                    )
                except Exception as e:
                    print("Error - {}".format(e))
        else:
            raise Exception(
                "Dataloader is not defined properly in saved_dataloader method".capitalize()
            )

    def create_dataloader(self):
        """
        Creates a dataloader with the specified transformations and batch size for the dataset.

        Returns:
            DataLoader: The DataLoader object for the dataset, ready for use in training or evaluation.
        """
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_height, self.image_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.ImageFolder(root=self.raw_image_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.saved_dataloader(dataloader=dataloader)

        return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Preprocessing".capitalize())

    parser.add_argument("--image_path", help="Define the Image path".capitalize())
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Define the Batch Size".capitalize()
    )
    parser.add_argument(
        "--image_height",
        default=64,
        type=int,
        help="Define the Image Height".capitalize(),
    )
    parser.add_argument(
        "--image_width",
        default=64,
        type=int,
        help="Define the Image Width".capitalize(),
    )

    args = parser.parse_args()

    if args.image_path is not None:
        if args.batch_size and args.image_height and args.image_width:
            loader = Loader(
                image_path=args.image_path,
                batch_size=args.batch_size,
                image_height=args.image_height,
                image_width=args.image_width,
                normalized=True,
            )

            loader.extract_features()

            dataloader = loader.create_dataloader()

            logging.info("Data Loader created".capitalize())

        else:
            raise ValueError("Please provide the image path".capitalize())
    else:
        raise ValueError("Please provide the image path".capitalize())
