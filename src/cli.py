import sys
import logging
import argparse

sys.path.append("src")

from dataloader import Loader
from trainer import Trainer
from test import Test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filemode="w",
    filename="./logs/cli.log",
)

if __name__ == "__main__":
    """
    # CLI for the Generative Adversarial Network (GAN)

    This script provides a command-line interface (CLI) for setting up and running a Generative Adversarial Network (GAN) for image generation tasks. It supports various configurations, including data loading, model training, testing, and result display options.

    ## Features

    - **Custom Image Path:** Specify the path to your dataset for training the GAN.
    - **Batch Size:** Adjust the batch size for model training.
    - **Image Size:** Define the resolution size of the images for the model.
    - **Learning Rate:** Set the learning rate for the training process.
    - **Epochs:** Specify the number of training cycles.
    - **Display:** Enable or disable the display of results during the training.
    - **Device:** Choose the computational device ('cpu' or 'cuda') for training.
    - **Number of Samples:** Define the number of samples to generate for testing.
    - **Latent Space Size:** Set the size of the latent space for generating noise samples.
    - **Testing Mode:** Enable testing mode to evaluate the model.

    ## Usage

    ```bash
    python <script_name>.py [--image_path PATH] [--batch_size INT] [--image_size INT] [--lr FLOAT] [--epochs INT] [--display BOOL] [--device DEVICE] [--num_samples INT] [--latent_space INT] [--test]
    ```

    ### Arguments

    - `--image_path`: Define the path to the image dataset. (Required for training)
    - `--batch_size`: Define the batch size. Default is 64.
    - `--image_size`: Define the size of the images (e.g., 64 for 64x64 pixels). Default is 64.
    - `--lr`: Set the learning rate. Default is 0.0002.
    - `--epochs`: Specify the number of training epochs. Default is 100.
    - `--display`: Enable result display after training. Pass `True` to enable. Default is `False`.
    - `--device`: Choose the computation device (`cpu` or `cuda`). Default is `cpu`.
    - `--num_samples`: Define the number of samples to generate in test mode. Default is 20.
    - `--latent_space`: Set the size of the latent space. Default is 100.
    - `--test`: Run the model in test mode. Generates images based on the trained model.

    ## Examples

    Training the GAN with custom configurations:

    ```bash
    python <script_name>.py --image_path ./data/images --batch_size 32 --image_size 128 --lr 0.0001 --epochs 150 --device cuda --display True
    ```

    Running the model in test mode to generate samples:

    ```bash
    python <script_name>.py --latent_space 100 --num_samples 30 --test
    ```

    ## Logging

    The script logs important events and errors during execution. Logs are saved to `./logs/cli.log`.

    """

    parser = argparse.ArgumentParser(description="CLI for the GAN".title())

    parser.add_argument("--image_path", help="Define the Image path".capitalize())
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Define the Batch Size".capitalize()
    )

    parser.add_argument(
        "--image_size", type=int, default=64, help="Define the image size".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Define the learning rate".capitalize()
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Define the number of epochs".capitalize(),
    )
    parser.add_argument(
        "--display", type=bool, default=False, help="Display the results".capitalize()
    )
    parser.add_argument(
        "--device", default="cpu", help="Choose the device".capitalize()
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Defined the num of samples".capitalize(),
    )
    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Define the latent space for creating noise samples".capitalize(),
    )
    parser.add_argument("--test", action="store_true", help="Run the test".capitalize())

    args = parser.parse_args()

    if args.display:
        if (
            args.image_path
            and args.batch_size
            and args.image_size
            and args.lr
            and args.epochs
            and args.device
            and args.latent_space
        ):
            loader = Loader(
                image_path=args.image_path, batch_size=args.batch_size, normalized=True
            )

            logging.info("Extracting features...".capitalize())

            loader.extract_features()
            dataloader = loader.create_dataloader()

            logging.info("Creating the model...".capitalize())

            logging.info("Training the model...".capitalize())

            trainer = Trainer(
                device=args.device,
                latent_space=args.latent_space,
                image_size=args.image_size,
                lr=args.lr,
                epochs=args.epochs,
                display=args.display,
            )
            trainer.train()

            logging.info("Training is completed".capitalize())
        else:
            logging.error("Initialization is wrong".capitalize())

    if args.test:
        if args.latent_space and args.num_samples:
            test = Test(latent_space=args.latent_space, num_samples=args.num_samples)
            test.test()
        else:
            raise Exception("Please provide the latent and num samples".capitalize())
    else:
        raise Exception("Please provide the test flag".capitalize())
