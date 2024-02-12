import sys
import logging
import argparse
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("src/")

from discriminator import Discriminator
from generator import Generator
from utils import weight_init, device_init


class Trainer:
    """
    Trainer class for a Generative Adversarial Network (GAN) encapsulates the training process, including initialization, training loops for the discriminator and generator, and saving the model. It handles training over a specified number of epochs, optimizes both the generator and discriminator models, and logs training progress.

    Parameters:
    - device (torch.device): The device to train on, e.g., 'cpu' or 'cuda'.
    - latent_space (int, optional): Dimension of the latent space vector. Defaults to 100.
    - image_size (int, optional): Height and width of the images to generate. Defaults to 64.
    - lr (float, optional): Learning rate for the Adam optimizers. Defaults to 0.0002.
    - epochs (int, optional): Number of training epochs. Defaults to 100.

    Attributes:
    - netG (Generator): The generator model.
    - netD (Discriminator): The discriminator model.
    - optimizerD (torch.optim.Optimizer): Optimizer for the discriminator.
    - optimizerG (torch.optim.Optimizer): Optimizer for the generator.
    - criterion (nn.Module): Loss function (Binary Cross Entropy Loss).
    - real_label (float): Label for real images (1.0).
    - fake_label (float): Label for fake images (0.0).
    - nz (int): Size of the latent vector (z).
    - num_epochs (int): Number of epochs for training.

    Methods:
    - model_init(): Initializes the models and applies weights initialization.
    - optimizer_init(generator, discriminator): Initializes the optimizers for both models.
    - train_discriminator(data): Performs a single training step for the discriminator.
    - train_generator(fake): Performs a single training step for the generator.
    - display_results(epoch, i, dataloader, errD, errG, D_x, D_G_z1, D_G_z2): Logs training progress to the console.
    - save_generator_model(epoch): Saves the current state of the generator model.
    - dataloader(): Loads and returns a dataloader instance.
    - train(): Executes the training loop over the specified number of epochs.

    Example:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(device=device, epochs=30, lr=0.0002)
        trainer.train()

    Note:
    This class assumes the presence of `Generator` and `Discriminator` classes, along with a `weights_init` function for model weight initialization. The dataloader is expected to be loaded using joblib from a specified path.
    """

    def __init__(
        self,
        device="cpu",
        latent_space=100,
        image_size=64,
        lr=0.0002,
        epochs=100,
        display=False,
    ):
        self.device = device_init(device=device)
        self.nz = latent_space
        self.image_size = image_size
        self.lr = lr
        self.num_epochs = epochs
        self.display = display

        try:
            self.netG, self.netD = self.model_init()
        except Exception as e:
            print("Exception caught in model initialization # {}".format(e))
            logging.exception("Model initialization error # {}".capitalize().format(e))
        else:
            self.netG = self.netG.to(self.device)
            self.netD = self.netD.to(self.device)
            self.optimizerD, self.optimizerG = self.optimizer_init(
                generator=self.netG, discriminator=self.netD
            )
        finally:
            self.criterion = nn.BCELoss()
            self.real_label = 1
            self.fake_label = 0

    def model_init(self):
        """
        Initializes the Generator and Discriminator models for the GAN. This method constructs the models with the specified latent space size and image size, moves them to the appropriate device (CPU or GPU), and applies a predefined weight initialization function to both models.

        The models are defined by the Generator and Discriminator classes, which should be available in the same scope as this Trainer class. The latent space size and image size are used to configure the models according to the specifics of the GAN architecture being trained.

        Returns:
            tuple: A tuple containing two nn.Module objects:
                - netG (Generator): The initialized generator model, ready for training.
                - netD (Discriminator): The initialized discriminator model, ready for training.

        Side effects:
            - Instantiates the Generator and Discriminator models with the specified configurations.
            - Applies a predefined weight initialization function to both models to ensure optimal training behavior.
            - Moves the models to the specified device, which is typically determined by whether a GPU is available for training.

        Note:
            The device used for training is determined by the 'device' attribute of the Trainer class instance. The weights initialization function applied to both models is defined externally and must be available in the same scope as this Trainer class.
        """
        netG = Generator(latent_space=self.nz, image_size=self.image_size)
        netD = Discriminator(image_size=self.image_size)
        netG.apply(weight_init)
        netD.apply(weight_init)

        return netG, netD

    def optimizer_init(self, generator, discriminator):
        """
        Initializes the optimizers for both the generator and discriminator models. This method sets up Adam optimizers with specified learning rates and betas parameters, which are critical for the training dynamics of the Generative Adversarial Network (GAN).

        Parameters:
        - generator (torch.nn.Module): The generator model for which the optimizer will be initialized. This model should already be instantiated and configured with the appropriate architecture for generating images.
        - discriminator (torch.nn.Module): The discriminator model for which the optimizer will be initialized. This model should already be instantiated and configured with the appropriate architecture for discriminating between real and generated images.

        Returns:
        - tuple: A tuple containing two optimizer objects:
            - optimizerD (torch.optim.Adam): The Adam optimizer configured for the discriminator model, including learning rate and betas parameters.
            - optimizerG (torch.optim.Adam): The Adam optimizer configured for the generator model, including learning rate and betas parameters.

        Note:
        - The learning rate (`lr`) and betas parameters for the Adam optimizers are critical hyperparameters that can affect the training stability and convergence of the GAN. These parameters are set based on best practices and empirical results but may require adjustment based on the specific characteristics of the dataset or model architecture.
        - This method assumes that the `lr` attribute (learning rate) is already set in the Trainer class instance and uses this value for both optimizers. The betas parameters are fixed in this implementation but could be exposed as parameters or attributes for more flexibility.
        """
        optimizerD = optim.Adam(
            params=discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        optimizerG = optim.Adam(
            params=generator.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )

        return optimizerD, optimizerG

    def train_discriminator(self, data):
        """
        Trains the discriminator model on both real and generated (fake) images. This method performs a forward pass with real images from the dataset and fake images generated by the generator, computes the loss for both, backpropagates to update the discriminator's weights, and returns the losses and discriminator outputs.

        Parameters:
        - data (torch.Tensor): A batch of real images from the dataset. This tensor should have the shape (N, C, H, W), where N is the batch size, C is the number of channels, and H and W are the height and width of the images.

        Returns:
        - tuple: A tuple containing the following elements:
            - errD (torch.Tensor): The total discriminator loss calculated as the sum of the loss for real and fake images.
            - D_x (float): The mean output of the discriminator for real images. This value is used to evaluate the discriminator's performance on real data.
            - D_G_z1 (float): The mean output of the discriminator for fake images before the generator update. This value is used to evaluate the discriminator's performance on fake data.
            - fake (torch.Tensor): A batch of fake images generated by the generator.

        The method performs the following steps:
        1. Zeroes the gradients of the discriminator.
        2. Processes a batch of real images, computes the loss against the true labels, backpropagates the error, and calculates the mean discriminator output (D_x).
        3. Generates a batch of fake images using the generator, computes the loss against the false labels, backpropagates the error, and calculates the mean discriminator output for the fake images (D_G_z1).
        4. Updates the discriminator's weights based on the total loss.

        Note:
        - This method updates the discriminator's weights once per call, using the combined loss from both real and fake images.
        - The real_label and fake_label attributes of the Trainer class are used to denote the true and false labels, respectively, for computing the loss.
        """
        self.netD.zero_grad()
        real_cpu = data[0].to(self.device)
        batch_size = real_cpu.size(0)
        label = torch.full(
            (batch_size,), self.real_label, dtype=torch.float, device=self.device
        )
        output = self.netD(real_cpu)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake = self.netG(noise)
        label.fill_(self.fake_label)
        output = self.netD(fake.detach())
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        self.optimizerD.step()

        return errD, D_x, D_G_z1, fake

    def train_generator(self, fake):
        """
        Trains the generator model by attempting to fool the discriminator.

        This method updates the generator's weights based on its ability to generate fake images that the discriminator classifies as real. It computes the loss using the output of the discriminator on the generated (fake) images, performs backpropagation to calculate the gradients, and updates the generator's weights to reduce this loss.

        Parameters:
        - fake (torch.Tensor): A batch of fake images generated by the generator model. The tensor should have dimensions (N, C, H, W), where N is the batch size, C is the number of channels, and H and W are the height and width of the images, respectively.

        Returns:
        - tuple: A tuple containing the following elements:
            - errG (torch.Tensor): The loss of the generator computed as the binary cross-entropy loss between the discriminator's output on the fake images and the real labels.
            - D_G_z2 (float): The average output of the discriminator for the fake images. This metric indicates how well the generator is fooling the discriminator, with higher values suggesting better performance.

        The training step involves:
        1. Zeroing the gradients of the generator to ensure that previous training steps do not affect the current update.
        2. Creating a tensor of real labels (since the generator's goal is to have its fake images classified as real by the discriminator) and computing the loss against the discriminator's predictions on the fake images.
        3. Performing backpropagation to calculate the gradients with respect to the generator's parameters.
        4. Updating the generator's weights using the optimizer to reduce the loss, thereby improving the generator's ability to produce realistic images.

        Note:
        - The method assumes the use of the binary cross-entropy loss (BCELoss) to quantify how well the generator is fooling the discriminator. The labels for the fake images are set to 'real' (self.real_label) for the purpose of this loss calculation.
        - This method directly influences the generator's performance by adjusting its ability to create images that are indistinguishable from real images to the discriminator.
        """
        self.netG.zero_grad()
        label = torch.full(
            (fake.size(0),), self.real_label, dtype=torch.float, device=self.device
        )
        output = self.netD(fake)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()

        return errG, D_G_z2

    def display_results(self, epoch, i, dataloader, errD, errG, D_x, D_G_z1, D_G_z2):
        """
        Displays the training results and progress metrics for the current batch and epoch.

        This method logs the losses of the discriminator and generator, as well as the discriminator's performance on real and fake images. It provides insights into how well the discriminator and generator are learning and adapting during the training process.

        Parameters:
        - epoch (int): The current epoch number during training.
        - i (int): The current batch number within the epoch.
        - dataloader (DataLoader): The DataLoader used for training, utilized here to determine the total number of batches.
        - errD (float): The current loss of the discriminator.
        - errG (float): The current loss of the generator.
        - D_x (float): The average output of the discriminator for real images. Closer to 1 indicates better performance on real images.
        - D_G_z1 (float): The average output of the discriminator for fake images before the generator update. Closer to 0 indicates better discrimination of fake images.
        - D_G_z2 (float): The average output of the discriminator for fake images after the generator update. Closer to 1 indicates the generator is improving in fooling the discriminator.

        Output:
        - The method prints a formatted string to the console, summarizing the training metrics for the current batch within the ongoing epoch.

        Note:
        - This method is intended for logging purposes and does not return any values. It provides a snapshot of the training progress at the moment it is called, allowing for monitoring of the GAN's learning dynamics.
        """
        if self.display == True:
            print(
                f"[{epoch}/{self.num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD:.4f} Loss_G: {errG:.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
            )
        else:
            logging.info(
                f"[{epoch}/{self.num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD:.4f} Loss_G: {errG:.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
            )

    def save_generator_model(self, epoch):
        """
        Saves the state dictionary of the generator model to a file, capturing its current weights.

        This method is typically called at the end of each training epoch to persist the state of the generator model, allowing for later use or further training from the saved state. The filename includes the epoch number for easy identification and versioning.

        Parameters:
        - epoch (int): The current epoch number. This is used to name the saved model file, indicating at which point in the training process the model was saved.

        Output:
        - The method saves the generator's state dictionary to a file in the current working directory. The file is named 'generator_epoch_{epoch}.pth', where `{epoch}` is replaced with the current epoch number.

        Note:
        - This method does not return any value. It performs a file I/O operation to write the generator model's state dictionary to disk.
        """
        if epoch != self.num_epochs:
            torch.save(
                self.netG.state_dict(), f"./models/checkpoints/generator_{epoch}.pth"
            )
        else:
            torch.save(
                self.netG.state_dict(), f"./models/best_model/generator_{epoch}.pth"
            )

    def dataloader(self):
        """
        Loads and returns the training data loader from a serialized file.

        This method is responsible for loading the training data loader, which has been previously saved to disk using serialization (e.g., with joblib). It allows for quick loading of preprocessed and prepared batches of data for training.

        Returns:
        - DataLoader: The loaded DataLoader object ready for iteration. This dataloader is expected to yield batches of training data during the training loop.

        Note:
        - The dataloader is loaded from a predefined path '../data/processed/dataloader.pkl'. This path must exist and contain a serialized DataLoader object. The method assumes the preprocessing and preparation of data are already completed and saved to this location.
        - This method performs a file I/O operation to read the DataLoader object from disk. Ensure the specified path is accessible and the file format is compatible with the joblib library.
        """
        return joblib.load("./data/processed/dataloader.pkl")

    def train(self):
        """
        Executes the training loop for the Generative Adversarial Network (GAN).

        This method orchestrates the training process by iterating over a specified number of epochs, during which it trains the discriminator and generator models in sequence. At each step of the training, it logs the progress, including the losses of both models and the discriminator's performance metrics. At the end of each epoch, it saves the current state of the generator model.

        The training loop follows these steps:
        1. Loads the data using the `dataloader` method, which should return an iterable DataLoader object containing the training data.
        2. Iterates over the specified number of epochs (as defined by `self.num_epochs`).
            a. For each batch in the DataLoader:
                i. Trains the discriminator on both real and fake data, computing its loss.
                ii. Generates a new batch of fake data and trains the generator, attempting to fool the discriminator, computing its loss.
                iii. Logs the current losses and discriminator performance metrics using the `display_results` method.
        3. Saves the state of the generator model after each epoch using the `save_generator_model` method.

        Note:
        - The actual training of the discriminator and generator is performed by the `train_discriminator` and `train_generator` methods, respectively. This method coordinates these calls and handles logging and model state saving.
        - Progress logging and model saving are designed to provide insights into the training process and to allow for interruption and resumption of training without loss of progress.
        """
        dataloader = self.dataloader()
        for epoch in range(self.num_epochs):
            for i, data in enumerate(dataloader, 0):
                errD, D_x, D_G_z1, fake = self.train_discriminator(data)
                errG, D_G_z2 = self.train_generator(fake)
                self.display_results(
                    epoch, i, dataloader, errD, errG, D_x, D_G_z1, D_G_z2
                )
            self.save_generator_model(epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Training".title())

    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Define the latent space".capitalize(),
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
    args = parser.parse_args()

    if args.display:
        if (
            args.image_size
            and args.lr
            and args.epochs
            and args.latent_space
            and args.device
        ):
            logging.info("Training the GAN model...".capitalize())

            trainer = Trainer(
                device=args.device,
                latent_space=args.latent_space,
                image_size=args.image_size,
                lr=args.lr,
                epochs=args.epochs,
                display=args.display,
            )
            trainer.train()

            logging.info("Training completed!".capitalize())
    else:
        raise Exception("Please, activate the display option".capitalize())
