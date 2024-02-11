import logging
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/discriminator.log",
)


class Discriminator(nn.Module):
    """
    A Discriminator class for a Generative Adversarial Network (GAN), designed to differentiate between real and fake images.

    This module implements a series of convolutional layers with LeakyReLU activations and BatchNorm, progressively doubling the number of feature maps while reducing the spatial dimensions of the input image. The final layer uses a Sigmoid activation to output a probability indicating the likelihood of the input image being real.

    Parameters:
    - image_size (int): The height / width of the square input images. Default is 64. This parameter also indirectly controls the complexity of the discriminator's architecture by setting the size of feature maps.

    Attributes:
    - ndf (int): The size of the feature maps in the discriminator, initially set based on the `image_size`.
    - layer_config (OrderedDict): An ordered dictionary that defines the architecture of the discriminator, including convolutional layers, batch normalization layers, and activation functions.
    - main (nn.Sequential): The sequential container of layers as defined in `layer_config`.

    The architecture starts with a convolutional layer with `ndf` (number of discriminator features) channels, followed by layers with `2*ndf`, `4*ndf`, and `8*ndf` channels, before concluding with a final convolution to a single output channel. Batch normalization is applied starting from the second convolutional layer.

    Methods:
    - forward(input): Defines the forward pass of the discriminator.

    Example:
        discriminator = Discriminator(image_size=64)
        # Assuming `images` is a batch of real or generated images
        predictions = discriminator(images)

    Note:
    The input images are expected to be 3-channel RGB images of size `[image_size, image_size]`. The discriminator dynamically adjusts its complexity based on the input image size.

    Layer Details:
    - Conv1: Input 3 channels, output `ndf` channels, 4x4 kernel, stride 2, padding 1, no bias.
    - LeakyReLU1: Negative slope 0.2, inplace.
    - Conv2: Input `ndf` channels, output `2*ndf` channels, 4x4 kernel, stride 2, padding 1, no bias.
    - BN2: BatchNorm on `2*ndf` channels.
    - LeakyReLU2: Negative slope 0.2, inplace.
    - Conv3: Input `2*ndf` channels, output `4*ndf` channels, 4x4 kernel, stride 2, padding 1, no bias.
    - BN3: BatchNorm on `4*ndf` channels.
    - LeakyReLU3: Negative slope 0.2, inplace.
    - Conv4: Input `4*ndf` channels, output `8*ndf` channels, 4x4 kernel, stride 2, padding 1, no bias.
    - BN4: BatchNorm on `8*ndf` channels.
    - LeakyReLU4: Negative slope 0.2, inplace.
    - Conv5: Input `8*ndf` channels, output 1 channel, 4x4 kernel, stride 1, no padding, no bias.
    - Sigmoid: Applied to the final layer output to obtain a probability.
    """

    def __init__(self, image_size=64):
        self.ndf = image_size
        super(Discriminator, self).__init__()
        self.layer_config = OrderedDict(
            [
                ("conv1", nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False)),
                ("leaky1", nn.LeakyReLU(0.2, inplace=True)),
                ("conv2", nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)),
                ("bn2", nn.BatchNorm2d(self.ndf * 2)),
                ("leaky2", nn.LeakyReLU(0.2, inplace=True)),
                ("conv3", nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)),
                ("bn3", nn.BatchNorm2d(self.ndf * 4)),
                ("leaky3", nn.LeakyReLU(0.2, inplace=True)),
                ("conv4", nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)),
                ("bn4", nn.BatchNorm2d(self.ndf * 8)),
                ("leaky4", nn.LeakyReLU(0.2, inplace=True)),
                ("conv5", nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)),
                ("sigmoid", nn.Sigmoid()),
            ]
        )
        self.main = nn.Sequential(self.layer_config)

    def forward(self, input):
        """
        Forward pass of the discriminator. Takes an image tensor and returns the discriminator's prediction.

        Parameters:
        - input (torch.Tensor): A batch of images of shape `(N, 3, image_size, image_size)`.

        Returns:
        - torch.Tensor: A tensor of shape `(N,)` containing the probability that each image in the batch is real.
        """
        return self.main(input).view(-1, 1).squeeze(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images using a GAN model.".capitalize()
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Image size for training.".capitalize(),
    )

    args = parser.parse_args()

    if args.image_size:
        logging.info(f"Training with image size: {args.image_size}")

        discriminator = Discriminator(image_size=args.image_size)
    else:
        logging.error("Please provide an image size.".capitalize())
