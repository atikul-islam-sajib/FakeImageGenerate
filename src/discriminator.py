import logging
import argparse
from collections import OrderedDict
import torch.nn as nn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    filemode="w",
                    filename="./logs/discriminator.log")


class Discriminator(nn.Module):
    """
    A Discriminator class for a Generative Adversarial Network (GAN), designed to classify images as real or fake.

    This discriminator uses a series of convolutional layers with LeakyReLU activation functions and optional batch normalization
    to process input images. The final output is a single value through a sigmoid activation function, indicating the likelihood
    of the input image being real.

    Attributes:
        image_size (int): The size of the input images. Defaults to 64.
        input_channels (int): The number of channels in the input images. Defaults to 3 (for RGB images).
        kernel_size (int): The size of the convolving kernel. Defaults to 4.
        stride (int): The stride of the convolution. Defaults to 2.
        padding (int): The padding added to all sides of the input. Defaults to 1.
        negative_slope (float): The negative slope value of the LeakyReLU activation function. Defaults to 0.2.
        layers_config (list of tuples): Configuration for each layer in the model, specifying in_channels, out_channels,
                                        kernel_size, stride, padding, negative_slope, and whether to use batch normalization.

    Methods:
        connected_layer(layers_config): Constructs the discriminator model based on the provided layer configuration.
        forward(x): Defines the forward pass of the discriminator.

    Example:
        >>> discriminator = Discriminator(image_size=64)
        >>> print(discriminator)
    """

    def __init__(self, image_size=64):
        """
        Initializes the Discriminator with a specific image size and default values for other parameters. It builds the
        discriminator model by setting up the layers based on a predefined configuration.

        Parameters:
            image_size (int): The size of the input images. Defaults to 64.
        """
        self.image_size = image_size
        self.input_channels = 3
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.negative_slope = 0.2

        super(Discriminator, self).__init__()

        self.layers_config = [
            (self.input_channels, self.image_size, self.kernel_size,
             self.stride, self.padding, self.negative_slope, False,),
            (self.image_size, self.image_size * 2, self.kernel_size,
             self.stride, self.padding, self.negative_slope, True,),
            (self.image_size * 2, self.image_size * 4, self.kernel_size,
             self.stride, self.padding, self.negative_slope, True,),
            (self.image_size * 4, self.image_size * 8, self.kernel_size,
             self.stride, self.padding, self.negative_slope, True,),
            (self.image_size * 8, 1, self.kernel_size, self.stride // 2, 0),
        ]

        self.model = self.connected_layer(layers_config=self.layers_config)

    def connected_layer(self, layers_config=None):
        """
        Constructs the discriminator model from the specified layer configuration. Each layer consists of a convolutional
        layer, possibly followed by batch normalization, and a LeakyReLU activation function.

        Parameters:
            layers_config (list of tuples): The configuration for each layer in the model. If None, an exception is raised.

        Returns:
            torch.nn.Sequential: The constructed discriminator model as a sequential container.
        """
        layers = OrderedDict()
        if layers_config is not None:
            for index, config in enumerate(layers_config[:-1]):
                (in_channels, out_channels, kernel_size, stride, padding, negative_slope, use_batch_norm,
                 ) = config

                layers[f"{index+1}_conv"] = nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                )
                layers[f"{index+1}_activation"] = nn.LeakyReLU(
                    negative_slope, inplace=True
                )
                if use_batch_norm:
                    layers[f"{index+1}_batch_norm"] = nn.BatchNorm2d(out_channels)

            in_channels, out_channels, kernel_size, stride, padding = layers_config[-1]
            layers["out_layer"] = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
            layers["out_activation"] = nn.Sigmoid()

            return nn.Sequential(layers)
        else:
            raise ValueError("Layer configuration is not defined properly.")

    def forward(self, x):
        """
        Defines the forward pass of the discriminator.

        Parameters:
            x (torch.Tensor): The input tensor containing the images to classify.

        Returns:
            torch.Tensor: The output tensor representing the likelihood of each image being real.

        Raises:
            ValueError: If the input tensor is not defined properly.
        """
        if x is not None:
            x = self.model(x)
            return x.view(-1, 1).squeeze(1)
        else:
            raise ValueError("Input is not defined properly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Define the Discriminator model".title())

    parser.add_argument("--image_size", type=int, default=64, help="The size of the input image.".capitalize())

    args = parser.parse_args()

    if args.image_size >= 64:
        logging.info("Discriminator model with image size {}".format(args.image_size))

        discriminator = Discriminator(args.image_size)

    else:
        logging.error("Image size must be greater than or equal to 64.".capitalize())
