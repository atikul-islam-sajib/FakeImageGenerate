import logging
import argparse
from collections import OrderedDict
import torch.nn as nn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filemode="w",
                    filename="./logs/generator.log")


class Generator(nn.Module):
    """
    A generator model for generating synthetic images from a latent space using a series of
    transposed convolutional layers. This model is typically used in Generative Adversarial Networks (GANs).

    Attributes:
        latent_space (int): The size of the latent space from which the generator synthesizes images.
        image_size (int): The size of one side of the square image to generate. Assumes images are square.
        kernel_size (int): The size of the kernel to use in convolutional layers.
        stride (int): The stride of the convolution.
        padding (int): The padding of the convolution.
        layers_config (list of tuples): A configuration list where each tuple specifies the parameters for each
                                        transposed convolutional layer in the form (in_channels, out_channels,
                                        kernel_size, stride, padding, bias).

    Methods:
        connected_layer(layers_config=None): Constructs the sequential model based on the provided layer configuration.
        forward(x): Defines the forward pass of the generator.
    """

    def __init__(self, latent_space=100, image_size=64):
        """
        Initializes the Generator model with the specified latent space size and image dimensions.

        Parameters:
            latent_space (int): The dimensionality of the input latent space.
            image_size (int): The height/width of the output image.
        """
        super(Generator, self).__init__()
        self.latent_space = latent_space
        self.image_size = image_size
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1

        self.layers_config = [
            (self.latent_space, self.image_size * 8, self.kernel_size, self.stride // 2, 0, False,),
            (self.image_size * 8, self.image_size * 4, self.kernel_size, self.stride, self.padding, False,),
            (self.image_size * 4, self.image_size * 2, self.kernel_size, self.stride, self.padding, False,),
            (self.image_size * 2, self.image_size, self.kernel_size, self.stride, self.padding, False,),
            (self.image_size, 3, self.kernel_size, self.stride, self.padding, False,),
        ]

        self.model = self._create_layers()

    def _create_layers(self):
        """
        Constructs the sequential model based on the provided layer configuration.

        Returns:
            nn.Sequential: A PyTorch Sequential model composed of the specified layers.
        """
        layers = OrderedDict()
        for index, (in_channels, out_channels, kernel_size, stride, padding, bias,) in enumerate(self.layers_config[:-1]):
            layers[f"{index}_convTrans"] = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            )
            layers[f"{index}_batch_norm"] = nn.BatchNorm2d(out_channels)
            layers[f"{index}_relu"] = nn.ReLU(inplace=True)

        (in_channels, out_channels, kernel_size, stride, padding, bias) = (
            self.layers_config[-1]
        )
        layers["final_convTrans"] = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        layers["final_tanh"] = nn.Tanh()

        return nn.Sequential(layers)

    def forward(self, x):
        """
        Defines the forward pass of the generator with the input x.

        Parameters:
            x (Tensor): A PyTorch tensor representing the latent space input.

        Returns:
            Tensor: The generated image as a PyTorch tensor.
        """
        return self.model(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a GAN model.".capitalize())

    parser.add_argument("--image_size", type=int, default=64, help="Image size for training.".capitalize())

    args = parser.parse_args()

    if args.image_size:
        logging.info(f"Training with image size: {args.image_size}")

        generator = Generator(args.image_size)
    else:
        logging.error("Please provide an image size.".capitalize())
