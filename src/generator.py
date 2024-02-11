import logging
import argparse
from collections import OrderedDict
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/generator.log",
)


class Generator(nn.Module):
    """
    The Generator class for a Generative Adversarial Network (GAN), responsible for generating synthetic images from a latent space vector.

    This module builds an architecture that progressively upsamples the input latent vector to a full-sized image (e.g., 64x64 pixels) through a series of ConvTranspose2d layers, each followed by BatchNorm2d and ReLU activations, except for the final layer which uses a Tanh activation to output RGB image data.

    Parameters:
    - nz (int): Size of the latent vector (z). Default is 100.
    - ngf (int): Defines the depth of feature maps carried through the generator, relating to the size of the generator's feature maps. Default is 64.

    Attributes:
    - layer_config (OrderedDict): An ordered dictionary specifying the layers and configurations of the generator model.
    - main (nn.Sequential): The sequential container that constitutes the generator, built from `layer_config`.

    The architecture details include upsampling from a latent vector to a 64x64 3-channel RGB image, progressively doubling the spatial dimensions of the feature maps while reducing their depth, starting from `ngf * 8` to `ngf`, and finally outputting to 3 channels.

    Methods:
    - forward(input): Defines the forward pass of the generator.

    Example:
        generator = Generator(nz=100, ngf=64)
        # Assuming `latent_vector` is a batch of random vectors from the latent space
        fake_images = generator(latent_vector)

    Note:
    The output images are normalized between -1 and 1, corresponding to the range of the Tanh activation function used in the final layer.

    Layer Details:
    - ConvTrans1: Upsamples the input latent vector to a spatial dimension of 4x4 with `ngf*8` feature maps.
    - BatchNorm1 and ReLU1: Applied after the first ConvTranspose2d layer.
    - ConvTrans2 to ConvTrans4: Further upsampling steps, each doubling the spatial dimensions and halving the depth of feature maps, with BatchNorm and ReLU activations.
    - ConvTrans5: Final upsampling step to produce a 3-channel RGB image of size 64x64. Followed by a Tanh activation.
    """

    def __init__(self, latent_space=100, image_size=64):
        """
        The Generator class for a Generative Adversarial Network (GAN), responsible for generating synthetic images from a latent space vector.

        This module builds an architecture that progressively upsamples the input latent vector to a full-sized image (e.g., 64x64 pixels) through a series of ConvTranspose2d layers, each followed by BatchNorm2d and ReLU activations, except for the final layer which uses a Tanh activation to output RGB image data.

        Parameters:
        - nz (int): Size of the latent vector (z). Default is 100.
        - ngf (int): Defines the depth of feature maps carried through the generator, relating to the size of the generator's feature maps. Default is 64.

        Attributes:
        - layer_config (OrderedDict): An ordered dictionary specifying the layers and configurations of the generator model.
        - main (nn.Sequential): The sequential container that constitutes the generator, built from `layer_config`.

        The architecture details include upsampling from a latent vector to a 64x64 3-channel RGB image, progressively doubling the spatial dimensions of the feature maps while reducing their depth, starting from `ngf * 8` to `ngf`, and finally outputting to 3 channels.

        Methods:
        - forward(input): Defines the forward pass of the generator.

        Example:
            generator = Generator(nz=100, ngf=64)
            # Assuming `latent_vector` is a batch of random vectors from the latent space
            fake_images = generator(latent_vector)

        Note:
        The output images are normalized between -1 and 1, corresponding to the range of the Tanh activation function used in the final layer.

        Layer Details:
        - ConvTrans1: Upsamples the input latent vector to a spatial dimension of 4x4 with `ngf*8` feature maps.
        - BatchNorm1 and ReLU1: Applied after the first ConvTranspose2d layer.
        - ConvTrans2 to ConvTrans4: Further upsampling steps, each doubling the spatial dimensions and halving the depth of feature maps, with BatchNorm and ReLU activations.
        - ConvTrans5: Final upsampling step to produce a 3-channel RGB image of size 64x64. Followed by a Tanh activation.
        """
        self.nz = latent_space
        self.ngf = image_size
        super(Generator, self).__init__()

        self.layer_config = OrderedDict(
            [
                (
                    "convTrans1",
                    nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
                ),
                ("batchNorm1", nn.BatchNorm2d(self.ngf * 8)),
                ("relu1", nn.ReLU(True)),
                (
                    "convTrans2",
                    nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
                ),
                ("batchNorm2", nn.BatchNorm2d(self.ngf * 4)),
                ("relu2", nn.ReLU(True)),
                (
                    "convTrans3",
                    nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
                ),
                ("batchNorm3", nn.BatchNorm2d(self.ngf * 2)),
                ("relu3", nn.ReLU(True)),
                (
                    "convTrans4",
                    nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
                ),
                ("batchNorm4", nn.BatchNorm2d(self.ngf)),
                ("relu4", nn.ReLU(True)),
                ("convTrans5", nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False)),
                ("tanh", nn.Tanh()),
            ]
        )

        self.main = nn.Sequential(self.layer_config)

    def forward(self, input):
        """
        Forward pass of the generator. Takes a latent vector and generates an image.

        Parameters:
        - input (torch.Tensor): A batch of latent vectors of shape `(N, nz, 1, 1)`, where N is the batch size.

        Returns:
        - torch.Tensor: A batch of generated images of shape `(N, 3, 64, 64)`, with pixel values normalized between -1 and 1.
        """
        return self.main(input)


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

    parser.add_argument(
        "--latent_space",
        type=int,
        default=64,
        help="Image size for training.".capitalize(),
    )

    args = parser.parse_args()

    if args.image_size and args.latent_space:
        logging.info(f"Training with image size: {args.image_size}")

        generator = Generator(
            latent_space=args.latent_space, image_size=args.image_size
        )
        logging.info("Generator loaded.".capitalize())
    else:
        logging.error("Please provide an image size.".capitalize())
