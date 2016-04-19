import logging
import numpy
from argparse import ArgumentParser


logging.basicConfig(level=logging.INFO)
parser = ArgumentParser("An example of training a convolutional network "
                                "on the MNIST dataset.")
parser.add_argument("--num-epochs", type=int, default=2,
                            help="Number of training epochs to do.")
parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                            help="Destination to save the state of the training "
                                                     "process.")
parser.add_argument("--feature-maps", type=int, nargs='+',
                            default=[20, 50], help="List of feature maps numbers.")
parser.add_argument("--mlp-hiddens", type=int, nargs='+', default=[500],
                            help="List of numbers of hidden units for the MLP.")
parser.add_argument("--conv-sizes", type=int, nargs='+', default=[5, 5],
                            help="Convolutional kernels sizes. The kernels are "
                                                "always square.")
parser.add_argument("--pool-sizes", type=int, nargs='+', default=[2, 2],
                            help="Pooling sizes. The pooling windows are always "
                                                     "square. Should be the same length as "
                                                                              "--conv-sizes.")
parser.add_argument("--batch-size", type=int, default=500,
                            help="Batch size.")
args = parser.parse_args()
