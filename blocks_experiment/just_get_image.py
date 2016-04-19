import logging
import numpy
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Softmax, Activation)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint, Load
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.log.log import TrainingLog
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream, ServerDataStream
from toolz.itertoolz import interleave
from get_datastream import get_datastream

image_size=(256,256)
batch_size=32
train_stream = get_datastream(image_size,1.5,300,batch_size,'train')























