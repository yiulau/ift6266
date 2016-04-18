"""Convolutional network example.

Run the training for 50 epochs with
```
python convnet1.py --num-epochs 50
```
It is going to reach around 0.8% error rate on the test set.

"""
import logging
import numpy
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale, Momentum
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
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.roles import WEIGHT
#from blocks.fitler import VariableFilter
from blocks.serialization import continue_training
from blocks_extras.extensions.plot import Plot
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream,ServerDataStream
from toolz.itertoolz import interleave
from get_datastream import get_datastream

class LeNet(FeedforwardSequence, Initializable):
    """LeNet-like convolutional network.

    The class implements LeNet, which is a convolutional sequence with
    an MLP on top (several fully-connected layers). For details see
    [LeCun95]_.

    .. [LeCun95] LeCun, Yann, et al.
       *Comparison of learning algorithms for handwritten digit
       recognition.*,
       International conference on artificial neural networks. Vol. 60.

    Parameters
    ----------
    conv_activations : list of :class:`.Brick`
        Activations for convolutional network.
    num_channels : int
        Number of channels in the input image.
    image_shape : tuple
        Input image shape.
    filter_sizes : list of tuples
        Filter sizes of :class:`.blocks.conv.ConvolutionalLayer`.
    feature_maps : list
        Number of filters for each of convolutions.
    pooling_sizes : list of tuples
        Sizes of max pooling for each convolutional layer.
    top_mlp_activations : list of :class:`.blocks.bricks.Activation`
        List of activations for the top MLP.
    top_mlp_dims : list
        Numbers of hidden units and the output dimension of the top MLP.
    conv_step : tuples
        Step of convolution (similar for all layers).
    border_mode : str
        Border mode of convolution (similar for all layers).

    """
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims,
                 conv_step=None, border_mode='valid', **kwargs):
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        conv_parameters = zip(filter_sizes, feature_maps)

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations,
            (MaxPooling(size, name='pool_{}'.format(i))
             for i, size in enumerate(pooling_sizes))]))

        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()
        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(LeNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims

def main(jobid, save_to, num_epochs, feature_maps=None, 
         mlp_hiddens=None,conv_sizes=None, pool_sizes=None, batch_size=32,
         from_server=0,num_batches=None,image_size=64,live_plot=0,resume_training=0):
    if feature_maps is None:
        feature_maps = [20, 50]
    if mlp_hiddens is None:
        mlp_hiddens = [500]
    if conv_sizes is None:
        conv_sizes = [5, 5]
    if pool_sizes is None:
        pool_sizes = [2, 2]
##################################################################################    
   # features_maps = [32, 32, 64, 64, 128, 256]
   # mlp_hiddens = [256]
   # conv_sizes = [5, 5, 5, 4, 4, 4]
   # pool_sizes = [2, 2, 2, 2, 2, 2]
   # image_size = 260
   # if num_batches == 0:
     #   num_batches = None  
    if from_server == 0:
        from_server = False
    else:
        from_server = True
    if live_plot == 0:
        live_plot = False
    else:
        live_plot = True
    if resume_training == 0:
        resume_training = False
    else:
        resume_training = True
    save_to = `jobid` + "_" + save_to  
    
################################################################################
    output_size = 2
    image_size = (image_size,image_size) 
    # Use ReLUs everywhere and softmax for the final prediction
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    convnet = LeNet(conv_activations, 3, image_size,
                    filter_sizes=zip(conv_sizes, conv_sizes),
                    feature_maps=feature_maps,
                    pooling_sizes=zip(pool_sizes, pool_sizes),
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='full',
                    weights_init=Uniform(width=.2),
                    biases_init=Constant(0))
    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.push_initialization_config()
    convnet.layers[0].weights_init = Uniform(width=.2)
    convnet.layers[1].weights_init = Uniform(width=.09)
    convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    convnet.initialize()
    logging.info("Input dim: {} {} {}".format(
        *convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))
    x = tensor.tensor4('image_features')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    cost = (CategoricalCrossEntropy().apply(y.flatten(), probs)
            .copy(name='cost'))
    error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                  .copy(name='error_rate'))

    cg = ComputationGraph([cost, error_rate])
    #weights = VariableFilter(roles=[WEIGHT])(cg.variables)
    #penalty = (weights[0]**2).sum() + (weights[1]**2).sum()
    #cost = lam * penalty + cost
    #cost.name = 'regularized_cost' 
    if from_server:
        train_stream = ServerDataStream(('image_features','targets'),produces_examples=False,port=5007)
        valid_stream = ServerDataStream(('image_features','targets'),produces_examples=False,port=5008)
    else:
        train_stream = get_datastream(image_size,1.5,300,batch_size,'train')
        valid_stream = get_datastream(image_size,1.5,300,batch_size,'valid')


    # Train with simple SGD
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        #step_rule=Momentum(learning_rate=0.001,momentum=0.1))
        step_rule=Scale(learning_rate=0.001))
# `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(every_n_epochs=num_epochs,
                              every_n_batches=num_batches),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      valid_stream,
                      prefix="valid",after_epoch=True),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to,save_separately=['log']),
                  ProgressBar(),
                  Printing()]
    if live_plot:
        extensions.append(Plot("Plotting",
                               channels=[['train_cost','valid_cost'],
                                         ['train_error_rate','valid_error_rate']],
                               after_batch=True))
    if resume_training:
        extensions.append(Load(path=save_to,load_log=True))
    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        train_stream,
        model=model,
        extensions=extensions)
    
    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                        "on the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=2,
                    help="Number of training epochs to do.")
    parser.add_argument("save_to", default="dogvcats.pkl", nargs="?",
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
    parser.add_argument("--batch-size", type=int, default=64,
                    help="Batch size.")
    parser.add_argument("--from-server",type=int,default=0,
                    help="Are we getting the data from the server.1 if true 0 otherwise")
    parser.add_argument("--num-batches",type=int,default=0,
                    help="number of batches before stopping training")
    parser.add_argument("--image-size",type=int,default=64,
                    help="side of the square image from datastream")
    parser.add_argument("--live-plot",type=int,default=0,
                    help="1 if we want live plotting, 0 otherwise")
    parser.add_argument("--resume-training",type=int,default=0,
                    help="1 if we want to resume training from save_to, 0 otherwise")
    parser.add_argument("jobid",type=int, 
                    help="job id. For now must be set by hand")

    args = parser.parse_args()
    main(**vars(args))
