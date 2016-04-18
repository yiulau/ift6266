
# Implements a simple convnet using only theano and fuel code

import theano
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np
from get_datastream import get_datastream

#############################################################################################
# Get datastreams
image_size = (128,128)
mhratio = 1.5
min_dim_size = 256
batchsize = 32
train_stream = get_datastream(image_size,mhratio,min_dim_size,batchsize,"train")
valid_stream = get_datastream(image_size,mhratio,min_dim_size,batchsize,"valid")

x,y = next(train_stream.get_epoch_iterator())

###############################################################################################
# Build computational graph 
batch_input = theano.tensor.ftensor4("image_batch")

batch_printed = theano.printing.Print("print intermediate value")(batch_input)

batch_shape = theano.printing.Print("print shape")(batch_input.shape)

f_with_print = theano.function([batch_input],batch_shape)

#f_with_print(x)

# layer 1 

K1 = theano.shared(np.random.randn(64,3,8,8).astype(theano.config.floatX),
                   "filters_layer1")

o1 = theano.tensor.nnet.conv2d(batch_input,K1,border_mode="full")

o1_shape = theano.printing.Print("print shape")(o1.shape)

f_with_print = theano.function([batch_input],o1_shape)

f_with_print(x)

down_sampled1 = max_pool_2d(o1,(2,2),True)

down_sampled1_shape = theano.printing.Print("print shape")(down_sampled1.shape)

f_with_print = theano.function([batch_input],down_sampled1_shape)

f_with_print(x)

relu_o1 = theano.tensor.nnet.relu(down_sampled1)

theano.printing.debugprint(relu_o1)

# layer 2

K2 = theano.shared(np.random.randn(128,64,8,8).astype(theano.config.floatX),
                           "filters_layer2")
o2 = theano.tensor.nnet.conv2d(relu_o1,K2,border_mode="full")

down_sampled2 = max_pool_2d(o2,(2,2),True)

relu_o2 = theano.tensor.nnet.relu(down_sampled2)

relu_o2_shape = theano.printing.Print("print shape")(relu_o2.shape)

f_with_print = theano.function([batch_input],relu_o2_shape)

f_with_print(x)

# layer 3

K3 = theano.shared(np.random.randn(256,128,4,4).astype(theano.config.floatX),
                           "filters_layer3")
o3 = theano.tensor.nnet.conv2d(relu_o2,K3,border_mode="full")

down_sampled3 = max_pool_2d(o3,(2,2),True)

relu_o3 = theano.tensor.nnet.relu(down_sampled3)

relu_o3_shape = theano.printing.Print("print shape")(relu_o3.shape)

f_with_print = theano.function([batch_input],relu_o3_shape)

f_with_print(x)

# layer 4

K4 = theano.shared(np.random.randn(512,256,4,4).astype(theano.config.floatX),
                           "filters_layer4")
o4 = theano.tensor.nnet.conv2d(relu_o3,K4,border_mode="full")

down_sampled4 = max_pool_2d(o4,(2,2),True)

relu_o4 = theano.tensor.nnet.relu(down_sampled4)

relu_o4_shape = theano.printing.Print("print shape")(relu_o4.shape)

f_with_print = theano.function([batch_input],relu_o4_shape)

f_with_print(x)

# layer 5 

K5 = theano.shared(np.random.randn(256,512,3,3).astype(theano.config.floatX),
                           "filters_layer5")
o5 = theano.tensor.nnet.conv2d(relu_o4,K5,border_mode="full")

down_sampled5 = max_pool_2d(o5,(2,2),True)

relu_o5 = theano.tensor.nnet.relu(down_sampled5)

relu_o5_shape = theano.printing.Print("print shape")(relu_o5.shape)

f_with_print = theano.function([batch_input],relu_o5_shape)

f_with_print(x)

# layer 6 

output_from_conv_inputsize = relu_o5.flatten(2).shape[-1]

tempo = theano.printing.Print("print shape")(output_from_conv_inputsize)

f_with_print = theano.function([batch_input],tempo)

f_with_print(x)


W5 = theano.shared(np.random.rand(9216,1000).astype(theano.config.floatX),
                   "weights_layer5")
b5 = theano.shared(np.random.rand(1000,).astype(theano.config.floatX),
                   "bias_layer5")

h5 = theano.tensor.nnet.relu(theano.tensor.dot(relu_o5.flatten(2),W5)+
                             b5.dimshuffle("x",0,"x","x"))


