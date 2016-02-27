import numpy
import theano
from theano import tensor

# Size of the data
n_in = 32 * 32 * 3 
# Number of classes
n_out = 2

x = tensor.matrix('x')
W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                  name='W',
                  borrow=True)
b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                  name='b',
                  borrow=True)
                  
p_y_given_x = tensor.nnet.softmax(tensor.dot(x, W) + b)
y_pred = tensor.argmax(p_y_given_x, axis=1)

y = tensor.lvector('y')
log_prob = tensor.log(p_y_given_x)
log_likelihood = log_prob[tensor.arange(y.shape[0]), y]
loss = - log_likelihood.mean()

g_W, g_b = theano.grad(cost=loss, wrt=[W, b])

learning_rate = numpy.float32(0.13)
new_W = W - learning_rate * g_W
new_b = b - learning_rate * g_b

train_model = theano.function(inputs=[x, y],
                              outputs=loss,
                              updates=[(W, new_W),
                                       (b, new_b)])
                                       

misclass_nb = tensor.neq(y_pred, y)
misclass_rate = misclass_nb.mean()

test_model = theano.function(inputs=[x, y],
                             outputs=misclass_rate)