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
                             

####################################################################################################################### 
from fuel.streams import ServerDataStream    
import timeit
# number of training batches
n_train_batches = 156

## early stopping parameters 
# maximum number of epochs
n_epochs = 100
# look at this many samples regardless
patience = 5000
# wait this much longer when a new best is found
patience_increase = 2
# a relative improvement of this much is considered significant
improvement_threshold = 0.995
# go through this many minibatches before checking the network on the validation set;
# in this case we check every epoch
validation_frequency = min(n_train_batches, patience / 2)

train_stream = ServerDataStream(sources=('image_features','target'), port=5557, produces_examples=False)
valid_stream = ServerDataStream(sources=('image_features','target'), port=5558, produces_examples=False)
test_stream = ServerDataStream(sources=('image_features','target'), port=5559, produces_examples=False)

time_array = numpy.zeros((n_epochs,),dtype = theano.config.floatX )

best_validation_loss = numpy.inf
test_score = 0.

done_looping = False
epoch = 0
while (epoch < n_epochs) and (not done_looping):
    start_time = timeit.default_timer()
    epoch = epoch + 1
    minibatch_index = 0
    for minibatch_x, minibatch_y in train_stream.get_epoch_iterator():
        minibatch_avg_cost = train_model(minibatch_x, minibatch_y.flatten())

        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = []
            for valid_xi, valid_yi in valid_stream.get_epoch_iterator():
                validation_losses.append(test_model(valid_xi, valid_yi.flatten()))
            this_validation_loss = numpy.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                  (epoch,
                   minibatch_index + 1,
                   n_train_batches,
                   this_validation_loss * 100.))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                # improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss

                # test it on the test set
                test_losses = []
                for test_xi, test_yi in test_stream.get_epoch_iterator():
                    test_losses.append(test_model(test_xi, test_yi.flatten()))

                test_score = numpy.mean(test_losses)
                print('     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                      (epoch,
                       minibatch_index + 1,
                       n_train_batches,
                       test_score * 100.))

                # save the best parameters
                numpy.savez('best_model.npz', W=W.get_value(), b=b.get_value())
        epoch_time = timeit.default_timer()-start_time 
        time_array[epoch-1]=epoch_time 
        numpy.savez('time_array.npz',time_array=time_array)
        minibatch_index += 1
        if patience <= iter:
            done_looping = True
            break

            
print('Optimization complete with best validation score of %f %%, '
      'with test performance %f %%' %
      (best_validation_loss * 100., test_score * 100.))

print('The code ran for %d epochs, with %f epochs/sec' %
      (epoch, 1. * epoch / (end_time - start_time)))