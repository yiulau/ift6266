import os
import timeit
from logit import *
#os.environ['FUEL_DATA_PATH'] = os.path.abspath('./fuel_data')

# Let's load and process the dataset
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop
from fuel.transformers.image import MinimumImageDimensions
from fuel.transformers import Flatten
from fuel.transformers import ScaleAndShift


train_set = DogsVsCats(('train',), subset=slice(0, 20000))
valid_set = DogsVsCats(('train',), subset=slice(20000, 25000))
test_set = DogsVsCats(('test',))

batch_size = 128
n_train_batches = train_set.num_examples // batch_size

#################################################################################
# Train Stream
# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream instead of DataStream.default_stream constructor we return
# our images exactly as is.
stream = DataStream(
    train_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size)
)

# Our images are of different sizes, so we'll use a Fuel transformer
# to upscale images to at least (512 x 512)

upscale_stream = MinimumImageDimensions(stream, (512,512),which_sources=('image_features',))

# Take random crops of (32 x 32) from each image
cropped_stream = RandomFixedSizeCrop(
    upscale_stream, (32, 32), which_sources=('image_features',))

# Convert images to [0,1] scale
default_cropped_stream = ScaleAndShift(cropped_stream,  1.0/(255.0), 0., which_sources=('image_features',))
# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
train_stream = Flatten(
    default_cropped_stream, which_sources=('image_features',))
    
######################################################################################
# Valid Stream

# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream instead of DataStream.default_stream constructor we return
# our images exactly as is.
stream = DataStream(
    valid_set,
    iteration_scheme=ShuffledScheme(valid_set.num_examples, batch_size)
)

# Our images are of different sizes, so we'll use a Fuel transformer
# to upscale images to at least (512,512)

valid_upscale_stream = MinimumImageDimensions(stream, (512,512),which_sources=('image_features',))

# Take random crops of (32 x 32) from each image
valid_cropped_stream = RandomFixedSizeCrop(
    valid_upscale_stream, (32, 32), which_sources=('image_features',))

# Convert images to [0,1] scale
valid_default_cropped_stream = ScaleAndShift(valid_cropped_stream,  1.0/(255.0), 0., which_sources=('image_features',))
# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
valid_stream = Flatten(
    valid_default_cropped_stream, which_sources=('image_features',))
    
##########################################################################################
# Test Stream 

# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream instead of DataStream.default_stream constructor we return
# our images exactly as is.
stream = DataStream(
    test_set,
    iteration_scheme=ShuffledScheme(test_set.num_examples, batch_size)
)

# Our images are of different sizes, so we'll use a Fuel transformer
# to upscale images to at least (512,512)

test_upscale_stream = MinimumImageDimensions(stream, (512,512),which_sources=('image_features',))

# Take random crops of (32 x 32) from each image
test_cropped_stream = RandomFixedSizeCrop(
    test_upscale_stream, (32, 32), which_sources=('image_features',))

# Convert images to [0,1] scale
test_default_cropped_stream = ScaleAndShift(test_cropped_stream,  1.0/(255.0), 0., which_sources=('image_features',))
# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
test_stream = Flatten(
    test_default_cropped_stream, which_sources=('image_features',))
    
####################################################################################################

## early stopping parameters 
# maximum number of epochs
n_epochs = 1000
# look at this many samples regardless
patience = 5000
# wait this much longer when a new best is found
patience_increase = 2
# a relative improvement of this much is considered significant
improvement_threshold = 0.995
# go through this many minibatches before checking the network on the validation set;
# in this case we check every epoch
validation_frequency = min(n_train_batches, patience / 2)


best_validation_loss = numpy.inf
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0
while (epoch < n_epochs) and (not done_looping):
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

        minibatch_index += 1
        if patience <= iter:
            done_looping = True
            break

end_time = timeit.default_timer()
print('Optimization complete with best validation score of %f %%, '
      'with test performance %f %%' %
      (best_validation_loss * 100., test_score * 100.))

print('The code ran for %d epochs, with %f epochs/sec' %
      (epoch, 1. * epoch / (end_time - start_time)))