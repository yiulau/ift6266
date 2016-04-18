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

upscale_stream = MinimumImageDimensions(stream, (256,256),which_sources=('image_features',))

# Take random crops of (32 x 32) from each image
cropped_stream = RandomFixedSizeCrop(
    upscale_stream, (32, 32), which_sources=('image_features',))

# Convert images to [0,1] scale
default_cropped_stream = ScaleAndShift(cropped_stream,  1.0/(255.0), 0., which_sources=('image_features',))
# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
train_stream = Flatten(
    default_cropped_stream, which_sources=('image_features',))
    
#################################################################################
# Valid Stream
# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream instead of DataStream.default_stream constructor we return
# our images exactly as is.
stream = DataStream(
    valid_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size)
)

# Our images are of different sizes, so we'll use a Fuel transformer
# to upscale images to at least (512 x 512)

upscale_stream = MinimumImageDimensions(stream, (256,256),which_sources=('image_features',))

# Take random crops of (32 x 32) from each image
cropped_stream = RandomFixedSizeCrop(
    upscale_stream, (32, 32), which_sources=('image_features',))

# Convert images to [0,1] scale
default_cropped_stream = ScaleAndShift(cropped_stream,  1.0/(255.0), 0., which_sources=('image_features',))
# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
valid_stream = Flatten(
    default_cropped_stream, which_sources=('image_features',))

#################################################################################
# Test Stream
# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream instead of DataStream.default_stream constructor we return
# our images exactly as is.
stream = DataStream(
    test_set,
    iteration_scheme=ShuffledScheme(train_set.num_examples, batch_size)
)

# Our images are of different sizes, so we'll use a Fuel transformer
# to upscale images to at least (512 x 512)

upscale_stream = MinimumImageDimensions(stream, (256,256),which_sources=('image_features',))

# Take random crops of (32 x 32) from each image
cropped_stream = RandomFixedSizeCrop(
    upscale_stream, (32, 32), which_sources=('image_features',))

# Convert images to [0,1] scale
default_cropped_stream = ScaleAndShift(cropped_stream,  1.0/(255.0), 0., which_sources=('image_features',))
# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
test_stream = Flatten(
    default_cropped_stream, which_sources=('image_features',))


#from fuel.server import start_server

#start_server(train_stream)

import timeit
start_time = timeit.default_timer()
for x,y in train_stream.get_epoch_iterator():
#        print(timeit.default_timer()-start_time)
        start_time = timeit.default_timer()

