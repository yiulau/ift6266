from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers.image import RandomFixedSizeCrop
from fuel.transformers import Flatten, ScaleAndShift, Cast
from fuel.server import start_server
from rescale_transformer import rescale_transformer

def get_datastream(random_cropsize,max_height_width_ratio,
             minimum_dim_len,batchsize,require_stream,rescale_outputsize=None):
    """ Returns a datastream with all the necessary transforms applied.
    
    Parameters
    ----------
    random_cropsize : tuple
        The size of the random crop as applied by RandomFixedSizeCrop
    max_heigh_width_ratio : float
        See rescale_transformer.
    minimum_dim_len : int
        See rescale_transformer.
    batchsize : int
        Batch size to be the minibatch to be returned by the datastream object.
    require_stream: str
        One of 'train','valid' or 'test'. Specifies which part of data to query.
    rescale_outputsize: tuple, optional
        If this argument is not set to none, the arguments max_height_width_ratio
        and minimum_dim_len are ignored. For details see rescale_transformer.
        
    
    """ 
    if require_stream == 'train':
        dataset = DogsVsCats(('train',),subset=slice(0,20000))
    if require_stream == 'valid':
        dataset = DogsVsCats(('train',),subset=slice(20000,22500))
    if require_stream == 'test':
        dataset = DogsVsCats(('train',),subset=slice(22500,25000))
        
    stream = DataStream(dataset,iteration_scheme=SequentialScheme(dataset.num_examples,batchsize))
    
    if rescale_outputsize == None:
        rescale_outputsize = 'no_target'
    

    rescale_stream = rescale_transformer(stream,max_height_width_ratio,minimum_dim_len,rescale_outputsize,
                                      which_sources=('image_features',))
    cropped_stream = RandomFixedSizeCrop(rescale_stream,random_cropsize,which_sources=('image_features',))
    sd_cropped_stream = ScaleAndShift(cropped_stream,1.0/255.0,0.,which_sources=('image_features',))
    final_stream = Cast(sd_cropped_stream, dtype='float32',which_sources=('image_features',))

    

    return (final_stream)
        
        
