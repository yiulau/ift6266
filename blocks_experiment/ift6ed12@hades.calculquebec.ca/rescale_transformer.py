from __future__ import division
import math

import numpy
from PIL import Image
from six import PY3

from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer


class rescale_transformer(SourcewiseTransformer, ExpectsAxisLabels):
    """Resize (lists of) images to square images.
    Parameters
    ----------
    data_stream : instance of :class:`AbstractDataStream`
        The data stream to wrap.
    output_shape : 2-dimensional tuple or "no_target" (set by default)
        The desired `(height, width)` dimensions every image must have.
        Every image is rescaled to have the desired dimensions. If set to 'no_target' 
        the image is passed as is. When this parameter is set no other rescaling is done and 
        the output dimensions are fixed. 
    max_height_width_ratio : double
        Every image must have max(height/width, width/height) <= max_height_width_ratio
        Must be > 1 to have any effect. Purpose is to guard against images that are not "square" enough.
        Recommended value : 2
    minimum_dim_len : integer
        The shortest dimension of every image must be greater than this value.
        Recommended value : 256
    resample : str, optional
        Resampling filter for PIL to use to upsample any images requiring
        it. Options include 'nearest' (default), 'bilinear', and 'bicubic'.
        See the PIL documentation for more detailed information.
    Notes
    -----
    This transformer expects stream sources returning individual images,
    represented as 2- or 3-dimensional arrays, or lists of the same.
    The format of the stream is unaltered.
    """
    def __init__(self, data_stream, max_height_width_ratio, minimum_dim_len, output_shape='no_target', resample='nearest',
                 **kwargs):
        self.output_shape = output_shape
        self.max_heigh_width_ratio = max_height_width_ratio
        self.minimum_dim_len = minimum_dim_len
        try:
            self.resample = getattr(Image, resample.upper())
        except AttributeError:
            raise ValueError("unknown resampling filter '{}'".format(resample))
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(rescale_transformer, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, batch, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return [self._example_transform(im, source_name) for im in batch]

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        return self._example_transform(example, source_name)

    def _example_transform(self, example, _):
        if example.ndim > 3 or example.ndim < 2:
            raise NotImplementedError
        min_dim_len = self.minimum_dim_len
        max_height_width_ratio = self.max_heigh_width_ratio
        original_height, original_width = example.shape[-2:]
        original_min_dim_len = min(original_height, original_width)
        original_hw_ratio = max(original_height/original_width, original_width/original_height)
        if not self.output_shape == 'no_target':
            dt = example.dtype
            # If we're dealing with a colour image, swap around the axes
            # to be in the format that PIL needs.
            if example.ndim == 3:
                im = example.transpose(1, 2, 0)
            else:
                im = example
            im = Image.fromarray(im)
            output_height, output_width = self.output_shape
            im = numpy.array(im.resize((output_width, output_height))).astype(dt)
            # If necessary, undo the axis swap from earlier.
            if im.ndim == 3:
                example = im.transpose(2, 0, 1)
            else:
                example = im
            
            
        elif original_min_dim_len < min_dim_len or  original_hw_ratio > max_height_width_ratio:
            dt = example.dtype
            # If we're dealing with a colour image, swap around the axes
            # to be in the format that PIL needs.
            if example.ndim == 3:
                im = example.transpose(1, 2, 0)
            else:
                im = example
            im = Image.fromarray(im)
            width, height = im.size
            # If the original image is not "square" enough we scale up the short side
            # to match the longer side.
            hmultiplier = 1
            wmultiplier = 1
            if original_hw_ratio > max_height_width_ratio:
                hmultiplier = max(1, original_width / original_height)
                wmultiploer = max(1, original_height / original_width)
                height = int(math.ceil(height * hmultiplier))
                width = int(math.ceil(width * wmultiplier))
            # After first (potential) scaling see if the shortest side is still smaller 
            # than min_dim_len
            if min(height,width) < min_dim_len:
                if height < min_dim_len:
                    hmultiplier = min_dim_len / height
                if width < min_dim_len:
                    wmultiplier = min_dim_len / width
                height = int(math.ceil(height * hmultiplier))
                width = int(math.ceil(width * wmultiplier))             
            im = numpy.array(im.resize((width, height))).astype(dt)
            # If necessary, undo the axis swap from earlier.
            if im.ndim == 3:
                example = im.transpose(2, 0, 1)
            else:
                example = im
        return example

