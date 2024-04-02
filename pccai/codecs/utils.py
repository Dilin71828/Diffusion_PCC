# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Utilities related to point cloud codec

import torch
import numpy as np

# Import all the codecs to be used
from pccai.codecs.grasp_codec import GeoResCompressionCodec
from pccai.codecs.diffusion_codec import DiffusionGeoResCompressionCodec


# List the all the codecs in the following dictionary 
codec_classes = {
    'grasp_codec': GeoResCompressionCodec,
    'diffusion_codec': DiffusionGeoResCompressionCodec,
}

def get_codec_class(codec_name):
    codec = codec_classes.get(codec_name.lower(), None)
    assert codec is not None, f'codec class "{codec_name}" not found, valid codec classes are: {list(codec_classes.keys())}'
    return codec
