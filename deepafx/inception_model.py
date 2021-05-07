#*************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
#**************************************************************************/
"""
For the Inception model, please also cite:

Disentangled Multidimensional Metric Learning for Music Similarity
Jongpil Lee, Nicholas J. Bryan, Justin Salamon, Zeyu Jin, and Juhan Nam.
Proceedings of the International Conference on Acoustics, Speech, and Signal Processing (ICASSP). IEEE, 2020.

"""

import numpy as np
import itertools
import json

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Multiply,
                                     Dense, Activation, Input, dot, Flatten, Lambda, Embedding, Dropout,
                                     Conv2D, MaxPool2D, GlobalAvgPool2D, concatenate, Reshape,
                                     Cropping1D, Cropping2D, Average, Add, AveragePooling2D,
                                     SeparableConv1D, SeparableConv2D,
                                     )
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.initializers import Constant, constant
from tensorflow.keras.activations import softmax
import tensorflow as tf
from tensorflow.keras.layers import Layer


import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import constant

def get_example_model(num_frames, dafx_params):
    """Example usage"""

    params = {}
    params['num_frames'] = num_frames #129 # number of melspectrogram frames
    params['num_classes'] = dafx_params # number of output classes to predict
    params['convtype'] = '2d' # Type of conv layer: 1d, 2d
    params['num_features'] = 6 # number of feature maps/channels for first layer = 2^num_features. (e.g. 2^6 = 64)
    params['paramfix'] = 0 # For comparing 1d, 2d, this will try to make the model parameters equal 
    params['convlayer'] = 'conv' # conv or mobile (separateble conv)
    params['pooltype'] = 'max' # max 
    params['inputnorm'] = 'batchnorm' # batch norm on input
    params['regulkernel'] = '0' # Conv kernel regularization 
    params['batchnorm'] = '1' # Use batchnorm or not
    params['blocktype'] = 'inception' # block type
    params['repeatblock'] = 2 # Extra blocks (always a stride block + extra repeat blocks)
    params['activation'] = 'relu' # Non linear activation type
    params['num_blocks'] = 7 # total number of blocks (e.g. 7 = 1 conv + 6 inception)
    model, backbone = mainmodel(params)
    return model, backbone



def get_args_from_params(params):
    """ Convert dictionary of parameters to struct with param members
    """
    class obj(object):
        def __init__(self, dict_):
            self.__dict__.update(dict_)
    return json.loads(json.dumps(params), object_hook=obj)


def inception_block(x, num_features, fp_length, i, name, Conv_fn, Mpool_fn, args):
    """Block for Inception models."""
    xin = x
    branch1x1 = basic_block(xin, 64, 1, i, name+'1x1', Conv_fn, Mpool_fn, args)
    branch5x5 = basic_block(xin, 48, 1, i, name+'5x5_1', Conv_fn, Mpool_fn, args)
    branch5x5 = basic_block(branch5x5, 64, 5, i, name+'5x5_2', Conv_fn, Mpool_fn, args)
    branch3x3dbl = basic_block(xin, 64, 1, i, name+'3x3dbl_1', Conv_fn, Mpool_fn, args)
    branch3x3dbl = basic_block(branch3x3dbl, 96, 3, i, name+'3x3dbl_2', Conv_fn, Mpool_fn, args)
    branch3x3dbl = basic_block(branch3x3dbl, 96, 3, i, name+'3x3dbl_3', Conv_fn, Mpool_fn, args)

    if args.convtype == '1d':
        Avgpool_fn = AveragePooling1D
    else:
        Avgpool_fn = AveragePooling2D

    branch_pool = Avgpool_fn(pool_size=3, strides=1, padding='same', name=f'{name}_branchpool')(xin)

    branch_pool = basic_block(branch_pool, 32, 1, i, name+'_branchpool2', Conv_fn, Mpool_fn, args)

    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name=f'{name}_mixed')
    return x


def basic_block(x, num_features, fp_length, i, name, Conv_fn, Mpool_fn, args):
    """Basic convolution block."""
    if not isinstance(num_features, list):
        tmp_feat = num_features
        tmp_pool = fp_length
        num_features = [None] * (i+1)
        fp_length = [None] * (i+1)
        num_features[i] = tmp_feat
        fp_length[i] = tmp_pool

    x = Conv_fn(num_features[i], fp_length[i], padding='same', use_bias=True, 
                kernel_initializer='he_uniform', 
                kernel_regularizer=l2(args.regulkernel), 

                name=f'{name}_conv1d')(x)

    # Whether to use batchnorm or not.
    if args.batchnorm == 1:
        x = BatchNormalization(name=f'{name}_bn')(x)

    x = Activation('relu', name=f'{name}_relu')(x)

    return x

def inception_block_stride(x, num_features, fp_length, pool_size, i, name, Conv_fn, Mpool_fn, args):
    """Block for Inception models."""
    xin = x
    branch3x3 = basic_block_stride(xin, 48, 1, pool_size, i, name+'3x3', Conv_fn, Mpool_fn, args)
    branch3x3dbl = basic_block(xin, 64, 1, i, name+'1x1dbl_1', Conv_fn, Mpool_fn, args)
    branch3x3dbl = basic_block(branch3x3dbl, 96, 3, i, name+'3x3dbl_2', Conv_fn, Mpool_fn, args)
    branch3x3dbl = basic_block_stride(branch3x3dbl, 96, 3, pool_size, i, name+'3x3dbl_3', Conv_fn, Mpool_fn, args)
    #TODO: what about 5x5 block
    branch_pool = Mpool_fn(pool_size=3, strides=pool_size, padding='same', name=f'{name}_branchpool')(xin)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name=f'{name}_mixed')
    return x


def basic_block_stride(x, num_features, fp_length, pool_size, i, name, Conv_fn, Mpool_fn, args):
    """Basic convolution block."""
    if not isinstance(num_features, list):
        tmp_feat = num_features
        tmp_pool = fp_length
        num_features = [None] * (i+1)
        fp_length = [None] * (i+1)
        num_features[i] = tmp_feat
        fp_length[i] = tmp_pool

    x = Conv_fn(num_features[i], fp_length[i], padding='same', use_bias=True, strides=pool_size, 
                kernel_initializer='he_uniform', 
                kernel_regularizer=l2(args.regulkernel), 

                name=f'{name}_conv1d')(x)

    # Whether to use batchnorm or not.
    if args.batchnorm == 1:
        x = BatchNormalization(name=f'{name}_bn')(x)

    x = Activation('relu', name=f'{name}_relu')(x)

    return x


def inception_block_mp(x, num_features, fp_length, pool_size, i, name, Conv_fn, Mpool_fn, args):
    print('inception_block_mp')
    """Block for Inception models."""
    xin = x
    branch3x3 = basic_block_mp(xin, 48, 1, pool_size, i, name+'3x3', Conv_fn, Mpool_fn, args)
    branch3x3dbl = basic_block(xin, 64, 1, i, name+'3x3dbl_1', Conv_fn, Mpool_fn, args)
    branch3x3dbl = basic_block(branch3x3dbl, 96, 3, i, name+'3x3dbl_2', Conv_fn, Mpool_fn, args)
    branch3x3dbl = basic_block_mp(branch3x3dbl, 96, 3, pool_size, i, name+'3x3dbl_3', Conv_fn, Mpool_fn, args)
    branch_pool = Mpool_fn(pool_size=3, strides=pool_size, padding='same', name=f'{name}_branchpool')(xin)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name=f'{name}_mixed')
    return x


def basic_block_mp(x, num_features, fp_length, pool_size, i, name, Conv_fn, Mpool_fn, args):
    print('basic_block_mp')
    """Basic convolution block."""
    if not isinstance(num_features, list):
        tmp_feat = num_features
        tmp_pool = fp_length
        num_features = [None] * (i+1)
        fp_length = [None] * (i+1)
        num_features[i] = tmp_feat
        fp_length[i] = tmp_pool

    x = Conv_fn(num_features[i], fp_length[i], padding='same', use_bias=True, strides=1, 
                kernel_initializer='he_uniform', 
                kernel_regularizer=l2(args.regulkernel), 

                name=f'{name}_conv1d')(x)

    # Whether to use batchnorm or not.
    if args.batchnorm == 1:

        x = BatchNormalization(name=f'{name}_bn')(x)

    x = Activation('relu', name=f'{name}_relu')(x)
    x = Mpool_fn(pool_size=pool_size, padding='same', name=f'{name}_pool')(x)

    return x


def crop(start, end, name=""):
    def func(x):
        if True:
            return x[:,start: end]

    return Lambda(func,name=name)

def mainmodel(params):
    """ Model used for the following paper and follow work:
    
        "Disentangled Multidimensional Metric Learning For Music Similarity." 
        J. Lee, N. J. Bryan, J. Salamon, Z. Jin J. Nam 
        IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Barcelona, Spain. May, 2020. 
    
    Input audio (melspectrogram prefered) -> Conv layer -> six inception blocks. 
    One onception block consists of two inception modeles - one for temporal dimension reduction and one for feature map reduction.
    """

    args = get_args_from_params(params)
    
    num_frames = args.num_frames


    # Base model input. 
    if args.convtype == '1d':
        # Base model input.
        x_in = Input(shape = (num_frames, 128), name='x_in')
    elif args.convtype == '2d':
        # Base model input.
        x_in = Input(shape = (num_frames, 128, 1), name='x_in')
    else:
        assert 0

    # Base model parameters.
    num_features = []
    fp_length = []

    pool_size = 2 #5
    num_blocks = args.num_blocks

    # Initial layer.
    # Convtype.
    if args.convtype == '1d':
        fp_size = 5
    elif args.convtype == '2d':
        fp_size = (5,5)
    else:
        assert 0
        
    num_features.append(2)
    fp_length.append(fp_size)

    # Middle layers.
    for i in range(num_blocks - 1):
        # Convtype.
        if args.convtype == '1d':
            fp_size = 3
        elif args.convtype == '2d':
            fp_size = (3,3)
        else:
            assert 0
        
        num_features.append(2)
        fp_length.append(fp_size)

    # Last layer convtype
    if args.convtype == '1d':
        fp_size = 1
    elif args.convtype == '2d':
        fp_size = (1,1)
    else:
        assert 0
        
    num_features.append(2)
    fp_length.append(fp_size)

    # Multiply hidden layer size by args.num_features.
    num_features = [x ** args.num_features for x in num_features]
    num_features = [2 * num_feature for num_feature in num_features]
    num_features[0] = int(num_features[0]/2)
    num_features[-1] = 2 * num_features[-1]

    # If parameter fix experiment, multiply these numbers. 
    if args.paramfix == 1:
        if args.convtype == '1d':
            num_features_proportion = 1.6
        elif args.convtype == '2d':
            num_features_proportion = 1.0
        else:
            assert 0
        num_features[:-1] = [int(num_features_proportion * num_feature) for num_feature in num_features[:-1]]


    # Convtype.
    if args.convtype == '1d':
        # Conv layer.
        if args.convlayer == 'conv':
            Conv_fn = Conv1D
        elif args.convlayer == 'mobile':
            Conv_fn = SeparableConv1D
        # Pool layer.
        if args.pooltype == 'max':
            Mpool_fn = MaxPool1D
            Gavgpool_fn = GlobalAvgPool1D


    elif args.convtype == '2d':

        # Conv layer.
        if args.convlayer == 'conv':
            Conv_fn = Conv2D
        elif args.convlayer == 'mobile':
            Conv_fn = SeparableConv2D
        # Pool layer.
        if args.pooltype == 'max':
            Mpool_fn = MaxPool2D
            Gavgpool_fn = GlobalAvgPool2D


    # Input normalization method.
    if args.inputnorm == 'batchnorm':
        x = BatchNormalization(name='input_batchnorm')(x_in)
        x = Conv_fn(num_features[0], fp_length[0], padding='same', use_bias=True, 
                kernel_regularizer=l2(args.regulkernel),
                kernel_initializer='he_uniform', name=f'block0_conv1d')(x)       
    elif args.inputnorm == 'batchrenorm':
        x = BatchNormalization(renorm=True, name='input_batchrenorm')(x_in)
        x = Conv_fn(num_features[0], fp_length[0], padding='same', use_bias=True, 
                kernel_regularizer=l2(args.regulkernel),
                kernel_initializer='he_uniform', name=f'block0_conv1d')(x)
    elif args.inputnorm == 'norm':
        x = Conv_fn(num_features[0], fp_length[0], padding='same', use_bias=True,
                kernel_regularizer=l2(args.regulkernel),
                kernel_initializer='he_uniform', name=f'block0_conv1d')(x_in)

    # Whether to use batchnorm or not.
    # if 1:
    if args.batchnorm == 1:
        x = BatchNormalization(name=f'block0_bn')(x)
    x = Activation('relu', name=f'block0_relu')(x)
    x = Mpool_fn(pool_size=pool_size, padding='valid', name=f'block0_pool')(x)

    # Building block.
    if args.blocktype == 'basic':
        block_fn = basic_block
        block_fn_stride = basic_block_stride
        block_fn_mp = basic_block_mp
    elif args.blocktype == 'inception':
        block_fn = inception_block
        block_fn_stride = inception_block_stride
        block_fn_mp = inception_block_mp
    else:
        assert 0, "invalid block type"


    if args.inputnorm == 'batchnorm':
        x_in_middle = BatchNormalization(name='input_batchnorm')(x_in)
    elif args.inputnorm == 'batchrenorm':
        print('Applying Batch RENorm to input!')
        x_in_middle = BatchNormalization(renorm=True, name='input_batchrenorm')(x_in)
    elif args.inputnorm == 'norm':
        print('Applying fixed norm to input!')
        x_in_middle = x_in
    else:
        assert "unknown batch norm type"

    x = Conv_fn(num_features[0], fp_length[0], padding='same', use_bias=True,
            kernel_regularizer=l2(args.regulkernel),
            kernel_initializer='he_uniform', name=f'block0_conv1d')(x_in_middle)

    # Whether to use batchnorm or not.
    if args.batchnorm == 1:
        x = BatchNormalization(name=f'block0_bn')(x)
    x = Activation('relu', name=f'block0_relu')(x)
    x = Mpool_fn(pool_size=pool_size, padding='valid', name=f'block0_pool')(x)

    # Backend (basic building block).
    for i in range(1,num_blocks):

        pool_size = 2 

        # Time-wise pool management.
        if args.num_frames == 64:
            not_list = [num_blocks-1]
            if i in not_list:
                pool_size = (1,2)
        elif args.num_frames == 32:
            not_list = [num_blocks-1, num_blocks-2]
            if i in not_list:
                pool_size = (1,2)
        elif args.num_frames == 16:
            not_list = [num_blocks-1, num_blocks-2, num_blocks-3]
            if i in not_list:
                pool_size = (1,2)
        elif args.num_frames == 8:
            not_list = [num_blocks-1, num_blocks-2, num_blocks-3, num_blocks-4]
            if i in not_list:
                pool_size = (1,2)


        # Stride blocks.
        x = block_fn_stride(x, 
                            num_features, 
                            fp_length, 
                            pool_size, 
                            i, 
                            f'block{i}-stride', 
                            Conv_fn, 
                            Mpool_fn, 
                            args)

        # Repeat blocks.
        for j in range(args.repeatblock - 1):
            x = block_fn(x, 
                         num_features, 
                         fp_length, 
                         i, 
                         f'block{i}-{j}', 
                         Conv_fn, 
                         Mpool_fn, 
                         args)


    # Embedding layer
    x = Conv_fn(num_features[-1], 
                fp_length[-1], 
                padding='same', 
                use_bias=True, 
                kernel_initializer='he_uniform', 
                kernel_regularizer=l2(0), 
                name=f'embed_conv1d')(x)

    # Activation on embedding layer.
    if args.activation == 'relu':
        x = Activation('relu', name=f'embed_relu')(x)

    # Global avg pool.
    x = Gavgpool_fn()(x)

    # Create base model.
    base_model = Model(inputs=[x_in], outputs=[x], name='base_model')

    # Classification model input.
    if args.convtype == '1d':
        anchor = Input(shape = (num_frames, 128), name='anchor_input')

    elif args.convtype == '2d':
        anchor = Input(shape = (num_frames, 128, 1), name='anchor_input')

    # Feedforward to embedding layer.
    embed_anchor = base_model(anchor)

    # Normalize
    embed_anchor = Lambda(lambda x: K.l2_normalize(x, axis=1))(embed_anchor)

    # Classification layer.
    output_list = Dense(args.num_classes, activation='sigmoid')(embed_anchor)
    model = Model(inputs = [anchor], outputs = output_list, name='inception')

    return model, base_model
