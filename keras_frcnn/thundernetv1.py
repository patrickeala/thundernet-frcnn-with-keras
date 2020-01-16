# -*- coding: utf-8 -*-
"""MobileNetv1 model for Keras.

"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras import layers
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Add, ReLU
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
#from keras.engine.topology import get_source_inputs
#from keras.utils import layer_utils
#from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv, PSRoiAlignPooling
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization
import tensorflow as tf

def context_enhancement_module(x1, x2, x3, size=20, name="cem_block"):
    
    #C4 output subjected to pointwise conv2d
    x1 = layers.Conv2D(256, (1,1),
                      activation='relu',
                      padding='same',
                      strides=1,
                      use_bias=True,
                      name='{}/c4_lat'.format(name))(x1)
    #print("x1:", x1.shape)
    #C5 output resized to [size, size] and passed through pointwise conv2d
    x2 = layers.Lambda(lambda img: tf.image.resize_bilinear(img, [size, size],
                                                        align_corners=True,
                                                        name='{}/c5_resize'.format(name)))(x2)
    
    x2 = layers.Conv2D(256, (1,1),
                      activation='relu',
                      padding='same',
                      strides=1,
                      use_bias=True,
                      name='{}/c5_lat'.format(name))(x2)
    #print("x2:", x2.shape)
    #C_glb is broadcasted (added to zeros of [size, size, in_channels])
    #result is then passed through pw_conv2d
    zero = K.zeros((1, size, size, 512))
    x3 = layers.Lambda(lambda img: layers.add([img, zero]))(x3)
    
    x3 = layers.Conv2D(256, (1,1),
                      activation='relu',
                      padding='same',
                      strides=1,
                      use_bias=True,
                      name='{}/c_glb_lat'.format(name))(x3)
    #print("x3:", x3.shape)
    return layers.add([x1, x2, x3])

def spatial_attention_module(base_layers):
    channel_axis = 3
    #Spatial Attention Module (SAM)
    x = layers.Conv2D(256, (1,1),
                      activation='relu',
                      padding='same',
                      strides=1,
                      use_bias=True,
                      name='sam/conv1x1')(base_layers)
    x = layers.BatchNormalization(axis=channel_axis, name="sam/bn")(x)
    x = layers.Lambda(K.sigmoid)(x)
    x = layers.multiply([x, base_layers])
    
    return x
    
def get_weight_path():
    
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for VGG with theano backend')
        return
    else:
        return './pretrain/mobilenet_1_0_224_tf.h5'


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16 # there is 4 strides.

    return get_output_length(width), get_output_length(height)    

def nn_base(input_tensor=None, trainable=False):


     #Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    
    # for testing..
    alpha = 1
    depth_multiplier = 1
    
    # need this layer to pass the input image size
    x = layers.ZeroPadding2D((3, 3))(img_input)
    
    #Input: 320x320
    #Output = 160x160
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    
    #Output 160x160 (Stride 1)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    
    #Output = 80x80
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    
    #Output = 40x40
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    
    #5x stride1 dw-separable, Output = 20x20
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    c4 = x
    
    #Output = 10x10
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=999)
    c5 = x
    
    #Obtain c_glb
    x = GlobalAveragePooling2D(name="features/final_pool")(x)
    c_glb = x
    
    #CEM
    x = context_enhancement_module(x1=c4,
                                   x2=c5,
                                   x3=c_glb,
                                   size=20)
    
    return x

def rpn(base_layers, num_anchors):

    x = layers.DepthwiseConv2D((5,5),
                               activation='relu',
                               padding='same',
                               depth_multiplier=1,
                               strides=1,
                               use_bias=False,
                               name='rpn/conv5x5')(base_layers)
    x = layers.Conv2D(256, (1,1),
                      activation='relu',
                      padding='same',
                      strides=1,
                      use_bias=True,
                      name='rpn/conv1x1')(x)
    
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    
    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 7
        #alpha=5
        input_shape = (num_rois,7,7,512)
    elif K.backend() == 'theano':
        pooling_regions = 7
        #alpha=5
        input_shape = (num_rois,512,7,7)
        
    x_sam = spatial_attention_module(base_layers)
    
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([x_sam, input_rois])
    #out_roi_pool = PSRoiAlignPooling(pooling_regions, num_rois, alpha)([x_sam, input_rois])
    
    # Flatten conv layer and connect to 2 FC 2 Dropout
    out = TimeDistributed(Flatten())(out_roi_pool)
    out = TimeDistributed(Dense(1024, activation='relu', name='fc'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    # outputs
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    
    return [out_class, out_regr]

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 3 #if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6., name='conv1_relu')(x)

def _conv_block_td(inputs, filters, input_shape, kernel=(3, 3), strides=(1, 1), trainable=True):
    channel_axis = 3 #if backend.image_data_format() == 'channels_first' else -1
    x = TimeDistributed(layers.Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, input_shape=input_shape), name='conv1_td')(inputs)
    x = TimeDistributed(layers.BatchNormalization(axis=channel_axis), name='conv1_bn_td')(x)
    return layers.ReLU(6., name='conv1_relu_td')(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating
            the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 3
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def _depthwise_conv_block_td(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):

    channel_axis = 3
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = TimeDistributed(layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False
                               ),name='conv_dw_td_%d' % block_id)(x)
    x = TimeDistributed(layers.BatchNormalization(
        axis=channel_axis), name='conv_dw_td_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_td_%d_relu' % block_id)(x)

    x = TimeDistributed(layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1)),
                      name='conv_pw_td_%d' % block_id)(x)
    x = TimeDistributed(layers.BatchNormalization(axis=channel_axis),
                                  name='conv_pw_rd_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_td_%d_relu' % block_id)(x)

