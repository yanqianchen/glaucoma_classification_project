import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Keras后端配置
K.set_image_data_format('channels_last')

# !! 核心升级 !!: 定义两个可被安全序列化的自定义层
class MeanAcrossChannel(layers.Layer):
    def call(self, inputs):
        return K.mean(inputs, axis=-1, keepdims=True)

class MaxAcrossChannel(layers.Layer):
    def call(self, inputs):
        return K.max(inputs, axis=-1, keepdims=True)


def cbam_block(input_feature, name='cbam_block', ratio=8):
    """
    一个完整的、可被调用的、可被安全序列化的CBAM模块。
    """
    channel_axis = -1
    channel = input_feature.shape[channel_axis]
    
    # --- 通道注意力模块 (与V1完全相同) ---
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', name=name+'_channel_avg_dense_1')
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros', name=name+'_channel_avg_dense_2')
    avg_pool = layers.GlobalAveragePooling2D(name=name+'_channel_avg_pool')(input_feature)
    avg_pool = layers.Reshape((1, 1, channel), name=name+'_channel_avg_reshape')(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = layers.GlobalMaxPooling2D(name=name+'_channel_max_pool')(input_feature)
    max_pool = layers.Reshape((1, 1, channel), name=name+'_channel_max_reshape')(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    cbam_feature = layers.Add(name=name+'_channel_add')([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid', name=name+'_channel_sigmoid')(cbam_feature)
    channel_attention = layers.multiply([input_feature, cbam_feature], name=name+'_channel_attention')
    
    # --- 空间注意力模块 (!! 告别Lambda !!) ---
    kernel_size = 7
    # !! 使用我们全新的、可被安全序列化的自定义层 !!
    avg_pool = MeanAcrossChannel(name=name+'_spatial_avg_pool')(channel_attention)
    max_pool = MaxAcrossChannel(name=name+'_spatial_max_pool')(channel_attention)
    
    concat = layers.Concatenate(axis=channel_axis, name=name+'_spatial_concat')([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False, name=name+'_spatial_conv')(concat)
    spatial_attention = layers.multiply([channel_attention, cbam_feature], name=name+'_spatial_attention')
    
    return spatial_attention