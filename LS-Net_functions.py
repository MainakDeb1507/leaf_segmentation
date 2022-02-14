def atrous_conv(img, filters, kernel = (1,1) ,strides = 1, rate = (1,1,1), pad = (1,1,1)):
    r1, r2, r3 = rate
    p1, p2, p3 = pad
    
    frist = tf.keras.layers.SeparableConv2D(filters, kernel_size = (1,1), padding= 'same', strides=strides)(img)
    
    s = tf.keras.layers.ZeroPadding2D(padding = p1)(img)
    second = tf.keras.layers.SeparableConv2D(filters, kernel_size = (3,3), strides=strides, dilation_rate = r1)(s)
    
    t = tf.keras.layers.ZeroPadding2D(padding=(p2, p2))(img)
    third = tf.keras.layers.SeparableConv2D(filters, kernel_size = (3,3), strides=strides, dilation_rate = r2)(t)
    
    f = tf.keras.layers.ZeroPadding2D(padding=(p3, p3))(img)
    fourth = tf.keras.layers.SeparableConv2D(filters, kernel_size = (3,3), strides=strides, dilation_rate = r3)(f)
    
    pool = tf.keras.layers.SeparableConv2D(filters, kernel_size = (1,1))(img)
    pool = tf.keras.layers.MaxPooling2D(3,strides=1,padding='same')(pool)
    
    mixed = tf.keras.layers.Concatenate(axis=3)([frist,second ,third ,fourth ,pool])
    return mixed

def normalize(x):
    return K.relu(x, threshold = -1.5)
  
  def relu6(x):
    return K.relu(x, max_value=6.0)
  
# Seperable Convulution Block (Activation->Batch-Norm->Sep_conv)

def sep_conv_block(inputs, filters, kernel, strides = 1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = tf.keras.layers.Activation(relu6)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.SeparableConv2D(filters, kernel, padding='same', strides=strides)(x)
    
    return x
  
# Convulution Block (Sep_conv->Batch-Norm->Activation)

def conv_block(inputs, filters, kernel, strides = 1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation(relu6)(x)
    
    return x

def bottleneck_for_encoder(inputs, filters, kernel, t, alpha, s, r=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
 
    tchannel = K.int_shape(inputs)[channel_axis] * t
    cchannel = int(filters * alpha)

    x = conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation(relu6)(x)

    x = tf.keras.layers.Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    
    if r:
        x = tf.keras.layers.Add()([x, inputs])

    return x

def bottleneck_for_decoder(inputs, filters, kernel, t, alpha, s, r=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = conv_block(inputs, filters*t, (1, 1), (1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation(relu6)(x)

    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    
    if r:
        x = tf.keras.layers.Add()([x, inputs])

    return x

def inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n, encoder = True, unpool = False):
    if encoder:
        x = bottleneck_for_encoder(inputs, filters, kernel, t, alpha, strides)
        for i in range(1, n):
            x = bottleneck_for_encoder(x, filters, kernel, t, alpha, 1, True)
        return x
    else:
        x = bottleneck_for_decoder(inputs, filters, kernel, t, alpha, s = 1)
        for i in range(1, n):
            x = bottleneck_for_decoder(x, filters, kernel, t, alpha, 1, True)
        if unpool:  
            x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)   
        x = bottleneck_for_decoder(x, filters, kernel, t, alpha, s = 1)
        return x



def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
