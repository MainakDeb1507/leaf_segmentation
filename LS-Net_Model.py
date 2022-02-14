from LS-Net_functions import *
def Create_Model(input_shape = (224,224,3), alpha=1.0, multiplier = 2):
  
  first_filters = _make_divisible(32 * alpha, 8)
  inputs = tf.keras.layers.Input(shape=input_shape)
  # Backbone
  backbone_out, x7, x14, x28, x56, x112, x224 = MobileNetv2(inputs)
  
  x = tf.keras.layers.SeparableConv2D(864, (1, 1), padding='same')(backbone_out)
  x = tf.keras.layers.BatchNormalization()(x)

  concat_1 = tf.keras.layers.Concatenate()([x, x7])

  x = inverted_residual_block(concat_1, 160, (3, 3), t=6, alpha=alpha, strides=1, n=1, encoder = False)   ##320

# Block 1
  x = inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3, encoder = False, unpool = True)
  concat_2 = tf.keras.layers.Concatenate()([x, x14])


# Block 2
  x = inverted_residual_block(concat_2, 64, (3, 3), t=6, alpha=alpha, strides=1, n=3, encoder = False)


# Block 3
  x = inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=1, n=4, encoder = False, unpool = True)
  concat_3 = tf.keras.layers.Concatenate()([x,x28])

# Block 4
  x = inverted_residual_block(concat_3, 24, (3, 3), t=6, alpha=alpha, strides=1, n=3, encoder = False, unpool = True)
  concat_4 = tf.keras.layers.Concatenate()([x, x56])

# Block 5
  x = inverted_residual_block(concat_4, 16, (3, 3), t=6, alpha=alpha, strides=1, n=2, encoder = False, unpool = True)
  concat_5 = tf.keras.layers.Concatenate()([x, x112])

# Block 6
  x_1 = inverted_residual_block(concat_5, 32, (3, 3), t=1, alpha=alpha, strides=1, n=1, encoder = False)

  x = sep_conv_block(x_1, 32, kernel = (3, 3), strides = 1)
  x = tf.keras.layers.Conv2DTranspose(24, (2, 2), strides=(2, 2), padding='same')(x)   
  concat_6 = tf.keras.layers.Concatenate()([x, x224])

  x = sep_conv_block(concat_6, 24, kernel = (3, 3))
  out = sep_conv_block(x, 24, kernel = (3, 3))

  out = tf.keras.layers.Activation(normalize)(out)

  out = tf.keras.layers.Conv2D(1, (1, 1))(out)
  seg_output = tf.keras.layers.Activation('sigmoid', name="seg_img")(out)
 
  model = tf.keras.Model(inputs=[inputs], outputs=[seg_output])
  
  return model
