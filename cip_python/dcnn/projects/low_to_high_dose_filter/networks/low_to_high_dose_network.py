import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, ZeroPadding2D, ZeroPadding3D
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K

from cip_python.dcnn.logic import Network, Utils


class LowToHighDoseNetwork(Network):
    """
            The refiner network, R, is a residual network (ResNet). It modifies the synthetic image on a pixel level,
            rather than holistically modifying the image content, preserving the global structure and annotations.
    """
    def __init__(self, img_shape, gf=32, df=64, lambda_cycle=10, use_lsgan=False): # lambda_id=0.1
        """
                Constructor
                :param patch_size: int
                :param nb_input_channels: int
        """
        # Calculate input/output sizes based on the patch sizes
        # Assume isometric patch size
        xs_sizes = img_shape
        ys_sizes = None

        # Use parent constructor
        Network.__init__(self, xs_sizes, ys_sizes)
        self.channels = 1

        if len(img_shape) == 2:
            self.img_shape = (img_shape[0], img_shape[1], self.channels)
        else:
            self.img_shape = (img_shape[0], img_shape[1], img_shape[2], self.channels)

        self.lambda_cycle = lambda_cycle
        self.use_lsgan = use_lsgan
        # self.lambda_id = lambda_id
        self.gf = gf
        self.df = df
        self.conv_init = tf.keras.initializers.RandomNormal(0, 0.02)
        self.gamma_init = tf.keras.initializers.RandomNormal(1., 0.02)

        self._expected_input_values_range_ = (0.0, 1.0)

    def generator_network_3d_to_2d(self):
        def down_block(x, f, stride, k_size, use_batchnorm=True):
            x = tf.keras.layers.Conv3D(f, kernel_initializer=self.conv_init, kernel_size=k_size, strides=stride,
                                       use_bias=not use_batchnorm, padding="same")(x)
            if use_batchnorm:
                x = self.batchnorm()(x, training=1)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            return x

        def up_block(x, f, stride, k_size, use_batchnorm=True):
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv3DTranspose(f, kernel_size=k_size, strides=stride, use_bias=not use_batchnorm,
                                                kernel_initializer=self.conv_init)(x)
            if use_batchnorm:
                x = self.batchnorm()(x, training=1)
            return x

        def sr_layer(dimension, start, end):  # Slice-Recovery Layer
            def func(x):
                if dimension == 0:
                    return x[start:end]
                if dimension == 1:
                    return x[:, start:end]
                if dimension == 2:
                    return x[:, :, start:end]
                if dimension == 3:
                    return x[:, :, :, start:end]
                if dimension == 4:
                    return x[:, :, :, :, start:end]

            return tf.keras.layers.Lambda(func)

        inputs = tf.keras.layers.Input(shape=self.img_shape)
        down1 = down_block(inputs, 16, (2, 2, 1), (4, 4, 3), use_batchnorm=False)  # 9 (128, 128, 5, 32)
        down2 = down_block(down1, 32, (2, 2, 1), (4, 4, 3), use_batchnorm=True)  # 9 (64, 64, 5, 64)
        down3 = down_block(down2, 64, (2, 2, 1), (4, 4, 3), use_batchnorm=True)  # 9 (32, 32, 5, 128)
        down4 = down_block(down3, 128, (2, 2, 2), (4, 4, 3), use_batchnorm=True)  # 5 (16, 16, 3, 256)
        down5 = down_block(down4, 128, (2, 2, 2), (4, 4, 1), use_batchnorm=True)  # 3 (8, 8, 2, 256)
        down6 = down_block(down5, 128, (2, 2, 2), (4, 4, 1), use_batchnorm=True)  # 2 (4, 4, 1, 256)
        # down7 = down_block(down6, 256, (2, 2, 2), (4, 4, 1), use_batchnorm=True)  # 1

        up1 = up_block(down6, 128, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up1 = tf.keras.layers.Dropout(0.5)(up1, training=1)
        up1 = tf.keras.layers.Concatenate(axis=-1)([up1, sr_layer(3, 1, 2)(down5)])

        up2 = up_block(up1, 128, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up2 = tf.keras.layers.Dropout(0.5)(up2, training=1)
        up2 = tf.keras.layers.Concatenate(axis=-1)([up2, sr_layer(3, 1, 2)(down4)])

        up3 = up_block(up2, 64, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up3 = tf.keras.layers.Dropout(0.5)(up3, training=1)
        up3 = tf.keras.layers.Concatenate(axis=-1)([up3, sr_layer(3, 2, 3)(down3)])

        up4 = up_block(up3, 32, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up4 = tf.keras.layers.Concatenate(axis=-1)([up4, sr_layer(3, 4, 5)(down2)])

        up5 = up_block(up4, 16, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up5 = tf.keras.layers.Concatenate(axis=-1)([up5, sr_layer(3, 4, 5)(down1)])

        # up6 = up_block(up5, 32, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        # up6 = tf.keras.layers.Concatenate(axis=-1)([up6, sr_layer(3, 4, 5)(down1)])

        up7 = up_block(up5, 1, (2, 2, 1), (2, 2, 1), use_batchnorm=False)
        outputs = tf.keras.layers.Activation('tanh')(up7)

        return tf.keras.models.Model(inputs=inputs, outputs=[outputs])

    def generator_network_2d_to_2d(self):
        def down_block(x, f, stride, k_size, use_batchnorm=True):
            x = tf.keras.layers.Conv3D(f, kernel_initializer=self.conv_init, kernel_size=k_size, strides=stride,
                                       use_bias=not use_batchnorm, padding="same")(x)
            if use_batchnorm:
                x = self.batchnorm()(x, training=1)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
            return x

        def up_block(x, f, stride, k_size, use_batchnorm=True):
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv3DTranspose(f, kernel_size=k_size, strides=stride, use_bias=not use_batchnorm,
                                                kernel_initializer=self.conv_init)(x)
            if use_batchnorm:
                x = self.batchnorm()(x, training=1)
            return x

        inputs = tf.keras.layers.Input(shape=(self.img_shape[0], self.img_shape[1], 1, 1))
        down1 = down_block(inputs, 16, (2, 2, 1), (4, 4, 1), use_batchnorm=False)
        down2 = down_block(down1, 32, (2, 2, 1), (4, 4, 1), use_batchnorm=True)
        down3 = down_block(down2, 64, (2, 2, 1), (4, 4, 1), use_batchnorm=True)
        down4 = down_block(down3, 128, (2, 2, 1), (4, 4, 1), use_batchnorm=True)
        down5 = down_block(down4, 128, (2, 2, 1), (4, 4, 1), use_batchnorm=True)
        down6 = down_block(down5, 128, (2, 2, 1), (4, 4, 1), use_batchnorm=True)
        # down7 = down_block(down6, 256, (2, 2, 1), (4, 4, 1), use_batchnorm=True)

        up1 = up_block(down6, 128, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up1 = tf.keras.layers.Dropout(0.5)(up1, training=1)
        up1 = tf.keras.layers.Concatenate(axis=-1)([up1, down5])

        up2 = up_block(up1, 128, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up2 = tf.keras.layers.Dropout(0.5)(up2, training=1)
        up2 = tf.keras.layers.Concatenate(axis=-1)([up2, down4])

        up3 = up_block(up2, 64, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up3 = tf.keras.layers.Dropout(0.5)(up3, training=1)
        up3 = tf.keras.layers.Concatenate(axis=-1)([up3, down3])

        up4 = up_block(up3, 32, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up4 = tf.keras.layers.Concatenate(axis=-1)([up4, down2])

        up5 = up_block(up4, 16, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        up5 = tf.keras.layers.Concatenate(axis=-1)([up5, down1])

        # up6 = up_block(up5, 32, (2, 2, 1), (2, 2, 1), use_batchnorm=True)
        # up6 = tf.keras.layers.Concatenate(axis=-1)([up6, down1])

        up7 = up_block(up5, 1, (2, 2, 1), (2, 2, 1), use_batchnorm=False)
        outputs = tf.keras.layers.Activation('tanh')(up7)

        return tf.keras.models.Model(inputs=inputs, outputs=[outputs])

    def discriminator_network_3d_to_2d(self):
        def d_layer(layer_input, filters, normalization=True):
            """Discriminator layer"""
            d = self.conv3d(filters, kernel_size=4, strides=(2,2,1), padding="same", use_bias=False)(layer_input)
            if normalization:
                d = self.batchnorm()(d, training=1)
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            return d

        input_ = tf.keras.layers.Input(shape=self.img_shape)
        _ = d_layer(input_, self.df, normalization=False)
        for layer in range(1, 3):
            out_feat = self.df * min(2 ** layer, 8)
            _ = d_layer(_, out_feat)

        out_feat = self.df * min(2 ** 3, 8)
        _ = ZeroPadding3D(1)(_)
        _ = self.conv3d(out_feat, kernel_size=4, use_bias=False)(_)
        _ = self.batchnorm()(_, training=1)
        _ = tf.keras.layers.LeakyReLU(alpha=0.2)(_)

        # output layer
        _ = ZeroPadding3D(1)(_)
        _ = self.conv3d(1, kernel_size=(4, 4, 1), activation="sigmoid" if not self.use_lsgan else None)(_)

        return tf.keras.models.Model(input_, _)

    def discriminator_network_2d_to_2d(self):
        def d_layer(layer_input, filters, normalization=True):
            """Discriminator layer"""
            d = self.conv3d(filters, kernel_size=(4, 4, 1), strides=(2, 2, 1), padding="same", use_bias=False)(layer_input)
            if normalization:
                d = self.batchnorm()(d, training=1)
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            return d

        input_ = tf.keras.layers.Input(shape=(self.img_shape[0], self.img_shape[1], 1, 1))
        _ = d_layer(input_, self.df, normalization=False)
        for layer in range(1, 3):
            out_feat = self.df * min(2 ** layer, 8)
            _ = d_layer(_, out_feat)

        out_feat = self.df * min(2 ** 3, 8)
        _ = ZeroPadding3D((1, 1, 0))(_)
        _ = self.conv3d(out_feat, kernel_size=(4, 4, 1), use_bias=False)(_)
        _ = self.batchnorm()(_, training=1)
        _ = tf.keras.layers.LeakyReLU(alpha=0.2)(_)

        # output layer
        _ = ZeroPadding3D((1, 1, 0))(_)
        _ = self.conv3d(1, kernel_size=(4, 4, 1), activation="sigmoid" if not self.use_lsgan else None)(_)

        return tf.keras.models.Model(input_, _)

    def conv2d(self, f, *a, **k):
        return tf.keras.layers.Conv2D(f, kernel_initializer=self.conv_init, *a, **k)

    def conv3d(self, f, *a, **k):
        return tf.keras.layers.Conv3D(f, kernel_initializer=self.conv_init, *a, **k)

    def batchnorm(self):
        return tf.keras.layers.BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                                  gamma_initializer=self.gamma_init)

    def generator_network_unet(self):
        max_nf = 8 * self.gf

        def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
            # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
            assert s >= 2 and s % 2 == 0
            if nf_next is None:
                nf_next = min(nf_in * 2, max_nf)
            if nf_out is None:
                nf_out = nf_in
            x = self.conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                       padding="same", name='conv_{0}'.format(s))(x)
            if s > 2:
                if use_batchnorm:
                    x = self.batchnorm()(x, training=1)
                x2 = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
                x2 = block(x2, s // 2, nf_next)
                x = tf.keras.layers.Concatenate(axis=-1)([x, x2])
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                                                kernel_initializer=self.conv_init, name='convt.{0}'.format(s))(x)
            x = tf.keras.layers.Cropping2D(1)(x)
            if use_batchnorm:
                x = self.batchnorm()(x, training=1)
            if s <= 8:
                x = tf.keras.layers.Dropout(0.5)(x, training=1)
            return x

        _ = inputs = tf.keras.layers.Input(shape=self.img_shape)
        _ = block(_, self.img_shape[0], 1, False, nf_out=1, nf_next=self.gf)
        _ = tf.keras.layers.Activation('tanh')(_)
        return tf.keras.models.Model(inputs=inputs, outputs=[_])

    def discriminator_network(self):
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = self.conv2d(filters, kernel_size=4, strides=2, padding="same", use_bias=False)(layer_input)
            if normalization:
                d = self.batchnorm()(d, training=1)
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            return d

        input_ = tf.keras.layers.Input(shape=self.img_shape)
        _ = d_layer(input_, self.df, normalization=False)
        for layer in range(1, 3):
            out_feat = self.df * min(2 ** layer, 8)
            _ = d_layer(_, out_feat)

        out_feat = self.df * min(2 ** 3, 8)
        _ = ZeroPadding2D(1)(_)
        _ = self.conv2d(out_feat, kernel_size=4, use_bias=False)(_)
        _ = self.batchnorm()(_, training=1)
        _ = tf.keras.layers.LeakyReLU(alpha=0.2)(_)

        # output layer
        _ = ZeroPadding2D(1)(_)
        _ = self.conv2d(1, kernel_size=4, activation="sigmoid" if not self.use_lsgan else None)(_)

        return tf.keras.models.Model(input_, _)

    def build_generator_unet(self, pretrained_weights_file_path=None):
        gg_model = self.generator_network_unet()
        if pretrained_weights_file_path:
            # Load previously saved weights
            print ("Loading weights from {}...".format(pretrained_weights_file_path))
            gg_model.load_weights(pretrained_weights_file_path, by_name=False)

        return gg_model

    def build_generator_3d(self, pretrained_weights_file_path=None, generator_type='l2h'):
        if generator_type == 'l2h':
            gg_model = self.generator_network_3d_to_2d()
        else:
            gg_model = self.generator_network_2d_to_2d()

        if pretrained_weights_file_path:
            # Load previously saved weights
            print ("Loading weights from {}...".format(pretrained_weights_file_path))
            gg_model.load_weights(pretrained_weights_file_path, by_name=False)

        return gg_model

    def build_discriminator_2d(self):
        return self.discriminator_network()

    def build_discriminator_3d(self, discr_type='l2h'):
        if discr_type == 'l2h':
            return self.discriminator_network_3d_to_2d()
        else:
            return self.discriminator_network_2d_to_2d()
