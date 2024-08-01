import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

bb = 512
# DATA
DATASET_NAME = "my3d"
BATCH_SIZE = 32
# AUTO = tf.data.AUTOTUNE
# INPUT_SHAPE = (8, 8, 8, 128)
# INPUT_SHAPE0 = (8, 8, 8, 128)
NUM_CLASSES = 3

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 100

# TUBELET EMBEDDING
PATCH_SIZE = (4,4,4)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

trial=3

    
from tensorflow.keras import layers


class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

LLL = 2

class Encoder(layers.Layer):

    def __init__(self, latent_dim=4, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        
        self.Conv3D_1 = layers.Conv3D(LLL, 3, activation="relu", strides=2, padding="same")
        self.BN_1 = keras.layers.BatchNormalization()
        self.Conv3D_2 = layers.Conv3D(LLL*2, 3, activation="relu", strides=2, padding="same")
        self.BN_2 = keras.layers.BatchNormalization()
        self.Conv3D_21 = layers.Conv3D(LLL*2, 3, activation="relu", strides=2, padding="same")
        self.BN_21 = keras.layers.BatchNormalization()
        
        self.Conv3D_3 = layers.Conv3D(LLL*4, 3, activation="relu", strides=1, padding="same")
        self.BN_3 = keras.layers.BatchNormalization()
        self.Conv3D_4 = layers.Conv3D(LLL*4, 1, activation="relu", strides=1, padding="same")
        self.BN_4 = keras.layers.BatchNormalization()
        self.outcn = layers.Conv3D(1, 1, activation="relu", strides=1, padding="same")
        
        
        self.BN_res1 = keras.layers.BatchNormalization()
        self.BN_res2 = keras.layers.BatchNormalization()
        self.BN_res3 = keras.layers.BatchNormalization()
        
        self.Flatten = layers.Flatten()
        # self.mogai = layers.Dense(16, name="z_mean")
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling()
        
        self.tubelet_embedder=TubeletEmbedding(embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE)
        self.positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM)

    def call(self, inputs):
        x = self.Conv3D_1(inputs)
        x = self.BN_1(x)
        x = self.Conv3D_2(x)
        x = self.BN_2(x)
        x = self.Conv3D_21(x)
        x = self.BN_21(x)

        
        res=x
        x = self.Conv3D_3(x)
        x = self.BN_3(x)
        x = self.Conv3D_4(x)
        x = self.BN_4(x)
        x+=res
        x = self.outcn(x)

        return x
    
class Neck(keras.Model):
    def __init__(self, latent_dim=4, intermediate_dim=64, name="neck", **kwargs):
        super(Neck, self).__init__(name=name, **kwargs)
        
        self.Flatten = layers.Flatten()
        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling()
        
    def call(self, inputs):
        x = self.Flatten(inputs)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z
    
class Neck2(keras.Model):
    def __init__(self, latent_dim=4, intermediate_dim=64, name="neck2", **kwargs):
        super(Neck2, self).__init__(name=name, **kwargs)
        
        self.dense_proj = layers.Dense(8 * 8*8* 1, activation="relu")
        self.reshape = layers.Reshape((8, 8,8, 1))
        
    def call(self, inputs):
        x = self.dense_proj(inputs)
        x = self.reshape(x)
        return x

class Decoder(layers.Layer):

    def __init__(self, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        
        self.Conv3DT_24 = layers.Conv3DTranspose(LLL*4, 1, activation="relu", strides=1, padding="same")
        self.BN_24 = keras.layers.BatchNormalization()
        self.Conv3DT_34 = layers.Conv3DTranspose(LLL*4, 3, activation="relu", strides=1, padding="same")
        self.BN_34 = keras.layers.BatchNormalization()
        
        self.Conv3DT_43 = layers.Conv3DTranspose(LLL*2, 3, activation="relu", strides=2, padding="same")
        self.BN_43 = keras.layers.BatchNormalization()
        self.Conv3DT_4 = layers.Conv3DTranspose(LLL*2, 3, activation="relu", strides=2, padding="same")
        self.BN_4 = keras.layers.BatchNormalization()
        self.Conv3DT_5 = layers.Conv3DTranspose(LLL, 3, activation="relu", strides=2, padding="same")
        self.BN_5 = keras.layers.BatchNormalization()
        self.Conv3DT_out = layers.Conv3DTranspose(1, 3, activation="tanh", padding="same")
        
        self.BN_res1 = keras.layers.BatchNormalization()
        self.BN_res2 = keras.layers.BatchNormalization()
        self.BN_res3 = keras.layers.BatchNormalization()
        self.BN_res4 = keras.layers.BatchNormalization()

    def call(self, inputs):
        x=inputs
        
        # res=x
        x = self.Conv3DT_24(x)
        x = self.BN_24(x)
        x = self.Conv3DT_34(x)
        x = self.BN_34(x)
        # x+=res
        

        x = self.Conv3DT_43(x)
        x = self.BN_43(x)
        x = self.Conv3DT_4(x)
        x = self.BN_4(x)
        x = self.Conv3DT_5(x)
        x = self.BN_5(x)
        return self.Conv3DT_out(x)

class Classifier(keras.Model):

    def __init__(
        self,
        intermediate_dim=64,
        latent_dim=4,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.neck = Neck(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.neck2 = Neck2(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim)

    def call(self, inputs):
        inputs = self.encoder(inputs)
        z_mean, z_log_var, z = self.neck(inputs)
        neck_out = self.neck2(z)
        reconstructed = self.decoder(neck_out)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
    
def plot_slices(num_rows, num_columns, width, height, data):
    # data = np.rot90(np.array(data))
    # data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()
    