import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

bb = 4
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



    
def plot_slices(num_rows, num_columns, width, height, data):
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
    
LLL = 8

class Classifier(keras.models.Model):

    def __init__(self, name="Classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.Conv3D_1 = layers.Conv3D(LLL, 3, activation="relu", strides=1, padding="same")
        self.BN_1 = keras.layers.BatchNormalization()
        self.Conv3D_2 = layers.Conv3D(LLL*2, 3, activation="relu", strides=1, padding="same")
        self.BN_2 = keras.layers.BatchNormalization()
        self.Conv3D_21 = layers.Conv3D(LLL*2, 3, activation="relu", strides=1, padding="same")
        self.BN_21 = keras.layers.BatchNormalization()
        self.Conv3D_22 = layers.Conv3D(LLL*4, 3, activation="relu", strides=1, padding="same")
        self.BN_22 = keras.layers.BatchNormalization()
        self.Conv3D_23 = layers.Conv3D(LLL*4, 3, activation="relu", strides=1, padding="same")
        self.BN_23 = keras.layers.BatchNormalization() 
        self.Flatten = layers.Flatten()
        self.Dense_out1 = layers.Dense(31,name='yaw')
        self.Dense_out2 = layers.Dense(31,name='pitch')
        self.Dense_out3 = layers.Dense(31,name='roll')
        
    def call(self, inputs):
        x = inputs
        x = self.Conv3D_1(x)
        # x = self.BN_1(x)
        x = layers.MaxPooling3D(pool_size=2)(x)
        x = self.Conv3D_2(x)
        x = layers.MaxPooling3D(pool_size=2)(x)
        # x = self.BN_2(x)
        x = self.Conv3D_21(x)
        x = layers.MaxPooling3D(pool_size=2)(x)
        # x = self.BN_21(x)
        x = self.Conv3D_22(x)
        x = layers.MaxPooling3D(pool_size=2)(x)
        # x = self.BN_22(x)
        x = self.Conv3D_23(x)
        x = layers.MaxPooling3D(pool_size=2)(x)
        # x = self.BN_23(x)
        x = self.Flatten(x)
        
        # yaw = self.Dense_out1(self.Dense_23(x))
        # pitch = self.Dense_out2(self.Dense_24(x))
        # roll = self.Dense_out3(self.Dense_25(x))
        
        # x_yaw = self.Dense_23(x)
        # x_pitch = self.Dense_24(x)
        # x_roll = self.Dense_25(x)
        yaw = self.Dense_out1(x)
        pitch = self.Dense_out2(x)
        roll = self.Dense_out3(x)

        return [yaw,pitch,roll]
    
    
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
