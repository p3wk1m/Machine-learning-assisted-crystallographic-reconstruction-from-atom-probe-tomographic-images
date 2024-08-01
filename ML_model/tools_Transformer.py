import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# bb = 512
# DATA
DATASET_NAME = "my3d"
BATCH_SIZE = 32
# AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (64, 64, 64, 1)
INPUT_SHAPE0 = (64,64, 64, 1)
NUM_CLASSES = 3

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 100

# TUBELET EMBEDDING
PATCH_SIZE = (16,16,16)
# PATCH_SIZE = (4,16,16)
# PATCH_SIZE = (8,8,8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 8
NUM_HEADS = 8
NUM_LAYERS = 8

trial=3

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches
    
# class PositionalEncoder(layers.Layer):
#     def __init__(self, embed_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim

#     def build(self, input_shape):
#         _, num_tokens, _ = input_shape
#         self.position_embedding = layers.Embedding(
#             input_dim=num_tokens, output_dim=self.embed_dim
#         )
#         self.positions = tf.range(start=0, limit=num_tokens, delta=1)

#     def call(self, encoded_tokens):
#         # Encode the positions and add it to the encoded tokens
#         encoded_positions = self.position_embedding(self.positions)
#         encoded_tokens = encoded_tokens + encoded_positions
#         return encoded_tokens

# from tensorflow.keras.layers import Layer

class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, max_len=10000, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.max_len = max_len

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.positions = tf.cast(tf.range(start=0, limit=num_tokens, delta=1), dtype=tf.float32)

    def call(self, encoded_tokens):
        angle_rads = self._get_angles()
        # Apply sin to even indices in the array; 2i
        sin_indices = tf.range(0, self.embed_dim, 2)
        sin = tf.math.sin(tf.gather(angle_rads, sin_indices, axis=-1))
        # Apply cos to odd indices in the array; 2i+1
        cos_indices = tf.range(1, self.embed_dim, 2)
        cos = tf.math.cos(tf.gather(angle_rads, cos_indices, axis=-1))
        pos_encoding = tf.concat([sin, cos], axis=-1)
        # Add position encoding to encoded tokens
        encoded_tokens += pos_encoding
        return encoded_tokens

    def _get_angles(self):
        angle_rates = 1 / tf.pow(10000, (2 * (tf.range(self.embed_dim, dtype=tf.float32) // 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rads = tf.matmul(self.positions[:, tf.newaxis], angle_rates[tf.newaxis, :])
        return angle_rads
    


    
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
    
LLL = 4


# class TT(layers.Layer):
#     def __init__(self, num_layers=8, **kwargs):
#         super(TT, self).__init__(**kwargs)
#         self.num_layers = num_layers
#         self.layer_norms = [layers.LayerNormalization(epsilon=1e-6) for _ in range(self.num_layers)]
#         self.multihead_attentions = [layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1) 
#                                      for _ in range(self.num_layers)]
#         self.dense_layers1 = [layers.Dense(units=PROJECTION_DIM * 4, activation=tf.nn.gelu) 
#                               for _ in range(self.num_layers)]
#         self.dense_layers2 = [layers.Dense(units=PROJECTION_DIM, activation=tf.nn.gelu) 
#                               for _ in range(self.num_layers)]
#         self.token_embedding = self.add_weight("token_embedding", shape=(1, PROJECTION_DIM), initializer="random_normal", trainable=True)

#     def call(self, inputs):
#         # Add trainable token to the input
#         token = tf.tile(self.token_embedding, [tf.shape(inputs)[0], 1])  # Repeat the token for each sequence in the batch
#         x = tf.concat([token, inputs], axis=1)  # Concatenate the token with input sequences
        
#         for i in range(self.num_layers):
#             x1 = self.layer_norms[i](x)
#             attention_output = self.multihead_attentions[i](x1, x1)
#             x2 = layers.Add()([attention_output, x])
#             x3 = self.dense_layers1[i](x2)
#             x3 = self.dense_layers2[i](x3)
#             x = layers.Add()([x3, x2])
#         return token
#         # return x

class TT(layers.Layer):
    def __init__(self, num_layers=8, **kwargs):
        super(TT, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.layer_norms = [layers.LayerNormalization(epsilon=1e-6) for _ in range(self.num_layers)]
        self.multihead_attentions = [layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1) 
                                     for _ in range(self.num_layers)]
        self.dense_layers1 = [layers.Dense(units=PROJECTION_DIM, activation=tf.nn.gelu) 
                              for _ in range(self.num_layers)]
        self.dense_layers2 = [layers.Dense(units=PROJECTION_DIM, activation=tf.nn.gelu) 
                              for _ in range(self.num_layers)]
        self.position_embedding = self.add_weight("position_embedding", shape=(1, 1, PROJECTION_DIM), initializer="random_normal", trainable=True)
        self.sinusoidal = PositionalEncoder(embed_dim=PROJECTION_DIM)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]

        # Learnable position embedding
        position_embedding = tf.tile(self.position_embedding, [batch_size, 1, 1])

        # Add position embedding to inputs
        inputs_with_position = tf.concat([position_embedding,inputs], axis=1)
        inputs_with_position = self.sinusoidal(inputs_with_position)

        for i in range(self.num_layers):
            x1 = self.layer_norms[i](inputs_with_position)
            attention_output = self.multihead_attentions[i](x1, x1)
            x2 = layers.Add()([attention_output, inputs_with_position])
            x3 = self.dense_layers1[i](x2)
            x3 = self.dense_layers2[i](x3)
            inputs_with_position = layers.Add()([x3, x2])

        # Remove position embedding from output
        final_output = inputs_with_position[:, 0, :]
        # final_output = inputs_with_position[:, :, :]
        # print(inputs_with_position.shape)
        return final_output

LLL000 = 64

class Classifier(keras.models.Model):

    def __init__(self, name="Classifier", **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.dropout = layers.Dropout(0.5)
        # self.Dense_22 = layers.Dense(256,activation='relu')
        self.Dense_23 = layers.Dense(LLL000,activation='tanh',name='dense_yaw')
        self.Dense_24 = layers.Dense(LLL000,activation='tanh',name='dense_pitch')
        self.Dense_25 = layers.Dense(LLL000,activation='tanh',name='dense_roll')
        self.Dense_out1 = layers.Dense(31,name='yaw')
        self.Dense_out2 = layers.Dense(31,name='pitch')
        self.Dense_out3 = layers.Dense(31,name='roll')
        self.TT = TT()
        self.tubelet_embedder=TubeletEmbedding(embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE)
        # self.positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM)
        
    def call(self, inputs):
        x = inputs

        patches = self.tubelet_embedder(x)
        # encoded_patches = self.positional_encoder(patches)
        # print(encoded_patches.shape)

        # for _ in range(8):
        #     x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        #     attention_output = layers.MultiHeadAttention(
        #         num_heads=NUM_HEADS, key_dim=8, dropout=0.1
        #     )(x1, x1)
        #     x2 = layers.Add()([attention_output, encoded_patches])
        #     x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        #     x3 = keras.Sequential(
        #         [
        #             layers.Dense(units=PROJECTION_DIM * 4, activation=tf.nn.gelu),
        #             layers.Dense(units=PROJECTION_DIM, activation=tf.nn.gelu),
        #         ]
        #     )(x3)
        #     encoded_patches = layers.Add()([x3, x2])
        # print(encoded_patches.shape)
        encoded_patches = self.TT(patches)

        x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        x = layers.Flatten()(x)
        x = self.dropout(x)
        # print(x.shape)
        # yaw = self.Dense_out1(self.Dense_23(x))
        # pitch = self.Dense_out2(self.Dense_24(x))
        # roll = self.Dense_out3(self.Dense_25(x))
        
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

