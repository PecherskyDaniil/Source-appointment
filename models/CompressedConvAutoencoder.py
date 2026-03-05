import tensorflow as tf

class CompressedConvAutoencoder(tf.keras.Model):
    """
    Compress Convolutional AutoEncoder
    name:str - name of model
    in_shape:tuple - shape of in layer
    out_dim:int - dimension of out layer
    bottleneck_dim:int - dimension of bottleneck
    """
    def __init__(self,name:str,in_shape:tuple,bottleneck_dim:int,out_dim:int):
        super().__init__()
        self.name=name

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=in_shape),
            tf.keras.layers.Conv2D(32, kernel_size=3, strides=2,activation="relu"),
            tf.keras.layers.Conv2D(16, kernel_size=3, strides=2,activation="relu"),
            tf.keras.layers.Conv2D(8, kernel_size=3, strides=2,activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(bottleneck_dim), 
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(out_dim),
        ])
    def call(self, x):
        latent_x = self.encoder(x) # compress
        decoded_x = self.decoder(latent_x) # unpack
        return decoded_x