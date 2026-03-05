import tensorflow as tf


class SeqAutoencoder(tf.keras.Model):
    """
    Sequential AutoEncoder
    name:str - name of model
    in_out_dim:int - dimension of in and out layers
    bottleneck_dim:int - dimension of bottleneck
    """
    def __init__(self,name:str,in_out_dim:int,bottleneck_dim:int):
        super().__init__()
        self.name=name
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(in_out_dim,)),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(bottleneck_dim), 
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(in_out_dim),
        ])
    def call(self, x):
        latent_x = self.encoder(x) # compress
        decoded_x = self.decoder(latent_x) # unpack
        return decoded_x