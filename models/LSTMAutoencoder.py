import tensorflow as tf

class LSTMAutoencoder(tf.keras.Model):
    """
    LSTM AutoEncoder
    name:str - name of model
    in_shape:tuple - shape of in layer
    out_dim:int - dimension of out layer
    bottleneck_dim:int - dimension of bottleneck
    """
    def __init__(self,name:str,in_shape:tuple,bottleneck_dim:int,out_dim:int):
        super().__init__()
        self.name=name

        self.encoder = tf.keras.Sequential([
            
            tf.keras.layers.LSTM(120,input_shape=in_shape,return_sequences=False),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(60, activation='relu'),
            tf.keras.layers.Dense(bottleneck_dim), 
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(out_dim)
        ])
    def call(self, x):
        latent_x = self.encoder(x) # compress
        decoded_x = self.decoder(latent_x) # unpack
        return decoded_x