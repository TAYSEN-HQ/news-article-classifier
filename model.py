import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config.Config as Config
class Transformer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, lr=0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation='relu'),
                layers.Dense(embed_dim)
            ]
        )

        self.layernorm1 = layers.LayerNormalisation(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalisation(epsilon=1e-6)
        self.dropout1 = layers.Dropout(lr)
        self.dropout2 = layers.Dropout(lr)
    
    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)

        output = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(output + ffn_output)
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embedding = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.position_embedding(positions)
        x = self.token_embedding(x)

        return x + positions



inputs = layers.Input(shape=(Config.maxlen,))
embedding_layer = TokenAndPositionEmbedding(Config.maxlen, Config.vocab_size, Config.embed_dim)

x = embedding_layer(inputs)

transformer = Transformer(Config.embed_dim, Config.num_heads, COnfig.ff_dim)
x = transformer(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation='relu')(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val))