# Import modules to create custom FW-RNN cell
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

from tensorflow.python.keras.layers.recurrent import (
    _generate_zero_filled_state_for_cell,
    _generate_zero_filled_state,
    activations,
    initializers,
    regularizers,
    nest,
)

# # Build FW-RNN model
# -  Build custom FW_RNN cell and wrap it in RNN layer (https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN), like this: RNN(FW_RNN)
#     -  "The cell abstraction, together with the generic keras.layers.RNN class, make it very easy to implement custom RNN architectures for your research."

# Created by using this guide: https://www.tensorflow.org/guide/keras/custom_layers_and_models

class FW_RNNCell(layers.Layer):
    def __init__(
        self,
        units,
        use_bias,
        batch_size,
        decay_rate,
        learning_rate,
        activation,
        step,
        **kwargs
    ):
        super(FW_RNNCell, self).__init__(**kwargs)
        self.units = units
        self.step = step
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.l = decay_rate
        self.e = learning_rate
        self.LN = layers.LayerNormalization()
        self.batch = batch_size
        self.state_size = self.units

        # Initializer for the slow input-to-hidden weights matrix
        self.C_initializer = initializers.get("glorot_uniform")

        # Initializer for the slow hidden weights matrix
        self.W_h_initializer = initializers.get("identity")

        # Initializer for the fast weights matrix
        self.A_initializer = initializers.get("zeros")

        # Initializer for the bias vector.
        self.b_x_initializer = initializers.get("zeros")

    def build(self, input_shape):
        # Build is only called at the start, to initialize all the weights and biases

        # C = Slow input-to-hidden weights [shape (features_vector, units)]
        self.C = self.add_weight(
            shape=(input_shape[-1], self.units),
            name="inputweights",
            initializer=self.C_initializer,
        )

        # W_h The previous hidden state via the slow transition weights [shape (units, units)]
        self.W_h = self.add_weight(
            shape=(self.units, self.units),
            name="hiddenweights",
            initializer=self.W_h_initializer,
        )
        self.W_h = tf.scalar_mul(0.05, self.W_h)

        # A (fast weights) [shape (batch_size, units, units)]
        self.A = self.add_weight(
            shape=(self.batch, self.units, self.units),
            name="fastweights",
            initializer=self.A_initializer,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,), name="bias", initializer=self.b_x_initializer,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0] if nest.is_sequence(states) else states

        # Next hidden state h(t+1) is computed in two steps:
        # Step 1 calculate preliminary vector: h_0(t+1) = f(W_h ⋅ h(t) + C ⋅ x(t))
        h = K.dot(prev_output, self.W_h) + K.dot(inputs, self.C)
        if self.bias is not None:
            h = h + self.bias
        if self.activation is not None:
            h = self.activation(h)

        # Reshape h to use with a
        h_s = tf.reshape(h, [self.batch, 1, self.units])

        # Define preliminary vector in variable
        prelim = tf.reshape(K.dot(prev_output, self.W_h), (h_s.shape)) + tf.reshape(
            K.dot(inputs, self.C), (h_s.shape)
        )

        # Fast weights update rule: A(t) = λ*A(t-1) + η*h(t) ⋅ h(t)^T
        self.A.assign(
            tf.math.add(
                tf.scalar_mul(self.l, self.A),
                tf.scalar_mul(
                    self.e, tf.linalg.matmul(tf.transpose(h_s, [0, 2, 1]), h_s)
                ),
            )
        )

        # Step 2: Initiate inner loop with preliminary vector, which runs for S steps
        # to progressively change the hidden state into h(t+1) = h_s(t+1)
        # h_s+1(t+1) f([W_h ⋅ h(t) + C ⋅ x(t)]) + A(t)h_s(t+1)
        for _ in range(self.step):
            h_s = tf.math.add(prelim, tf.linalg.matmul(h_s, self.A))
            if self.activation is not None:
                h_s = self.activation(h_s)

            # Apply layer normalization on hidden state
            h_s = self.LN(h_s)

        h = tf.reshape(h_s, [self.batch, self.units])

        output = h
        new_state = [output] if nest.is_sequence(states) else output
        return output, new_state

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {
            "units": self.units,
            "step": self.step,
            "batch_size": self.batch,
            "use_bias": self.use_bias,
            "activation": activations.serialize(self.activation),
            "decay_rate": self.l,
            "learning_rate": self.e
        }
        base_config = super(FW_RNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))