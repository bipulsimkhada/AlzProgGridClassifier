import keras
from keras import ops


class RMSNorm(keras.layers.Layer):
    """Root Mean Square Layer Normalization."""

    def __init__(self, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        dim = input_shape[-1]
        self.scale = self.add_weight(
            name="scale",
            shape=(dim,),
            initializer="ones",
            trainable=True,
        )

    def call(self, x):
        # RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale
        rms = ops.sqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.scale


class TransformerBlock(keras.layers.Layer):
    """Pre-norm Transformer block with attention and SwiGLU FFN."""

    def __init__(
            self,
            dim,
            num_heads,
            ff_dim,
            attention_dropout=0.0,
            residual_dropout=0.05,
            **kwargs):
        super().__init__(**kwargs)

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.norm1 = RMSNorm()

        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            use_bias=False,
            dropout=attention_dropout
        )

        self.norm2 = RMSNorm()

        self.residual_dropout = keras.layers.Dropout(residual_dropout)

        # SwiGLU
        self.gate = keras.layers.Dense(ff_dim, use_bias=False)
        self.up = keras.layers.Dense(ff_dim, use_bias=False)
        self.down = keras.layers.Dense(dim, use_bias=False)

    def call(self, x, training=None):
        # Attention
        h = self.norm1(x)
        h = self.attn(h, h)
        x = x + h

        # Feed-forward network
        h = self.norm2(x)

        # SwiGLU:
        # SiLU(W_gate x) * (W_up x)
        h = ops.silu(self.gate(h)) * self.up(h)
        h = self.down(h)

        # Residual connection
        return x + self.residual_dropout(h, training=training)