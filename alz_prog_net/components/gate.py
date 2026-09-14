import keras
from keras import layers, ops
from typing import Any, Dict, Optional


@keras.saving.register_keras_serializable(package="CustomLayers")
class StaticGate(layers.Layer):
    """Static element-wise gating layer with dynamic shape resolution."""

    def __init__(
        self,
        units: Optional[int] = None,
        initializer: str = "zeros",
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.initializer = initializer
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tuple) -> None:
        units = self.units if self.units is not None else input_shape[-1]
        
        self.gate_logits = self.add_weight(
            shape=(units,),
            initializer=self.initializer,
            trainable=True,
            name="gate_logits",
        )
        self.dropout = layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        g = ops.sigmoid(self.gate_logits)
        x = inputs * g
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "units": self.units,
            "initializer": self.initializer,
            "dropout_rate": self.dropout_rate,
        })
        return config


@keras.saving.register_keras_serializable(package="CustomLayers")
class DynamicGate(layers.Layer):
    """Dynamic input-dependent feature gating layer."""

    def __init__(
        self,
        units: Optional[int] = None,
        initializer: str = "glorot_uniform",
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.initializer = initializer
        self.dropout_rate = dropout_rate

        self.input_norm = layers.LayerNormalization(axis=-1, name="input_norm")

    def build(self, input_shape: tuple) -> None:
        units = self.units if self.units is not None else input_shape[-1]

        self.gate_dense = layers.Dense(
            units=units,
            activation="sigmoid",
            kernel_initializer=self.initializer,
            bias_initializer="zeros",
            name="gate_dense",
        )
        self.dropout = layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        
        self.gate_dense.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        normalized_inputs = self.input_norm(inputs)
        gate = self.gate_dense(normalized_inputs)
        x = ops.multiply(inputs, gate)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "units": self.units,
            "initializer": self.initializer,
            "dropout_rate": self.dropout_rate,
        })
        return config


@keras.saving.register_keras_serializable(package="CustomLayers")
class SwiGLU(layers.Layer):
    """Swish Gated Linear Unit (SwiGLU) layer."""

    def __init__(
        self,
        units: Optional[int] = None,
        initializer: str = "glorot_uniform",
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.initializer = initializer
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tuple) -> None:
        units = self.units if self.units is not None else input_shape[-1]
        
        self.proj = layers.Dense(
            units=units * 2,
            kernel_initializer=self.initializer,
            name="proj",
        )
        self.dropout = layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        
        self.proj.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        x = self.proj(inputs)
        a, b = ops.split(x, 2, axis=-1)
        x = a * ops.silu(b)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        units = self.units if self.units is not None else input_shape[-1]
        return input_shape[:-1] + (units,)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "units": self.units,
            "initializer": self.initializer,
            "dropout_rate": self.dropout_rate,
        })
        return config