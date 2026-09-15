import keras
from keras import layers
from typing import Any, Dict, List, Optional


@keras.saving.register_keras_serializable(package="CustomLayers")
class ModalityEncoder(keras.Model):
    """
    MLP-based modality encoder.

    Input:
        (batch, input_dim)

    Output:
        (batch, 1, 64)

    Architecture:
        Dense -> GeLU -> Dropout -> ...
        -> Dense(64) -> Reshape(1, 64)
    """

    def __init__(
        self,
        hidden_units: tuple[int],
        modality_name: str,
        dropout_rate: float = 0.1,
        output_dim: int = 64,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if not hidden_units:
            raise ValueError("hidden_units must contain at least one value.")

        if any(units <= 0 for units in hidden_units):
            raise ValueError("All hidden_units must be positive.")

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                "dropout_rate must be in the range [0, 1)."
            )

        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.hidden_units = list(hidden_units)
        self.modality_name = modality_name
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim

        self.mlp = []

        for i, units in enumerate(hidden_units):
            self.mlp.append(
                layers.Dense(
                    units,
                    activation="gelu",
                    name=f"{modality_name}_dense_{i}",
                )
            )

            if i < len(hidden_units) - 1:
                self.mlp.append(
                    layers.Dropout(
                        dropout_rate,
                        name=f"{modality_name}_dropout_{i}",
                    )
                )

        # Ensure the encoder always produces output_dim features.
        if hidden_units[-1] != output_dim:
            self.mlp.append(
                layers.Dense(
                    output_dim,
                    activation="gelu",
                    name=f"{modality_name}_output",
                )
            )

        self.reshape = layers.Reshape(
            (1, output_dim),
            name=f"{modality_name}_reshape",
        )

    def call(
        self,
        inputs,
        training: Optional[bool] = None,
    ):
        x = inputs

        for layer in self.mlp:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)

        return self.reshape(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "modality_name": self.modality_name,
            "dropout_rate": self.dropout_rate,
            "output_dim": self.output_dim,
        })
        return config
