import keras
from keras import layers, ops

from alz_prog_net.components.encoder import ModalityEncoder
from alz_prog_net.components.transformer import TransformerBlock
from alz_prog_net.components.progression import (
    DiseaseProgressionDecoder,
    SwiGLU
)


@keras.saving.register_keras_serializable(
    package="AlzProgNet"
)
class AlzProgNet(keras.Model):

    def __init__(
        self,
        num_modalities,
        modalities_hidden_dims=(16, 128),
        modality_output_dim=64,
        latent_dim=256,
        num_transformer_layers=4,
        num_heads=2,
        ff_dim=512,
        dropout=0.1,
        progression_hidden_dims=(128, 64, 32),
        time_points=(0, 6, 12, 24),
        temporal_levels=(True, True, False),
        time_dim=16,
        n_time_frequencies=4,
        use_time=True,
        use_gate=True,
        use_residual=True,
        initial_residual_scale=0.1,
        output_dim=3,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Store ALL constructor arguments
        self.num_modalities = num_modalities
        self.modalities_hidden_dims = tuple(modalities_hidden_dims)
        self.modality_output_dim = modality_output_dim
        self.latent_dim = latent_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.progression_hidden_dims = tuple(progression_hidden_dims)
        self.time_points = tuple(time_points)
        self.temporal_levels = tuple(temporal_levels)
        self.time_dim = time_dim
        self.n_time_frequencies = n_time_frequencies
        self.use_time = use_time
        self.use_gate = use_gate
        self.use_residual = use_residual
        self.initial_residual_scale = initial_residual_scale
        self.output_dim = output_dim

        # --------------------------------------------------
        # Modality encoders
        # --------------------------------------------------

        self.modality_encoders = [
            ModalityEncoder(
                hidden_units=self.modalities_hidden_dims,
                output_dim=self.modality_output_dim,
                dropout_rate=self.dropout,
                modality_name=f"modality_encoder_{i}"
            )
            for i in range(self.num_modalities)
        ]

        # --------------------------------------------------
        # Transformer blocks
        # --------------------------------------------------

        self.transformer_blocks = [
            TransformerBlock(
                dim=self.modality_output_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                name=f"transformer_{i}"
            )
            for i in range(self.num_transformer_layers)
        ]

        self.transformer_norm = layers.LayerNormalization()
        self.flattern = layers.Flatten()
        self.latent_norm = layers.LayerNormalization()

        # --------------------------------------------------
        # Latent projection
        # --------------------------------------------------

        self.latent_projection = SwiGLU(
            output_dim=self.latent_dim
        )

        # --------------------------------------------------
        # Disease progression decoder
        # --------------------------------------------------

        self.disease_progression = DiseaseProgressionDecoder(
            latent_dim=self.latent_dim,
            hidden_dims=self.progression_hidden_dims,
            time_points=self.time_points,
            output_dim=self.output_dim,
            time_dim=self.time_dim,
            n_time_frequencies=self.n_time_frequencies,
            temporal_levels=self.temporal_levels,
            use_time=self.use_time,
            use_gate=self.use_gate,
            use_residual=self.use_residual,
            initial_residual_scale=self.initial_residual_scale,
            name="disease_progression"
        )

    def call(
        self,
        inputs,
        training=None,
        return_latent=False
    ):
        modality_embeddings = []

        for encoder, modality_input in zip(
            self.modality_encoders,
            inputs
        ):
            embedding = encoder(
                modality_input,
                training=training
            )

            modality_embeddings.append(embedding)

        x = ops.concatenate(
            modality_embeddings,
            axis=1
        )

        for transformer in self.transformer_blocks:
            x = transformer(
                x,
                training=training
            )

        x = self.transformer_norm(x)
        x = self.flattern(x)
        x = self.latent_norm(x)

        latent = self.latent_projection(x)

        predictions = self.disease_progression(
            latent,
            training=training
        )

        if return_latent:
            return {
                "predictions": predictions,
                "latent": latent
            }

        return predictions

    # ======================================================
    # Keras serialization
    # ======================================================

    def get_config(self):
        config = super().get_config()

        config.update({
            "num_modalities": self.num_modalities,
            "modalities_hidden_dims": self.modalities_hidden_dims,
            "modality_output_dim": self.modality_output_dim,
            "latent_dim": self.latent_dim,
            "num_transformer_layers": self.num_transformer_layers,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "progression_hidden_dims": self.progression_hidden_dims,
            "time_points": self.time_points,
            "temporal_levels": self.temporal_levels,
            "time_dim": self.time_dim,
            "n_time_frequencies": self.n_time_frequencies,
            "use_time": self.use_time,
            "use_gate": self.use_gate,
            "use_residual": self.use_residual,
            "initial_residual_scale": self.initial_residual_scale,
            "output_dim": self.output_dim,
        })

        return config
