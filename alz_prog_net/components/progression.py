import keras
import math
from keras import layers, ops

@keras.saving.register_keras_serializable(
    package="DiseaseProgression"
)
class SwiGLU(layers.Layer):
    def __init__(
        self,
        output_dim,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim

        self.value_projection = layers.Dense(
            output_dim,
            name="value_projection"
        )

        self.gate_projection = layers.Dense(
            output_dim,
            name="gate_projection"
        )

    def call(self, inputs):
        value = self.value_projection(inputs)
        gate = ops.silu(
            self.gate_projection(inputs)
        )

        return value * gate

    def get_config(self):
        config = super().get_config()

        config.update({
            "output_dim": self.output_dim
        })

        return config

@keras.saving.register_keras_serializable(
    package="DiseaseProgression"
)
class FourierTimeEncoder(keras.Model):
    def __init__(
        self,
        time_dim=16,
        n_frequencies=4,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.time_dim = time_dim
        self.n_frequencies = n_frequencies

        self.projection_1 = layers.Dense(
            time_dim,
            activation="silu",
            name="time_projection_1"
        )

        self.projection_2 = layers.Dense(
            time_dim,
            name="time_projection_2"
        )

    def call(
        self,
        inputs
    ):
        t, dt = inputs
        frequencies = ops.power(
            ops.cast(2.0, t.dtype),
            ops.arange(
                self.n_frequencies,
                dtype=t.dtype
            )
        )

        t_angles = (
            ops.cast(
                2.0 * math.pi,
                t.dtype
            )
            * t
            * frequencies
        )

        dt_angles = (
            ops.cast(
                2.0 * math.pi,
                t.dtype
            )
            * dt
            * frequencies
        )

        time_features = ops.concatenate(
            [   
                t,
                ops.sin(t_angles),
                ops.cos(t_angles),
                dt,
                ops.sin(dt_angles),
                ops.cos(dt_angles),
            ],
            axis=-1
        )

        x = self.projection_1(
            time_features
        )

        return self.projection_2(x)

    def get_config(self):
        config = super().get_config()

        config.update({
            "time_dim": self.time_dim,
            "n_frequencies": self.n_frequencies
        })

        return config


@keras.saving.register_keras_serializable(
    package="DiseaseProgression"
)
class InitialReduction(layers.Layer):
    def __init__(
        self,
        output_dim,
        use_layer_norm=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.norm = layers.LayerNormalization(
                name = "pre-norm"
            )

        self.swiglu = SwiGLU(
            output_dim,
            name="swiglu"
        )

    def call(self, inputs):
        x = inputs

        if self.use_layer_norm:
            x = self.norm(x)

        return self.swiglu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "use_layer_norm": self.use_layer_norm
        })

        return config


@keras.saving.register_keras_serializable(
    package="DiseaseProgression"
)
class TemporalProgressionBlock(layers.Layer):
    def __init__(
        self,
        state_dim,
        time_dim=16,
        use_time=True,
        use_gate=True,
        use_residual=True,
        initial_residual_scale=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.state_dim = state_dim
        self.time_dim = time_dim

        self.use_time = use_time
        self.use_gate = use_gate
        self.use_residual = use_residual

        self.initial_residual_scale = (
            initial_residual_scale
        )

        self.state_norm = layers.LayerNormalization(
            name="state_norm"
        )

        self.input_projection = layers.Dense(
            state_dim,
            name="input_projection"
        )

        self.delta_network = SwiGLU(
            state_dim,
            name="delta_network"
        )

        if self.use_gate:

            self.gate_network = layers.Dense(
                state_dim,
                activation="sigmoid",
                name="temporal_gate"
            )

    def build(self, input_shape):
        self.residual_scale = self.add_weight(
            name="residual_scale",
            shape=(),
            initializer=keras.initializers.Constant(
                self.initial_residual_scale
            ),
            trainable=True
        )

        super().build(input_shape)

    def call(
        self,
        inputs
    ):
        if self.use_time:
            (
                current_input,
                previous_state,
                time_embedding
            ) = inputs

        else:
            (
                current_input,
                previous_state
            ) = inputs

            time_embedding = None

        #previous disease state
        previous_normalized = (
            self.state_norm(previous_state)
        )

        #current information
        current_features = (
            self.input_projection(
                current_input
            )
        )

        #proposed progression
        delta_inputs = [
            previous_normalized,
            current_features
        ]

        if self.use_time:
            delta_inputs.append(
                time_embedding
            )

        delta_input = ops.concatenate(
            delta_inputs,
            axis=-1
        )

        delta = self.delta_network(delta_input)

        #temporal gate
        if self.use_gate:
            gate_inputs = [
                previous_normalized,
                delta
            ]

            if self.use_time:
                gate_inputs.append(time_embedding)

            gate_input = ops.concatenate(
                gate_inputs,
                axis=-1
            )

            gate = self.gate_network(gate_input)

        else:
            gate = ops.ones_like(
                delta
            )

        update = self.residual_scale * gate * delta

        # residual progression
        if self.use_residual:
            state = previous_state + update
        else:
            state = update

        return {
            "state": state,
            "delta": delta,
            "gate": gate,
            "update": update
        }


    def get_config(self):
        config = super().get_config()

        config.update({
            "state_dim": self.state_dim,
            "time_dim": self.time_dim,
            "use_time": self.use_time,
            "use_gate": self.use_gate,
            "use_residual": self.use_residual,
            "initial_residual_scale": self.initial_residual_scale
        })

        return config


@keras.saving.register_keras_serializable(
    package="DiseaseProgression"
)
class DiseaseProgressionDecoder(keras.Model):
    def __init__(
        self,
        latent_dim=256,
        hidden_dims=(128, 64, 32),
        time_points=(0, 6, 12, 24),
        output_dim=3,
        time_dim=16,
        n_time_frequencies=4,
        temporal_levels=None,
        use_time=True,
        use_gate=True,
        use_residual=True,
        initial_residual_scale=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.time_points = time_points

        self.output_dim = output_dim
        self.time_dim = time_dim

        self.n_time_frequencies = n_time_frequencies

        self.use_gate = use_gate
        self.use_time = use_time
        self.use_residual = use_residual

        self.initial_residual_scale = initial_residual_scale

        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims cannot be empty.")

        if len(self.time_points) == 0:
            raise ValueError("time_points cannot be empty.")

        if self.time_points[0] != 0:
            raise ValueError("first timepoint must be baseline 0.")

        if temporal_levels is None:
            temporal_levels = (True,) * len(self.hidden_dims)

        self.temporal_levels = tuple(temporal_levels)

        self.max_time = float(max(self.time_points))

        if self.use_time:
            self.time_encoder = FourierTimeEncoder(
                time_dim=time_dim,
                n_frequencies=n_time_frequencies,
                name="time_encoder"
            )
        else:
            self.time_encoder = None

        self.initial_layers = []

        #initial layers for T0
        for level, hidden_dim in enumerate(self.hidden_dims):
            layer = InitialReduction(
                output_dim=hidden_dim,
                name=f"initial_level_{level}_dim_{hidden_dim}"
            )

            self.initial_layers.append(layer)

        # temporal blocks
        self.temporal_blocks = []

        for level, hidden_dim in enumerate(self.hidden_dims):
            if self.temporal_levels[level]:
                temporal_block = TemporalProgressionBlock(
                    state_dim=hidden_dim,
                    time_dim=time_dim,
                    use_time=use_time,
                    use_gate=use_gate,
                    use_residual=use_residual,
                    initial_residual_scale=initial_residual_scale,
                    name = f"temporal_level_{level}_dim_{hidden_dim}"
                )

            else:
                temporal_block = None

            self.temporal_blocks.append(temporal_block)

        # output head
        self.output_head = layers.Dense(
            output_dim,
            activation="softmax",
            name ="output_head"
        )

    def _get_time_embedding(
        self,
        reference_tensor,
        current_time,
        previous_time
    ):
        if not self.use_time:
            return None

        batch_size = ops.shape(
            reference_tensor
        )[0]

        dtype = reference_tensor.dtype

        normalized_t = float(current_time) / self.max_time
        normalized_dt = float(current_time - previous_time) / self.max_time

        t = ops.ones(
            (batch_size, 1),
            dtype=dtype
        ) * normalized_t

        dt = ops.ones(
            (batch_size, 1),
            dtype=dtype
        ) * normalized_dt

        return self.time_encoder(
            (t, dt)
        )

    def call(
        self,
        inputs,
        training=None,
        return_details=False
    ):
        z = inputs

        baseline_states = []
        x = z

        # baseline T0
        for layer in self.initial_layers:
            x = layer(x)

            baseline_states.append(x)

        previous_states = list(baseline_states)

        baseline_prediction = self.output_head(baseline_states[-1])

        predictions = [baseline_prediction]

        #store for analysis
        hidden_history = [
            [state] for state in baseline_states
        ]
        delta_history = [
            [] for _ in self.hidden_dims
        ]
        gate_history = [
            [] for _ in self.hidden_dims
        ]
        update_history = [
            [] for _ in self.hidden_dims
        ]

        for time_index in range(1, len(self.time_points)):
            previous_time = self.time_points[time_index -1]
            current_time = self.time_points[time_index]

            #time encoding
            time_embedding = self._get_time_embedding(
                reference_tensor=z,
                current_time=current_time,
                previous_time=previous_time
            )

            current_states = []
            vertical_input = z

            for level in range(len(self.hidden_dims)):
                previous_state = previous_states[level]
                temporal_block = self.temporal_blocks[level]

                if temporal_block is not None:
                    if self.use_time:
                        result = temporal_block(
                            (
                                vertical_input,
                                previous_state,
                                time_embedding
                            )
                        )
                    else:
                        result = temporal_block(
                            (
                                vertical_input,
                                previous_state
                            )
                        )

                    current_state = result["state"]

                    delta_history[level].append(result["delta"])
                    gate_history[level].append(result["gate"])
                    update_history[level].append(result["update"])

                else:
                    current_state = self.initial_layers[level](vertical_input)

                current_states.append(current_state)
                hidden_history[level].append(current_state)

                vertical_input = current_state

            prediction = self.output_head(current_states[-1])
            predictions.append(prediction)

            previous_states = current_states

        predictions = ops.stack(predictions, axis=1)

        if not return_details:
            return predictions

        return {
            "output": predictions,
            "hidden_states": hidden_history,
            "deltas": delta_history,
            "gates": gate_history,
            "updates": update_history
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "hidden_dims": self.hidden_dims,
            "time_points": self.time_points,
            "output_dim": self.output_dim,
            "time_dim": self.time_dim,
            "n_time_frequencies": self.n_time_frequencies,
            "temporal_levels": self.temporal_levels,
            "use_time": self.use_time,
            "use_gate": self.use_gate,
            "use_residual": self.use_residual,
            "initial_residual_scale": self.initial_residual_scale
        })

        return config

        




    


    