from alz_prog_net.components.progression import DiseaseProgressionModel
import keras

model = DiseaseProgressionModel(
    latent_dim=256,
    hidden_dims= (128, 64, 32),
    time_points=(0, 6, 12, 24),
    temporal_levels=(True, True, False),
    output_dim=3,
    use_time=True,
    use_gate=True,
    use_residual=True,
    initial_residual_scale=0.1
)

x = keras.random.normal(shape=(32,256))
y=model(x)
print("output:", y)