from experiments.loss import run_loss_stage
import tensorflow as tf
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(devices))
print(devices)

for device in devices:
    tf.config.experimental.set_memory_growth(device,True)

def main():
    run_loss_stage("stage_1_severity")

if __name__ == "__main__":
    main()