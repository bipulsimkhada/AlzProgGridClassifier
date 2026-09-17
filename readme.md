# registering gpu to tf

https://dev.to/metal3d/how-to-resolve-the-dlopen-problem-with-nvidia-and-pytorch-or-tensorflow-inside-a-virtual-env-181e


find .venv -name "*.so*" | grep nvidia | xargs dirname | sort -u

find .venv -name "*.so*" | grep nvidia | xargs dirname | sort -u | paste -d ":" -s -

export LD_LIBRARY_PATH=$(find .venv -name "*.so*" | grep nvidia | xargs dirname | sort -u | paste -d ":" -s -)


# to test
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"