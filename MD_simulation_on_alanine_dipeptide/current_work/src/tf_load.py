import tensorflow as tf, os
import keras.backend as K
from keras.models import load_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True      # avoid tensorflow using all GPU memory
K.tensorflow_backend.set_session(tf.Session(config=config))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # disable tensorflow warning messages  (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
