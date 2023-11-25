import tensorflowjs as tfjs
from keras.models import load_model

model = load_model('model1.h5')
tfjs.converters.save_keras_model(model, 'path/to/save')
