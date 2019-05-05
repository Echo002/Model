import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from mobilenet import MobileNet
from keras.applications.imagenet_utils import decode_predictions

img_file = 'demo.png'
img = load_img(img_file, target_size=(32, 32))
image = image.img_to_array(img)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
model = MobileNet()
model.load_weights('mobileV1-lite.h5')

# model.summary()
result = model.predict(image)
print(result)