import cv2
import tensorflow as tf

categories = ['Dog', 'Cat']

def prepare_images(file_path):
    img_size = 60
    img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)

model = tf.keras.models.load_model("64x3-CNN.model")

pred = model.predict([prepare_images('dog.jpg')])
print(categories[int(pred[0][0])])

pred = model.predict([prepare_images('cat.jpg')])
print(categories[int(pred[0][0])])