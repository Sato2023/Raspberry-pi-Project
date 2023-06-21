import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import layers, models
import numpy as np
from keras.callbacks import EarlyStopping
datagen = ImageDataGenerator(rotation_range=10,
            rescale = 1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.1,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2  # set validation split to 20% 
            )

training_set = datagen.flow_from_directory("images",
                                              batch_size = 32,
                                              class_mode = 'categorical',
                                              target_size=(64,64),
                                              subset = 'training'
                                            )

test_set = datagen.flow_from_directory("images",
                                              batch_size = 32,
                                              class_mode = 'categorical',
                                              target_size=(64,64),
                                              subset = 'validation'
                                            )

training_set.class_indices

# Initializing the CNN
cnn = models.Sequential()

# Convolution, pooling, second layer pooling, flattening, dense
cnn.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(layers.Flatten())

#cnn.add(layers.Dense(units=128, activation='relu'))
cnn.add(layers.Dense(units=9, activation='sigmoid'))

#compile
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Implement early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)

#Train the model
cnn.fit(x = training_set, validation_data = test_set, epochs = 10, steps_per_epoch=100, validation_steps=50, callbacks=[early_stop])


#convert/compress
converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
open('object_recognition_fruit_v7.tflite', 'wb').write(converter.convert())
print('complete')

import keras.utils as image
test_image = image.load_img('images/apple/Image_10.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image /= 255
result = cnn.predict(test_image)
sample = np.argmax(result)
sample