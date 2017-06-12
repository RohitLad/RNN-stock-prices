from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

path = 'path of the image to be predicted'
# Initialize stuff
model = Sequential()

# Convolution layer1
model.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))

# Pooling layer1
model.add(MaxPool2D(pool_size = (2,2)))

#Convolution layer 2
model.add(Conv2D(32,(3,3),activation = 'relu'))

# Pooling layer 2
model.add(MaxPool2D(pool_size=(2,2)))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compile
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Image fitting

train_data = ImageDataGenerator(
                                rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

test_data = ImageDataGenerator(rescale = 1./255)

training_set = train_data.flow_from_directory('dataset/training_set',
                                              target_size = (64,64),
                                              batch_size = 32,
                                              class_mode = 'binary')

test_set = test_data.flow_from_directory('dataset/test_set',
                                         target_size = (64,64),
                                         batch_size = 32,
                                         class_mode = 'binary')

model.fit_generator(training_set,
                    steps_per_epoch = 8000,
                    epochs = 25,
                    validation_data = test_set,
                    validation_steps = 2000)

# Predictions

test_image = image.load_img(path, target_size = (64,64))
test_image = np.expand_dims(test_image,axis=0)
test_image = image.img_to_array(test_image)
result = model.predict(test_image)
training_set.class_indices