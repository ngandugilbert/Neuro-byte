# In this tutorial, I am creating a model that Identifies items from the fashion mnist data set
# Import the tensorflow api

import tensorflow as tf

# get the dataset as train and test data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# print the shapes of the datat sets
# print(f'Training data shape: {x_train.shape}')
# print(f'Test data shape: {x_train.shape}')

# Normalize to the dataset to be between 1 and 0

x_train = x_train/ 255.0
test = x_test/ 255.0

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Build the model
# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu')])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Make predictions on new, unseen data
predictions = model.predict(new_data)



