import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

# Load the MNIST dataset locally
data = np.load('/Users/akritgupta/Desktop/AKRIT/HDRS/mnist.npz')
x_train, y_train = data['x_train'], data['y_train']
x_test, y_test = data['x_test'], data['y_test']
data.close()

# Reshape and normalize as needed
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0


# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encoding of labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# Save the model
model.save('digit_recognition_cnn.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# from tensorflow.keras.models import load_model
# import numpy as np
# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
#
# # Load the trained model
# model = load_model('digit_recognition_cnn.h5')
#
# # Load and preprocess the uploaded image
# img_path = '/Users/akritgupta/Desktop/AKRIT/HDRS/3.png'  # Replace with your image filename
# img = Image.open(img_path).convert('L')  # Convert to grayscale
# img = ImageOps.invert(img)  # Invert colors (white background, black digit)
# img = img.resize((28, 28))  # Resize to 28x28
# img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Normalize pixel values
#
# # Display the processed image
# plt.imshow(img_array.reshape(28, 28), cmap='gray')
# plt.title("Processed Image")
# plt.show()
#
# # Predict the digit
# prediction = np.argmax(model.predict(img_array))
# print(f"Predicted Digit: {prediction}")
#
# prediction = np.argmax(model.predict(x_test[index].reshape(1, 28, 28, 1)))
# print(f"Predicted Digit: {prediction}")
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('digit_recognition_cnn.h5')

# Load and preprocess the uploaded image
img_path = '/Users/akritgupta/Desktop/AKRIT/HDRS/9.jpeg'  # Replace with your image filename
img = Image.open(img_path).convert('L')  # Convert to grayscale
img = ImageOps.invert(img)  # Invert colors (white background, black digit)
img = img.resize((28, 28))  # Resize to 28x28 pixels
img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Normalize pixel values

# Display the preprocessed image
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title("Processed Image")
plt.show()

# Predict the digit
prediction = np.argmax(model.predict(img_array))
print(f"Predicted Digit: {prediction}")
