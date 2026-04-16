from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
x_train_flat = x_train.reshape(60000, 784)       #28*28
print(x_train_flat.shape)
x_train_flat = x_train_flat/255.0
selected_images = []                            # store selected image data
selected_labels = []                            # store corresponding labels
for class_label in range(10):
    indices = np.where(y_train == class_label)[0]
    indices = indices[:500]                     # first 500 
    selected_images.append(x_train_flat[indices]) # select corresponding images
    selected_labels.append(y_train[indices])      # select corresponding labels
x_data = np.concatenate(selected_images, axis=0)  # combine class wise image arrays
y_data = np.concatenate(selected_labels, axis=0)  # combine class wise label arrays
print(x_data.shape)                               # shape of final image dataset
print(y_data.shape)                               # shape of final label
x_train_final = x_data[:4000]
y_train_final = y_data[:4000]

x_test_final = x_data[4000:]
y_test_final = y_data[4000:]

print(x_train_final.shape)
print(x_test_final.shape)
np.save('x_train_final_data.npy', x_train_final)
np.save('y_train_final_data.npy', y_train_final)
np.save('x_test_final_data.npy', x_test_final)
np.save('y_test_final_data.npy', y_test_final)

print("saved")


