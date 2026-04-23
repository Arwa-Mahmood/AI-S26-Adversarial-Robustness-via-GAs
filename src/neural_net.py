import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical

# load the saved npy files, first 4800 for train, rest for test
x_train = np.load('x_train_final_data.npy')
x_test = np.load('x_test_final_data.npy')
y_train = np.load('y_train_final_data.npy')
y_test = np.load('y_test_final_data.npy')

print(x_train.shape)
print(x_test.shape)
print(y_test[:10])

# convert labels from single number to array of 10 numbers e.g 3 becomes [0,0,0,1,0,0,0,0,0,0]
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model = Sequential([
    Input(shape=(784,)),
    Dense(256, activation='relu'),          # hidden layer 1 - 256 neurons
    Dense(128, activation='relu'),                              # hidden layer 2 - 128 neurons
    Dense(10, activation='softmax')                             # output - 1 digit
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# adam adjusts weight on every guess
# categorical_crossentropy measures how wrong of a guess was
# accuracy prinitng

model.fit(x_train, y_train_cat, epochs=10, batch_size=32)
# epochs=10, it will go through data 10 times 
# batch_size=32 looks at 32 at one time

loss, accuracy = model.evaluate(x_test, y_test_cat)             # returns loss = how wrong it was and accuracy = how often it was right
print(f"Test Accuracy {accuracy* 100:.2f}%")                    # make it a percentage

def predict_nn(x):
    return np.argmax(model.predict(x), axis=1)                  # returns 10 confidence for each digit

# function that takes an image and returns confidence scores for all 10 classes
def predict_proba_nn(x):
    return model.predict(x)
    
model.save('nn_model.keras')
print('saved')