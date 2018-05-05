import numpy as np
import pandas as pd
import random
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, TimeDistributed, Input, ConvLSTM2D, Conv3D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

# read data
origin_data = np.loadtxt(open('LSTM_CSV_data/LSTM_data_bu.csv'), delimiter = ',', skiprows=0)

img_rows = 108
img_cols = 192
TIME_STEPS = 5

data = origin_data[:, 0:img_rows*img_cols]
origin_label = origin_data[:, img_rows*img_cols]
data = data.reshape(data.shape[0]//TIME_STEPS, TIME_STEPS, img_rows, img_cols, 1)        
# label = label.reshape(label.shape[0]//TIME_STEPS, TIME_STEPS) # , 1, 1, 1)
label = np.zeros([origin_label.shape[0]//TIME_STEPS, 5, img_rows, img_cols, 1])

for i in np.where(origin_label==1):
    label[i//5,:,:,:, :] = 1
# Disarrange the order of data and label
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]

# Take data as general training data, the other half as test data
train_num = (int)(data.shape[0]*0.5)
train_data = data[0:train_num:, :, :, :]
train_label = label[0:train_num, :, :, :, :]
test_data = data[train_num:data.shape[0], :, :, :]
test_label = label[train_num:data.shape[0], :, :, :, :]

# one-hot label
train_label = to_categorical(np.asarray(train_label))
test_label = to_categorical(np.asarray(test_label))

print('shape of data tensor', train_data.shape)
print('shape of label tensor', train_label.shape)

# Hyperparameter
BATCH_SIZE = 50
CELL_SIZE = 256                                        
OUTPUT_SIZE = 2
LR = 1E-3

# create the model
print('Build model...')
model = Sequential()

# build the model


# compiling
adam = Adam(LR)
model.compile(
        loss='binary_crossentropy
        optimizer='adadelta',                                 
        metrics = ['accuracy']              
)

# training
print('Train...')
model.fit(
        train_data, 
        train_label, 
        batch_size = BATCH_SIZE, 
        epochs=5, 
        validation_data = (test_data, test_label)
        )

# evaluate
score, acc = model.evaluate(test_data,
                            test_label,
                            batch_size = BATCH_SIZE)

# predict
predict_label = model.predict(test_data, batch_size = BATCH_SIZE)


print('test score:', score)
print('test accuracy', acc)