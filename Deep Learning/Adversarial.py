from __future__ import print_function

import os,sys
import numpy as np
import scipy.io as scio
import tensorflow as tf
import keras
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, TimeDistributed, Bidirectional, Multiply, Permute, RepeatVector, Concatenate, Dot, Lambda
from keras.models import Model, load_model
import keras.backend as K
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.backend import set_session 
from sklearn.model_selection import train_test_split


# Parameters
use_existing_model = False
fraction_for_test = 0.1
data_dir = '/srv/node/NAS/NAS/zhangyi/TutorialData/DFS/20181130/'
ALL_MOTION = [1,2,3,4,5,6]
N_MOTION = len(ALL_MOTION)
N_LOCATION = 5
T_MAX = 0
n_epochs = 100
f_dropout_ratio = 0.5
n_gru_hidden_units = 64
n_batch_size = 32
f_learning_rate = 0.001
loss_weight_domain = 1

def normalize_data(data_1):
    # data(ndarray)=>data_norm(ndarray): [6,121,T]=>[6,121,T]
    data_1_max = np.amax(data_1,(0,1),keepdims=True)    # [6,121,T]=>[1,1,T]
    data_1_min = np.amin(data_1,(0,1),keepdims=True)    # [6,121,T]=>[1,1,T]
    data_1_max_rep = np.tile(data_1_max,(data_1.shape[0],data_1.shape[1],1))    # [1,1,T]=>[6,121,T]
    data_1_min_rep = np.tile(data_1_min,(data_1.shape[0],data_1.shape[1],1))    # [1,1,T]=>[6,121,T]
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep + sys.float_info.min)
    return  data_1_norm

def zero_padding(data, T_MAX):
    # data(list)=>data_pad(ndarray): [6,121,T1/T2/...]=>[6,121,T_MAX]
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0,0),(0,0),(T_MAX - t,0)), 'constant', constant_values = 0).tolist())
    return np.array(data_pad)

def onehot_encoding(label, num_class):
    # label(ndarray)=>_label(ndarray): [N,]=>[N,num_class]
    label = np.array(label).astype('int32')
    label = np.squeeze(label)
    _label = np.eye(num_class)[label-1]
    return _label

def load_data(path_to_data):
    global T_MAX
    data = []
    label = []
    domain = []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:

            file_path = os.path.join(data_root,data_file_name)
            try:
                data_1 = scio.loadmat(file_path)['doppler_spectrum']    # [6,121,T]
                label_1 = int(data_file_name.split('-')[1])
                location_1 = int(data_file_name.split('-')[2])
                orientation_1 = int(data_file_name.split('-')[3])
                repetition_1 = int(data_file_name.split('-')[4])
                
                # Downsample
                data_1 = data_1[:,:,0::10]

                # Select Motion
                if (label_1 not in ALL_MOTION):
                    continue

                # Normalization
                data_normed_1 = normalize_data(data_1)
                
                # Update T_MAX
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]                
            except Exception:
                continue

            # Save List
            data.append(data_normed_1.tolist())
            label.append(label_1)
            domain.append(location_1)
            
    # Zero-padding
    data = zero_padding(data, T_MAX)

    # Swap axes
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)   # [N,6,121,T_MAX]=>[N,T_MAX,6,121]
    data = np.expand_dims(data, axis=-1)                # [N,T_MAX,6,121]=>[N,T_MAX,6,121,1]

    # Convert label and domain to ndarray
    label = np.array(label)
    domain = np.array(domain)

    # data(ndarray): [N,T_MAX,6,121,1], label(ndarray): [N,], domain(ndarray): [N,]
    return data, label, domain

def custom_loss_label():
    def lossfn(y_true, y_pred):
        myloss_batch = -1 * K.sum(y_true*K.log(y_pred+K.epsilon()), axis=-1, keepdims=False)
        myloss = K.mean(myloss_batch, axis=-1, keepdims=False)
        return myloss
    return lossfn

def custom_loss_domain():
    def lossfn(y_true, y_pred):
        myloss_batch = -1 * K.sum(y_true*K.log(y_pred+K.epsilon()), axis=-1, keepdims=False)
        myloss = K.mean(myloss_batch, axis=-1, keepdims=False)
        return myloss
    return lossfn

def assemble_model(input_shape, n_class, n_domain):
    model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')    # (@,T_MAX,6,121,1)

    # CNN+RNN+Adversarial
    x = TimeDistributed(Conv2D(16,kernel_size=(3,6),activation='relu',data_format='channels_last',\
        input_shape=input_shape))(model_input)              # (@,T_MAX,6,121,1)=>(@,T_MAX,4,116,16)
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)   # (@,T_MAX,4,116,16)=>(@,T_MAX,2,58,16)

    x = TimeDistributed(Flatten())(x)                       # (@,T_MAX,2,58,16)=>(@,T_MAX,2*58*16)
    x = TimeDistributed(Dense(128,activation='relu'))(x)    # (@,T_MAX,2*58*16)=>(@,T_MAX,128)
    x = TimeDistributed(Dropout(f_dropout_ratio))(x)
    x = TimeDistributed(Dense(64,activation='relu'))(x)     # (@,T_MAX,128)=>(@,T_MAX,64)

    x = GRU(n_gru_hidden_units,return_sequences=False)(x)   # (@,T_MAX,64)=>(@,64)
    x_feat = Dropout(f_dropout_ratio)(x)

    # Label prediction part
    x_1 = Dense(64, activation='relu')(x_feat)      # (@,64)=>(@,64)
    x_1 = Dense(32, activation='relu')(x_1)         # (@,64)=>(@,32)
    model_output_label = Dense(n_class, activation='softmax', name='name_model_output_label')(x_1)      # (@,32)=>(@,n_class)

    # Domain prediction part
    x_2 = Dense(64, activation='relu')(x_feat)      # (@,64)=>(@,64)
    x_2 = Dense(32, activation='relu')(x_2)         # (@,64)=>(@,32)
    model_output_domain = Dense(n_domain, activation='softmax', name='name_model_output_domain')(x_2)   # (@,32)=>(@,n_domain)


    model = Model(inputs=model_input, outputs=[model_output_label, model_output_domain])
    model.compile(optimizer=keras.optimizers.RMSprop(lr=f_learning_rate),
            loss = {'name_model_output_label':custom_loss_label(), 'name_model_output_domain':custom_loss_domain()},
            loss_weights={'name_model_output_label':1, 'name_model_output_domain':-1*loss_weight_domain},
            metrics={'name_model_output_label':'accuracy', 'name_model_output_domain':'accuracy'}
            )
    
    return model

# ==============================================================
# Let's BEGIN >>>>
if len(sys.argv) < 2:
    print('Please specify GPU sequence...')
    exit(0)
if (sys.argv[1] == '1' or sys.argv[1] == '0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    tf.compat.v1.set_random_seed(1)
else:
    print('Wrong GPU number, 0 or 1 supported!')
    exit(0)

# Load data
data, label, domain = load_data(data_dir)
print('\nLoaded dataset of ' + str(label.shape[0]) + ' samples, each sized ' + str(data[0,:,:,:,:].shape) + '\n')

# Split train and test
[data_train, data_test, label_train, label_test, domain_train, domain_test] = train_test_split(data, label, domain, test_size=fraction_for_test)
print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' +\
    'Test on ' + str(label_test.shape[0]) + ' samples\n')

# One-hot encoding for train data
label_train = onehot_encoding(label_train, N_MOTION)
domain_train = onehot_encoding(domain_train, N_LOCATION)

# Train Model
model = assemble_model(input_shape=(T_MAX, 6, 121, 1), n_class=N_MOTION, n_domain=N_LOCATION)
model.summary()
model.fit({'name_model_input': data_train},{'name_model_output_label': label_train, 'name_model_output_domain': domain_train},
        batch_size=n_batch_size,
        epochs=n_epochs,
        verbose=1,
        validation_split=0.1, shuffle=True)

# Testing...
print('Testing...')
[label_test_pred,_] = model.predict(data_test)
label_test_pred = np.argmax(label_test_pred, axis = -1) + 1

# Confusion Matrix
cm = confusion_matrix(label_test, label_test_pred)
print(cm)
cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm = np.around(cm, decimals=2)
print(cm)

# Accuracy
test_accuracy = np.sum(label_test == label_test_pred) / (label_test.shape[0])
print(test_accuracy)