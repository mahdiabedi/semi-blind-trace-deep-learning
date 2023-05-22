#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is a demo application of 
"Semi-blind-trace algorithm for self-supervised attenuation of vertically coherent noise"

For terminalogy and the definition of parameters refere to the paper.

Created on Sun Apr 30 20:00:47 2023

@author: mabedi
"""
import tensorflow as tf
import os
import time
import numpy as np
import scipy.io
# from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Concatenate, Conv2DTranspose, ReLU

#%% Loading data
used_folder = os.getcwd()# You can modify the folder address (Please unzip the provided data)
filename='/Data_with_noise_syn.npy'

X_train_with_noise = np.load(used_folder + filename)

sh = tf.shape(X_train_with_noise).numpy()
sc = sh[2]
st = sh[1]
ss = sh[0]

BUFFER_SIZE = ss // 2
BATCH_SIZE = 30
train_dataset = tf.data.Dataset.from_tensor_slices(X_train_with_noise)
train_dataset = train_dataset.shuffle(BUFFER_SIZE, seed=0, reshuffle_each_iteration=True).batch(BATCH_SIZE)
#%% Model
initializer = tf.random_normal_initializer(0., 0.02, seed=1)
X_input = Input(shape=[st, sc, 1])

# UNet
X = Conv2D(32, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(X_input)
X1 = ReLU()(X)

X = Conv2D(128, 3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(X1)
X = BatchNormalization()(X)
X2 = ReLU()(X)

X = Conv2D(256, 3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(X2)
X = BatchNormalization()(X)
X = ReLU()(X)

X = Conv2DTranspose(128, 3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(X)
X = BatchNormalization()(X)
X = ReLU()(X)
X = Concatenate()([X2, X])

X = Conv2DTranspose(32, 3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(X)
X = BatchNormalization()(X)
X = ReLU()(X)
X = Concatenate()([X1, X])

X = Conv2DTranspose(32, 3, strides=2, padding='same', kernel_initializer=initializer, use_bias=True)(X)
X = BatchNormalization()(X)
X = ReLU()(X)

X = Conv2D(1, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=True)(X)
X = tf.keras.activations.tanh(X)

model = Model(inputs=X_input, outputs=X)
model.summary()
initial_weights = model.get_weights()


#%% defining loss and training_step
@tf.function
def loss_sbt(output_f_dm, target, m_2):
    m2p = tf.abs(1 - m_2)
    metric_non_active = tf.reduce_mean(tf.multiply(m2p, tf.abs(target - output_f_dm))) / tf.reduce_mean(m2p) # L1 norm
    loss_active = tf.reduce_mean(tf.multiply(m_2, tf.abs(target - output_f_dm))) / tf.reduce_mean(m_2) # L1 norm
    return metric_non_active, loss_active

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(input_d, target, m_2):
    with tf.GradientTape() as gen_tape:
        output_f_dm = model(input_d)
        metric_non_active, loss_active = loss_sbt(output_f_dm, target, m_2)
        
    model_gradients = gen_tape.gradient(loss_active, model.trainable_variables)
    optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
    return metric_non_active, loss_active

model.compile(optimizer=optimizer, loss=loss_sbt)

@tf.function
def masking(m_1,b_batch,d_batch):
    # Masking the input to the network
    d_m = tf.math.multiply(m_1, b_batch) + tf.math.multiply((1 - m_1), d_batch)
    return d_m
# %% custom training loop

def fit(dataset, epochs, num_of_masked_traces=15, epsilon=0.05, partly_noisy=False, mask_realizations=50):
    
    # num_of_masked_traces: The number of traces masked in each data sample in each training step
    # epsilon:              The relative weight of non-active traces
    # partly_noisy:         True = noise only covers the bottom part of traces
    # mask_realizations:    The num of different masks that are applied to each batch in each training step
    
    train_history_non_active = []
    train_history_active = []
    #to reset the network:
    tf.keras.backend.clear_session()
    model=Model(inputs=X_input, outputs=X)
    model.set_weights(initial_weights)        
    
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}, n = {num_of_masked_traces}, epsilone = {epsilon}")
        start_time = time.time()
        metric_non_active_vec = []
        loss_active_vec = []

        for steps, (d_batch) in enumerate(dataset):
            shap = np.shape(d_batch)
            max_abs_value = tf.expand_dims(tf.expand_dims(tf.math.reduce_max(tf.math.abs(d_batch), axis=[1, 3]), axis=1), axis=3)
            b_batch = tf.random.uniform(shape=shap, minval=-1, maxval=1, dtype=tf.dtypes.float32)

            # Apply gaussian filtering:
            sig_window = np.random.uniform(low=.01, high=3.0)
            b_batch = scipy.ndimage.gaussian_filter1d(b_batch, sigma=sig_window, axis=1)

            # Apply bandpass filtering:
            # low_band = np.random.uniform(low=1, high=30)
            # b, a = butter(5, [low_band, 2*low_band], fs=1/0.006, btype='band')
            # b_batch = lfilter(b, a, b_batch, axis=1)
            # b_batch = tf.dtypes.cast(b_batch, tf.float32)
            
            # Normalizing the noise to data amplidues range
            maxabs_b_batch = tf.expand_dims(tf.expand_dims(tf.math.reduce_max(tf.math.abs(b_batch), axis=[1, 3]), axis=1), axis=3)
            b_batch = tf.math.multiply(tf.math.divide(b_batch, maxabs_b_batch), max_abs_value)
            
            # To limit the temporal length of the noise to the bottom of data (partly noisy traces)
            if partly_noisy:
                start_vector = tf.random.uniform(shape=[shap[0], shap[2], shap[3]], minval=-20, maxval=-10, dtype=tf.dtypes.float32)
                end_vector = tf.random.uniform(shape=[shap[0], shap[2], shap[3]], minval=10, maxval=20, dtype=tf.dtypes.float32)
                mask_temporal = tf.math.sigmoid(tf.linspace(start_vector, end_vector, st))
                mask_temporal = tf.transpose(mask_temporal, perm=[1, 0, 2, 3])
                        
            for step in range(mask_realizations): # num of different masks that are applied to each batch in each training step 
                # Indeces of Masked traces   
                indices = np.sort(np.random.choice(range(0, sc), num_of_masked_traces, replace=False)) # Random integer without repeat

                # Tapered traces in loss
                indices_m_1 = np.append(indices-1, indices-2)
                indices_m_1 = indices_m_1[indices_m_1>=0]
                indices_p_1 = np.append(indices+1, indices+2)
                indices_p_1 = indices_p_1[indices_p_1<=sc-1]

                # Mask
                m_1 = np.zeros((sc,1))
                m_1[indices] = 1       
                m_1 = tf.constant(m_1,dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
                
                if partly_noisy:
                    m_1 = tf.math.multiply(m_1, mask_temporal) # To limit the temporal length of the noise (partly noisy traces)
                
                # Active zones for calculation of loss
                m_2 = np.zeros((sc, 1))
                m_2[indices_m_1] = epsilon 
                m_2[indices_p_1] = epsilon 
                m_2[indices] = 1       
                m_2 = tf.constant(m_2, dtype=tf.float32)[tf.newaxis, tf.newaxis, :]
                
                d_m = masking(m_1,b_batch,d_batch)   
                # Applying the training step
                metric_non_active, loss_active = train_step(d_m, d_batch, m_2)
                metric_non_active_vec.append(metric_non_active)
                loss_active_vec.append(loss_active)

            print('.', end='', flush=True)

        # Epoch losses
        metric_non_active_mean = tf.reduce_mean(metric_non_active_vec)
        loss_active_mean = tf.reduce_mean(loss_active_vec)

        print("Active loss: %.6f, NonActive metric: %.6f, Time taken: %.1fs" 
              % (float(loss_active_mean), float(metric_non_active_mean), time.time() - start_time), end='\n')
        
        train_history_non_active.append(float(metric_non_active_mean))
        train_history_active.append(float(loss_active_mean))

        if epoch % 5 == 0: #Saving the trained model and history every 5 epoches
            print('Saving...')
            model.save(str(used_folder)+'/Results'+'/model_epoch_'+str(epoch))
            history = {"NonActive_metric": train_history_non_active, "Loss": train_history_active}
            np.save(str(used_folder)+'/Results'+'/history.npy', history)
            
    return history

#%% Training the network

os.system('rm -rfv '+str(used_folder)+'/Results'+'/*')#Empty the save folder before starting the training
print(sh)
history = fit(train_dataset, epochs=30, num_of_masked_traces=20, epsilon=0.1, partly_noisy=True)

#%% Plotting the history

history = np.load(str(used_folder)+'/Results'+'/history.npy', allow_pickle='TRUE').item()

fig = plt.figure()
plt.plot(history['Loss'][:], 'k-')
plt.plot(history['NonActive_metric'][:], 'b-')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(['Active trace', 'Non-active traces'])

#%% Inference

Model_trained = tf.keras.models.load_model(str(used_folder)+'/Results'+'/model_epoch_30', compile=False)
X_predict = Model_trained.predict(X_train_with_noise)

sample_number=20

plt.figure()
caxis = [-1, 1]
plt.subplot(131)
plt.imshow(tf.squeeze(X_train_with_noise[sample_number, :, :, :]), cmap='seismic_r', aspect='auto', interpolation='none')
plt.clim(caxis)
plt.title('Input (noisy)')

plt.subplot(132)
plt.imshow(tf.squeeze(X_predict[sample_number, :, :, :]), cmap='seismic_r', aspect='auto', interpolation='none')
plt.clim(caxis)
plt.title('Predict')

plt.subplot(133)
plt.imshow(tf.squeeze(X_train_with_noise[sample_number, :, :, :] - X_predict[sample_number, :, :, :]), cmap='seismic_r', aspect='auto', interpolation='none')
plt.clim(caxis)
plt.title('Difference')
plt.colorbar()
