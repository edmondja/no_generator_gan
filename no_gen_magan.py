import gym, time, os, math, itertools # Many useless imports
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pandas as pd
import numpy as np
np.random.seed(8)
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint as sp_randint
from collections import deque
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from PIL import Image
from keras.optimizers import Nadam, SGD
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input, Activation, BatchNormalization
from keras.layers.merge import concatenate, add, average
from keras.models import Model, Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, SeparableConv2D, \
                         Lambda, AveragePooling2D, MaxPooling2D, Cropping2D, \
                         Conv2DTranspose, UpSampling2D, SpatialDropout2D, \
                         DepthwiseConv2D, Reshape, ELU, GRU
from keras.callbacks import EarlyStopping
from copy import deepcopy
from sklearn import mixture
import mnist
from keras import backend as K
from sklearn.model_selection import KFold

mish = lambda x: x * K.tanh(K.softplus(x))

def nadam_numpy(t, gt, prod_mus_t, mu_t, mt_1, nt_1):
    mu, v = .99, 0.999
    mu_t_p1 = mu*(1-.5*(.96**((t+1)/250)))
    prod_mus_t_p1 = prod_mus_t * mu_t_p1
    
    if t == 0:
        return gt, prod_mus_t_p1, mu_t_p1, mt_1, nt_1
    else:        
        g_hat = gt / (1 - prod_mus_t)
        
        mt = mu * mt_1 + (1-mu)*gt
        m_hat = mt / (1-prod_mus_t_p1)
        
        nt = v * nt_1 + (1-v) * (gt**2)
        n_hat = nt / (1-v**t)
        
        m_bar = (1-mu_t) * g_hat + mu_t_p1 * m_hat
        new_grad = m_bar / (np.sqrt(n_hat) + 1e-08)
        
        return new_grad, prod_mus_t_p1, mu_t_p1, mt, nt

def mean_pred(y_true, y_pred):
    return K.sqrt(K.sum(y_pred, axis=[i for i in range(1, len(y_true.shape))]))

def hinge(y_true, y_pred):
    m = 1. * y_true[-1][-1][-1]
    y_true_norm = K.mean(y_true / m, axis=[i for i in range(1, len(y_true.shape))])
    obs_y_pred = K.sqrt(K.sum(y_pred, axis=[i for i in range(1, len(y_true.shape))]))
    
    fake_part = y_true_norm * K.maximum(m - obs_y_pred, 0.)
    true_part = (1 - y_true_norm) * obs_y_pred
    
    return fake_part + true_part

train = mnist.train_images() / 256
y_train = np.vstack((train * 0, train * 0 + 1))
train = np.vstack((train, np.random.random((train.shape))))
train =  (train - 0.5) * 2
test = mnist.test_images() / 256
y_test = np.vstack((test * 0, test * 0 + 1))
test = np.vstack((test, np.random.random((test.shape))))
test = (test - 0.5) * 2

train_noise = np.random.normal(0, 0.001, 
                               size=train.shape[0]*train.shape[1]
                               *train.shape[2]).reshape(train.shape)
test_noise = np.random.normal(0, 0.001, 
                              size=test.shape[0]*test.shape[1]
                              *test.shape[2]).reshape(test.shape)
train, test = np.clip(train+train_noise, -0.99999, .99999), np.clip(test+test_noise, -0.99999, .99999)

# ML
pics = Input(shape=train[0].shape, dtype='float32', name='pics')

conv1 = Dense(256, activation=mish)(Flatten()(pics))
conv2 = Dense(50, activation='linear')(conv1)
code = Dense(784, activation=mish)(conv2)
dc1 = Reshape((28, 28))(Dense(784, activation='tanh')(code))

sq_error = Lambda(lambda x: K.square(x[0]-x[1]))([dc1, pics])

critic = Model(pics, sq_error, name="critic")
critic.summary()
critic.trainable = False

crit_p = critic(Activation('tanh')(pics))
tuning = Model(pics, crit_p, name="tuning")
tuning.compile(loss=mean_pred, optimizer=SGD(lr=5.0)) #)
tuning.summary()

loss = K.sum(tuning.output)
grads = K.gradients(loss, tuning.input)[0]

# this function returns the loss and grads given the input picture
iterate = K.function([tuning.input], [loss, grads])

crit_preds = 0
critic.trainable = True
critic.compile(loss=mean_pred, optimizer='nadam')
critic.fit(train[:60000], y_train[:60000], batch_size=1000, epochs=2, verbose=1)
critic.compile(loss=hinge, optimizer='nadam')

m = np.mean(np.sqrt(np.sum(np.sum(critic.predict(test[:10000]), axis=1), axis=1)))

bsize = 1000
nsplits = int(train.shape[0]/(int(bsize)))
y_train = train[:bsize] * 0
y_train[int(bsize/2):] = y_train[int(bsize/2):] + 1
old_gen_loss = np.inf
tmp_train_images = np.arctanh(np.clip(train[:60000], -0.99999, .99999))#+train_noise
tmp_test_images = np.arctanh(np.clip(test[10000:], -0.99999, .99999))#+test_noise
tmp_images = np.concatenate((tmp_train_images, tmp_test_images))
momentum_init = tmp_images * 0
prod_mus_t, mu_t, mt, nt = 1, 0, momentum_init, momentum_init

for episode in range(0, 10000):
    kf = KFold(n_splits=nsplits, shuffle=True)
    train_indexes = list(kf.split(train[:60000]))
    test_indexes = list(kf.split(train[:60000]))
    for k, (_, indexes) in enumerate(train_indexes):
        # Train critic
        real_images = train[indexes]
        fake_images = train[test_indexes[k][1]+60000]
        crit_batch_loss = critic.train_on_batch(np.concatenate((real_images, fake_images)), 
                                                y_train * m)
        
        # Train gen
        #train_noise = np.random.normal(0, .1, size=60000*28*28).reshape((60000,28,28))
        #test_noise = np.random.normal(0, .1, size=10000*28*28).reshape((10000,28,28))
        tmp_train_images = np.arctanh(np.clip(train[60000:], -0.99999, .99999))#+train_noise
        tmp_test_images = np.arctanh(np.clip(test[10000:], -0.99999, .99999))#+test_noise
        tmp_images = np.concatenate((tmp_train_images, tmp_test_images))
    
        loss_value, grads_value = iterate([tmp_images])
        
        new_grad, prod_mus_t, mu_t, mt, nt = nadam_numpy((k+1)*episode, grads_value, 
                                                         prod_mus_t, mu_t, mt, nt)
        tmp_images -= new_grad * 0.1
        formed_images = np.tanh(tmp_images)
        assert np.sum(np.isnan(tmp_images)*1)==0
        
        train[60000:], test[10000:] = formed_images[:-10000], formed_images[-10000:]
        
        if k%10==0:
            print({'gen_batch_loss':loss_value, 'crit_batch_loss': crit_batch_loss, 'm': m})
        
    threshold = np.mean(np.sqrt(np.sum(np.sum(critic.predict(test[:10000]), axis=1), axis=1)))
    gen_losses = np.sqrt(np.sum(np.sum(critic.predict(test[10000:]), axis=1), axis=1))
    gen_loss = np.mean(gen_losses)
    print({'threshold': threshold, 'gen_loss': gen_loss,
           'successful_attacks_rate': 
               np.mean(gen_losses<threshold*1)})
    # Save pics
    for i in range(0, 100):
        img = Image.fromarray(np.uint8((formed_images[-10000:][i]/2+0.5)*256), 'L')
        name = 'edmond_no_generator_gan_result_'+str(i)+'.png'
        img.save(name)
    
    if (threshold < m) and (threshold < gen_loss) and (old_gen_loss < gen_loss):
        m = threshold
    
    old_gen_loss = gen_loss