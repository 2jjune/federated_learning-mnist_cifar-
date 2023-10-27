import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import create_model
import set_epoch_num
import set_mnist_generator
import set_cifar_generator
import set_generator
import numpy as np
import gc
# from numba import cuda
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

set_epoch = set_epoch_num.set_epoch
set_num = set_epoch_num.set_num
print(set_epoch, set_num)
train_step = set_epoch_num.train_step
model = create_model.create_model()

# for i in range(set_num):
#     model.append(create_model.create_model())
    # print(model[i])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# for i in range(set_num):
#     model[i].compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
    # model[i].summary()


train_dir = set_mnist_generator.train_dir
mnist_train_generator, validation_generator = set_mnist_generator.set_generator(train_dir)
x=0

'''
tmp=[]
for i in range(int(set_num/2)):
    # model_i = model[i]
    print()

    if x+1>10:
        x = int(x/10)-1
    print(i,"*******************************")
    print(train_dir + str(x+1))
    # train_generator, validation_generator = set_mnist_generator.set_generator(train_dir+str(x+1))
    x+=1


    print('학습 데이터 수 :', train_step*64)
    model.load_weights('first_rgb.h5')

    # model[0].fit(mnist_train_generator, steps_per_epoch=len(train_generator), epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
    hist = model.fit(mnist_train_generator, steps_per_epoch=train_step, epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
    model.save_weights('{}_epoch {}.h5'.format(set_epoch,i+1))
    # print(hist.history['val_accuracy'])
    # print(np.array(hist.history['val_accuracy']).mean(axis=0))
    tmp.append(hist.history['val_accuracy'][-1])

f = open('result.txt', 'w')
data = "mnist accuracy : %f\n" % np.array(tmp).mean(axis=0)
f.write(data)
f.close()

#---------------------------------------------
tmp=[]'''

train_dir = set_cifar_generator.train_dir
cifar_train_generator, validation_generator = set_cifar_generator.set_generator(train_dir)
'''
for i in range(int(set_num/2), set_num):
    # model_i = model[i]
    print()

    if x+1>10:
        x = int(x/10)-1
    print(i,"*******************************")
    print(train_dir + str(x+1))
    # train_generator, validation_generator = set_cifar_generator.set_generator(train_dir+str(x+1))
    x+=1

    print('학습 데이터 수 :', train_step*64)
    model.load_weights('first_rgb.h5')
    # model[0].fit(cifar_train_generator, steps_per_epoch=len(train_generator), epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
    hist = model.fit(cifar_train_generator, steps_per_epoch=train_step, epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
    model.save_weights('{}_epoch {}.h5'.format(set_epoch,i+1))
    tmp.append(hist.history['val_accuracy'][-1])
f = open('result.txt', 'a')
data = "cifar accuracy : %f\n"%np.array(tmp).mean(axis=0)
f.write(data)
f.close()
del tmp
del model'''
test_generator = set_generator.test_generator

for x in range(1000):
    tmp_model = []
    weights = []
    new_weights_median = list()
    new_weights_avg = list()

    for i in range(set_num):
        tmp_model.append(create_model.create_model())

    for i in range(set_num):
        if i < 10:
            tmp_model[i].load_weights("{}_epoch {}.h5".format(set_epoch, i + 1))
            print("{}_epoch {}.h5".format(set_epoch, i + 1))
        # else:
        #     tmp_model[i].load_weights("{}_epoch {}.h5".format(set_epoch, i + 1))
        #     print("{}_epoch {}.h5".format(set_epoch, i + 1))

    for i in range(set_num):
        weights.append(tmp_model[i].get_weights())
    print('weights 갯수 : ', len(weights))

    for weights_list_tuple in zip(*weights):
        # new_weights_median.append(
        #     np.array([np.median(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
        # )

        new_weights_avg.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )

    # model[0].set_weights(new_weights_median)
    model = create_model.create_model()
    model.set_weights(new_weights_avg)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    test_loss, test_acc = model.evaluate(test_generator)
    print()
    # print('---------------{}번째 median 테스트 정확도: {}, loss: {}-----------------'.format(x, test_acc, test_loss))
    print('---------------{}번째 avg 테스트 정확도: {}, loss: {}-----------------'.format(x, test_acc, test_loss))
    f = open('result.txt', 'a')
    # data = "%d번 median result accuracy : %f, loss : %f\n\n" % (x,test_acc, test_loss)
    data = "%d번 avg result accuracy : %f, loss : %f\n\n" % (x+23,test_acc, test_loss)
    f.write(data)
    f.close()
    del model
    print()
    print('dataset retraining...')
    tmp=[]
    for i in range(int(set_num / 2)):
        print()
        print(i, "*******************************")

        print('학습 데이터 수 :', train_step * 64)
        model=create_model.create_model()
        model.set_weights(new_weights_avg)
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        # model[0].fit(mnist_train_generator, steps_per_epoch=len(train_generator), epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
        hist= model.fit(mnist_train_generator, steps_per_epoch=train_step, epochs=set_epoch,
                     validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
        model.save_weights('{}_epoch {}.h5'.format(set_epoch, i + 1))
        tmp.append(hist.history['val_accuracy'][-1])
    f = open('result.txt', 'a')
    data = "%d번 mnist accuracy : %f\n" % (x+1+23, np.array(tmp).mean(axis=0))
    f.write(data)
    f.close()
    del model
    tmp=[]
    for i in range(int(set_num / 2), set_num):
        print()

        print(i, "*******************************")

        print('학습 데이터 수 :', train_step * 64)
        model=create_model.create_model()
        model.set_weights(new_weights_avg)
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=set_epoch_num.lr, decay=set_epoch_num.decay),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        # model[0].fit(cifar_train_generator, steps_per_epoch=len(train_generator), epochs=set_epoch, validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
        hist=model.fit(cifar_train_generator, steps_per_epoch=train_step, epochs=set_epoch,
                     validation_data=set_generator.test_generator, validation_steps=len(set_generator.test_generator))
        model.save_weights('{}_epoch {}.h5'.format(set_epoch, i + 1))
        tmp.append(hist.history['val_accuracy'][-1])
    f = open('result.txt', 'a')
    data = "%d번 cifar accuracy : %f\n" % (x+1+23, np.array(tmp).mean(axis=0))
    f.write(data)
    f.close()

    del model
    del tmp_model
    del weights
    del new_weights_median
    del new_weights_avg
    del tmp
    del data
    del hist
    # cuda.select_device(0)
    # cuda.close()
    gc.collect()
    tf.keras.backend.clear_session()

# 1. MNIST 데이터셋 임포트
# def set_mnist():
#     mnist = tf.keras.datasets.fashion_mnist
#     (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#     training_images = training_images.reshape(60000, 28, 28, 1)
#     training_images = training_images / 255.0
#     test_images = test_images.reshape(10000, 28, 28, 1)
#     test_images = test_images / 255.0
#
#     training_labels = to_categorical(training_labels)
#     test_labels = to_categorical(test_labels)
#     return training_images, training_labels
# 3. 모델 구성
