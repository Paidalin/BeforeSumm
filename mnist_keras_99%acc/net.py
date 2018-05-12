from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model

def mnistnet(input_shape):
    net = {}

    img_size = (input_shape[1], input_shape[0], 1)
    net['input'] = Input(shape=input_shape)
    # Block1
    net['conv1'] = Conv2D(32, 3,
                          activation='relu',
                          padding='same',
                          name='conv1')(net['input'])
    net['pool1'] = MaxPooling2D(2, 2, padding='same',
                                name='pool1')(net['conv1'])

    # Block2
    net['conv2'] = Conv2D(48, 3,
                          activation='relu',
                          padding='same',
                          name='conv2')(net['pool1'])
    net['pool2'] = MaxPooling2D(2, 2, padding='same',
                                name='pool2')(net['conv2'])

    # Block3
    net['conv3'] = Conv2D(64, 2,
                          activation='relu',
                          padding='same',
                          name='conv3')(net['pool2'])
    net['pool3'] = MaxPooling2D(2, 2, padding='same',
                                name='pool3')(net['conv3'])

    # fc层
    net['pool3_drop'] = Dropout(0.25, name='pool3_drop')(net['pool3'])
    net['pool3_drop_fla'] = Flatten(name='pool3_drop_fla')(net['pool3_drop'])
    net['fc1'] = Dense(2048, activation='relu', name='fc1')(net['pool3_drop_fla'])

    net['fc1_drop'] = Dropout(0.25, name='fc1_drop')(net['fc1'])
    net['fc2'] = Dense(1024, activation='relu', name='fc2')(net['fc1_drop'])

    net['out'] = Dense(10, activation='softmax', name='out')(net['fc2'])
    model = Model(net['input'], net['out'])
    return model