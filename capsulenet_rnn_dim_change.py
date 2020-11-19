"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""
"""
Demo of network with 5x5 convolutional layer, two 3x3 caps layers with
capsule-wise convolution and no routing and a capslayer with routing
Created on Sat Nov 24 16:35:22 2017
@author: - Ruslan Grimov
"""

from keras import backend as K
from keras import layers, models, optimizers
from keras.datasets import mnist, cifar10
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Conv2D, Activation, Dense, Dropout, Lambda, Reshape, Concatenate
from keras.layers import BatchNormalization, MaxPooling2D, Flatten, Conv1D, Deconvolution2D, Conv2DTranspose
from capslayers import ConvertToCaps, Conv2DCaps, FlattenCaps
from capslayers import DenseCaps, CapsToScalars
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras import regularizers
from keras import losses
import numpy as np
import tensorflow as tf
import os
from snapshot import SnapshotCallbackBuilder
import capslayers
# import rescaps
from keras.utils import plot_model
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv3D
# import memory_saving_gradients
from keras import initializers
from keras.utils.conv_utils import conv_output_length, deconv_length
from keras.models import Model, Sequential, load_model
import ema
import os
import sys
# from rescaps_v3D import *
from keras.utils import multi_gpu_model
import numpy as np
from keras import layers, models,activations
import matplotlib.pyplot as plt
from utils import combine_images_1d
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from sklearn.model_selection import train_test_split
from capslayers import *
# K.set_image_data_format('channels_last')




def margin_loss(y_true, y_pred):
    # L= y_true * K.clip(0.9 - y_pred, 0, 1) ** 2 + 0.5 * (1 - y_true) * K.clip(y_pred - 0.1, 0, 1) ** 2
    # L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1)) +  K.square(K.maximum(0.,y_pred))
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.1 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape[0], tuple):  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])



class ConvCapsuleLayer3(layers.Layer):

    def __init__(self, kernel_size, num_capsule, num_atoms, strides=1, padding='valid', routings=3,
                 kernel_initializer='he_normal', **kwargs):
        super(ConvCapsuleLayer3, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) == 5, "The input Tensor should have shape=[None, input_height, input_width," \
                                      " input_num_capsule, input_num_atoms]"
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_capsule = input_shape[3]
        self.input_num_atoms = input_shape[4]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_atoms, self.kernel_size, 1, 1, self.num_capsule * self.num_atoms],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.b = self.add_weight(shape=[self.num_capsule, self.num_atoms, 1, 1],
                                 initializer=initializers.constant(0.1),
                                 name='b')

        self.built = True

    def call(self, input_tensor, training=None):

        input_transposed = tf.transpose(input_tensor, [0, 3, 4, 1, 2])
        input_shape = K.shape(input_transposed)
        print("###########################################################", input_transposed.get_shape)
        input_tensor_reshaped = K.reshape(input_tensor, [input_shape[0], 1, self.input_num_capsule * self.input_num_atoms, self.input_height, self.input_width])
        print("###########################################################", input_tensor_reshaped.get_shape)

        input_tensor_reshaped.set_shape((None, 1, self.input_num_capsule * self.input_num_atoms, self.input_height, self.input_width))

        conv = K.conv3d(input_tensor_reshaped, self.W, strides=(self.input_num_atoms, self.strides, self.strides), padding=self.padding, data_format='channels_first')
        conv  = Lambda(lambda x : tf.nn.sigmoid(x) )(conv)       
        
        print("*******%%%%%%%%%%5", conv.get_shape())
        votes_shape = K.shape(conv)
        _, _, _, conv_height, conv_width = conv.get_shape()
        conv = tf.transpose(conv, [0, 2, 1, 3, 4])
        votes = K.reshape(conv, [input_shape[0], self.input_num_capsule, self.num_capsule, self.num_atoms, votes_shape[3], votes_shape[4]])
        print("*******%%%%%%%%%%5", votes.get_shape())
        votes.set_shape((None, self.input_num_capsule, self.num_capsule, self.num_atoms, conv_height.value, conv_width.value))
        print("*******%%%%%%%%%%5", votes.get_shape())

        logit_shape = K.stack([input_shape[0], self.input_num_capsule, self.num_capsule, votes_shape[3], votes_shape[4]])
        biases_replicated = K.tile(self.b, [1, 1, conv_height.value, conv_width.value])

        activations = update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings)

        a2 = tf.transpose(activations, [0, 3, 4, 1, 2])
        return a2

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-2]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_output_length(space[i], self.kernel_size, padding=self.padding, stride=self.strides, dilation=1)
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.num_capsule, self.num_atoms)

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'num_capsule': self.num_capsule,
            'num_atoms': self.num_atoms,
            'strides': self.strides,
            'padding': self.padding,
            'routings': self.routings,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(ConvCapsuleLayer3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def _squash(input_tensor):
    norm = tf.norm(input_tensor, axis=-1, keep_dims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

class Concatanate_mid(layers.Layer):
    def __init__(self, **kwargs):
        super(Concatanate_mid, self).__init__(**kwargs)
#         self.b_initializer = initializers.get(constant_initializer)
#         self.a_initializer = initializers.get(constant_initializer)

    def build(self, input_shape):
        # Transform matrix
        self.A = self.add_weight(shape=[1],
                                 initializer=initializers.constant(1),
                                 name='A')
#        self.B = self.add_weight(shape=[1],
#                                 initializer=initializers.constant(1),
#                                 name='B')
        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]

        inputs1 = inputs[0]
        inputs2 = inputs[1]
            
        alpha = self.A
#        beta = self.B
        print(alpha)
        output_cat =layers.Concatenate(axis=-2)([(1-alpha)*inputs1, alpha*inputs2])
                      
        return output_cat
    
    def compute_output_shape(self, input_shape):
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        output_shape[1]=output_shape[1]+list(input_shapes[1])[1]
        output_shape[0] = None
        return tuple(output_shape)


def _squash_d3(input_tensor):
    in2 = tf.transpose(input_tensor, [0,1,3,4,2])
    norm = tf.norm(in2, axis=-1, keep_dims=True)
    norm_squared = norm * norm
    x = (in2 / norm) * (norm_squared / (1 + norm_squared))
    p = tf.transpose(x, [0,1,4,2,3])
    return p


def update_routing(votes, biases, logit_shape, num_dims, input_dim, output_dim,
                   num_routing):
    if num_dims == 6:
        votes_t_shape = [3, 0, 1, 2, 4, 5]
        r_t_shape =     [1, 2, 3, 0, 4, 5]
    elif num_dims == 4:
        votes_t_shape = [3, 0, 1, 2]
        r_t_shape = [1, 2, 3, 0]
    else:
        raise NotImplementedError('Not implemented')

    votes_trans = tf.transpose(votes, votes_t_shape)
    _, _, _, height, width, caps = votes_trans.get_shape()

    def _body(i, logits, activations):
        """Routing while loop."""
        # route: [batch, input_dim, output_dim, ...]
        route = tf.nn.softmax(logits, dim=2)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1) + biases
        activation = _squash_d3(preactivate)
        activations = activations.write(i, activation)

        act_3d = K.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)

    i = tf.constant(0, dtype=tf.int32)
    _, logits, activations = tf.while_loop(
        lambda i, logits, activations: i < num_routing,
        _body,
        loop_vars=[i, logits, activations],
        swap_memory=True)
    a = K.cast(activations.read(num_routing - 1), dtype='float32')
    print("###########################################################", a.get_shape)
    return K.cast(activations.read(num_routing - 1), dtype='float32')




def CapsNet(input_shape, n_class, routings,inst_parameter):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=(360,1))
    
    
    l = layers.Conv1D(filters=64, kernel_size=7, strides=1, padding='same', activation='relu', name='conv1')(x)
    l = layers.normalization.BatchNormalization()(l)
    l = layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name='conv2')(l)
    l2= layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu', name='convl1')(l)
    l2 = layers.normalization.BatchNormalization()(l2)
    l2 = layers.Reshape((36,10,32))(l2)
    l2 = Lambda(lambda x: K.expand_dims(x,2))(l2)
#    l2 = Lambda(lambda x: tf.transpose(x,[0,3,2,1]))(l2)
    l2 = sqash_caps()(l2)
    l2 = ConvCapsuleLayer3(kernel_size=3, num_capsule=10, num_atoms=8, strides=1, padding='same', routings=3)(l2)
    l2 = layers.normalization.BatchNormalization()(l2)
    
    
    l1 = Lambda(lambda x: K.expand_dims(x,2))(l)
    l1 = layers.Reshape((360,1,8,8))(l1)
    l1 = sqash_caps()(l1)
    l1 = ConvCapsuleLayer3(kernel_size=3, num_capsule=8, num_atoms=8, strides=1, padding='same', routings=3)(l1)
    l1 = layers.normalization.BatchNormalization()(l1)
    
    
    la = FlattenCaps()(l2)
    lb = FlattenCaps()(l1)
    
#     lb = FlattenCaps()(l_skip)
    
    l = Concatanate_mid()([la, lb])
    print(l.get_shape())
#    layers.Concatenate(axis=-2)([la, lb])
    
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=inst_parameter, routings=routings,
                             name='digitcaps')(l)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction
    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    #decoder.add(layers.Dense(56, activation='relu', input_dim=16*n_class))
    #decoder.add(layers.GRU(512,dropout=0.1,recurrent_dropout=0.1))
    #decoder.add(layers.Conv2DTranspose(filters=16*n_class,kernel_size=(10,1),
     #                                        data_format="channels_last"))
    decoder.add(layers.Dense(256, activation='tanh',input_shape=(inst_parameter*n_class,)))
    decoder.add(layers.Dense(45, activation='tanh'))
    decoder.add(layers.Reshape((1,45,1)))
        
    decoder.add(layers.Deconvolution2D(32, 1, 3,subsample=(1, 1),border_mode='same'))
    decoder.add(layers.normalization.BatchNormalization())
    decoder.add(layers.Deconvolution2D(16, 1, 3,subsample=(1, 2),border_mode='same'))
    decoder.add(layers.normalization.BatchNormalization())
    decoder.add(layers.Deconvolution2D(8, 1, 5,subsample=(1, 2),border_mode='same'))
    decoder.add(layers.normalization.BatchNormalization())
    decoder.add(layers.Deconvolution2D(4, 1, 7,subsample=(1, 2),border_mode='same'))
    decoder.add(layers.normalization.BatchNormalization())
    decoder.add(layers.Deconvolution2D(1, 1, 7,subsample=(1, 1),border_mode='same'))
    decoder.add(layers.normalization.BatchNormalization())
#    decoder.add(Activation("tanh"))
    decoder.add(layers.Reshape((360,1)))
#     decoder.add(layers.GRU(56,dropout=0.1,recurrent_dropout=0.1))
#     decoder.add(Flatten())
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
#     decoder.add(layers.Reshape(target_shape=(360,1), name='out_recon'))
    
    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    print("SSSSSSSSSSSSSSSSS",decoder.layers[-1].output_shape)

    # manipulate model
    noise = layers.Input(shape=(n_class, inst_parameter))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model



def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
#     tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
#                                batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                            save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})
    # from sklearn.utils.class_weight import compute_class_weight
    # class_weights=compute_class_weight('balanced',np.unique(np.argmax(y_train,axis=1)),np.argmax(y_train,axis=1))
    # Training without data augmentation:
    model.fit([x_train[:,0:-1,:], y_train], [y_train, x_train[:,0:-1,:]], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test[:,0:-1,:], y_test], [y_test, x_test[:,0:-1,:]]],callbacks=[log,checkpoint, lr_decay],shuffle=True)
            #   ,class_weight=dict(enumerate(class_weights)))
    

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
#     def train_generator(x, y, batch_size, shift_fraction=0.):
#         train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
#                                            height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
#         generator = train_datagen.flow(x, y, batch_size=batch_size)
#         while 1:
#             x_batch, y_batch = generator.next()
#             print(x_batch.shape, y_batch.shape)
#             yield ([x_batch, y_batch], [y_batch, x_batch])

#     # Training with data augmentation. If shift_fraction=0., also no augmentation.
#     model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
#                         steps_per_epoch=int(y_train.shape[0] / args.batch_size),
#                         epochs=args.epochs,
#                         validation_data=[[x_test, y_test], [y_test, x_test]])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

#     from utils import plot_log
#     plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test[:,0:-1,:], batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    print('#####################recon shape')
    #mage = combine_images_1d(x_test[:50],x_recon[:50],args.save_dir)
    num = 500
    plt.figure(figsize=(8,500))
    for i in range(1, num+1):
        plt.subplot(num,2, i)
        if (i%2==1):
            plt.plot(x_test[i//2,:,:])
        else:
            plt.plot(x_recon[i//2,:,:])
    plt.savefig(args.save_dir + "/real_and_recon.png")
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(np.argmax(y_pred,axis=1),np.argmax(y_test,axis=1))
    print(cm)
    
#     plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
#     plt.show()


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    x_train=[]
    x_test=[]
    ind=0
    train_label=[]
    test_label=[]
    folder=os.listdir('./train_96_low_classes')
    print(folder)
    for fl in folder:
        file_name=os.listdir('./train_96_low_classes'+'/'+fl)
        for i in file_name:
            try:
                train_image = np.load('./train_96_low_classes'+'/'+fl+'/'+i,allow_pickle=True)
            except:
                pass
            temp = train_image-np.mean(train_image)
            x_train.append(temp/np.max(np.abs(temp)))
        len_train=len(file_name)     
        train_label.extend([ind]*len_train)
        ind=ind+1
    x_train=np.array(x_train)
    folder=os.listdir('./test_96_low_classes')
    ind=0
    for fl in folder:
        file_name=os.listdir('./test_96_low_classes'+'/'+fl)
        for i in file_name:
            try:
                test_image = np.load('./test_96_low_classes'+'/'+fl+'/'+i,allow_pickle=True)
            except:
                pass
            
            temp = test_image-np.mean(test_image)
            x_test.append(temp/np.max(np.abs(temp)))
        len_test=len(file_name)     
        test_label.extend([ind]*len_test)
        ind=ind+1                     
    x_test=np.array(x_test)
    x_train = x_train.reshape(-1, 361, 1).astype('float32')
    x_test = x_test.reshape(-1, 361, 1).astype('float32')
    y_train = to_categorical(np.array(train_label).astype('float32'))
    y_test = to_categorical(np.array(test_label).astype('float32'))
    print(x_train.shape)
    print(x_test.shape)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--inst_parameter', default=8, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./after_notations_more_classes_changed_decoder_alpha_8')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) 

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(x_train.shape, y_train.shape)
    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=(360,1),
                                                  n_class=y_train.shape[1],
                                                  routings=args.routings,
                                                 inst_parameter=args.inst_parameter)
    model.summary()

    # train or test
#     if args.weights is not None:  # init the model weights with provided one
#         model.load_weights(args.weights)
#     if not args.testing:
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
#     else:  # as long as weights are given, will run testing
#         if args.weights is None:
#             print('No weights are provided. Will test using random initialized weights.')
#         manipulate_latent(manipulate_model, (x_test, y_test), args)
#    maximum_weights=os.listdir(args.save_dir)
#    model.load_weights(args.save_dir+'\\'+maximum_weights[-3])
    
    test(model=eval_model, data=(x_test, y_test), args=args)
