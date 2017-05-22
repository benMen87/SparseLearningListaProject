import argparse
import tensorflow as tf
from keras.datasets import mnist
import lista_convdict2d as sparse_encoder

parser = argparse.ArgumentParser(description='Sparse encoder decoder model')


parser.add_argument('-b', '--batch_size', default=50,
                    type=int, help='size of train batches')
parser.add_argument('-n', '--num_steps', default=1, type=int,
                    help='number of training steps')
parser.add_argument('-ks', '--kernal_size', default=3, type=int,
                    help='kernal size to be used in lista_conv')
parser.add_argument('-kc', '--kernel_count', default=6, type=int,
                    help='amount of kernal to use in lista_conv')
parser.add_argument('-u', '--unroll_count', default=7,
                    type=int,
                    help='Amount of times to run lcod/list block')
parser.add_argument('-o', '--output_dir_path',
                    default='',
                    type=str,
                    help='output directory to save model if non is given\
                          model wont be saved')

#
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("training size: {}, test size: {}".format(y_train.shape[0],
      y_test.shape[0]))

# flatten for encoder
X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

# normilize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one hot
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# split train set
(X_valid, y_valid) = (X_train[:5000], y_train[:5000])
(X_train, y_train) = (X_train[5000:], y_train[5000:])

#
# build encoder-decoder
in_encoder_shape = 28 * 28
out_encoder_shape = args.kc * in_encoder_shape
We_shape = (in_encoder_shape, out_encoder_shape)

with tf.variable_scope('encoder'):
    encoder = sparse_encoder.LISTAConvDict2d(We_shape=We_shape,
                                             unroll_count=args.unroll_count,
                                             L=8, batch_size=args.batch_size,
                                             kernel_size=args.kernel_size,
                                             kernel_count=args.kernel_count)
encoder.build_model()
with tf.variable_scope('decoder'):
    init_de = tf.random_normal([args.kernel_size, args.kernel_size,
                                args.kernel_count, 1])
    D = tf.Variable(init_de)
    Xhat = tf.nn.conv2d(encoder.output2D, D, strides=[1, 1, 1, 1], padding='SAME')

loss = tf.reduce_mean(tf.square(encoder.input - Xhat)) + \
       0.5 / 8 * tf.reduce_sum(tf.abs(encoder.output))

encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder/")
decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder/")
optimizer_en = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=encoder_vars)
optimizer_de = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=decoder_vars)


def nextbatch(X, Y, batch_size):
    offset = 0
    data_len = X.shape[0]
    while True:
        if offset + batch_size < data_len:
            batch_X = X[offset: batch_size + offset]
            batch_Y = Y[offset: batch_size + offset]
            offset = offset + batch_size
        else:
            batch_X = X[offset: data_len] + X[:batch_size - (data_len - offset)]
            batch_Y = Y[offset: data_len] + X[:batch_size - (data_len - offset)]
            offset = batch_size - (data_len - offset)
        yield X, Y

#
# build model
train_batch = nextbatch(X_train, Y_train, args.batch_size)
vaild_batch = nextbatch(X_valid, y_valid, 500)
test_batch = nextbatch(X_test, y_test, 500)

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    print('Initialized')

    for iter in range(args.num_steps):

        X_batch, _ = train_batch.next()

        if iter % 10 == 0:
            _, loss = sess.run([optimizer_de, loss], {encoder.input: X_batch})
        else:
            _, loss = sess.run([optimizer_en, loss], {encoder.input: X_batch})
        

