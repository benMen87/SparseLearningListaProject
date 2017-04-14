import os
import sys
import lcod
import lista
import lista_conv
import lista_convdict
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt
import numpy as np
from sparse_coding import cod, ista
# from dict_learning.traindict import display_atoms

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def tf_sparse_count(tf_vec):
    non_zeros = tf.not_equal(tf_vec, 0)
    as_ints = tf.cast(non_zeros, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count


def zero_none_grad(grad, var):
    return grad if grad is not None else tf.zeros_like(var)


def cape_g_byname(grad, var, name, min=-1, max=1):
    grad = zero_none_grad(grad, var)
    if var.name == model.theta.name:
        grad = tf.clip_by_value(zero_none_grad(grad, var), min, max)
    return grad


def train(sess, model, train_gen, num_optimization_steps, valid_gen=None,
          valid_steps=0, logdir=DIR_PATH + '/logdir'):
    """ Train.

    Args:
        sess: A Session.
        model: A Model.
        train_gen: A generator (or iterator) that yields `(inputs, targets)` tuples, with
            `inputs` and `targets` both having shape `[dynamic_duration, 1]` from Train set.
        num_optimization_steps: An integer.
        valid_gen: A generator (or iterator) that yields `(inputs, targets)` tuples, with
            `inputs` and `targets` both having shape `[dynamic_duration, 1]` from Validation set.
        valid_step: An integer how many seteps per valid run.
        logdir: A string. The log directory default "./logdir" .
    """

    if valid_gen is None and valid_steps != 0:
        raise ValueError('no valid generator was given bad valid steps is not zero.')
    if os.path.exists(logdir + '/train'):
        shutil.rmtree(logdir)
    if os.path.exists(logdir + '/validation'):
        shutil.rmtree(logdir)
    if not os.path.exists(logdir + '/plots'):
        os.makedirs(logdir + '/plots')
    plotdir = logdir + '/plots'
    if not os.path.exists(plotdir + '/approx_sparse'):
        os.makedirs(plotdir + '/approx_sparse')
    approxplotdir = plotdir + '/approx_sparse'
    #
    # optimize graph with gradient decent with LR of 1/t
    global_step = tf.Variable(0, trainable=False)
    learning_rate = 0.001
    k = 0.5
    decay_rate = 1
    learning_rate = tf.train.inverse_time_decay(learning_rate, global_step,
                                                k, decay_rate)
    #
    # Clip gradients to avoid overflow due to recurrent nature of algorithm
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(model.loss)
    # gvs = optimizer.compute_gradients(model.loss)
    # capped_gvs = [(tf.clip_by_value(zero_none_grad(grad, var), -10, 10), var)
    #               for grad, var in gvs]
    # optimizer = optimizer.apply_gradients(capped_gvs)

    print(30*'='+'Restart' + 30*'=')

    tf.global_variables_initializer().run(session=sess)
    print('Initialized')

    train_loss = []
    valid_loss = []
    Z_sparcity = []
    Z_star_sparcity = []

    for step in range(num_optimization_steps):
        X_train, Z_train = next(train_gen)

        _, loss, Z = sess.run([optimizer, model.loss, model.output],
                              {model.input: X_train,
                              model.target: Z_train})
        train_loss.append(loss)
        Z_sparcity.append(np.count_nonzero(Z) / len(Z))
        Z_star_sparcity.append(np.count_nonzero(Z_train) / len(Z_train))

        # print('\rTrain step: %d. Loss %.6f.' % (step+1, loss))

        if (step) % 100 == 0:
            X_train, Z_train = next(train_gen)
            loss, Z, We = sess.run([model.loss, model.output,
                                    model.We],
                                   {model.input: X_train,
                                    model.target: Z_train})

            # print('\rSanity run: Loss %.6f current avg sparsity %.6f.' %
            #      (loss, np.mean(Z_sparcity)))
            # print('S norm: %.6f Wd norm: %.6f thea: %.6f'%(np.linalg.norm(S,'fro'),
            #                                                np.linalg.norm(We,'fro'),
            #                                                np.linalg.norm(theta)))

        if (step % 500 == 0) and valid_steps != 0:
            val_avg_loss = 0
            for _ in range(valid_steps):
                X_val, Z_val = next(valid_gen)

                loss = sess.run(model.loss, {model.input: X_val,
                                             model.target: Z_val})
                val_avg_loss += loss

            val_avg_loss /= valid_steps
            valid_loss.append(val_avg_loss)
            print('Valid Loss Avg loss: %.6f.'%val_avg_loss)
    plt.figure()
    plt.subplot(211)
    plt.plot(train_loss[10:])
    plt.ylabel('loss')
    plt.title('Train loss per iteration {}'.format(model.unroll_count))
    plt.subplot(212)
    plt.plot(valid_loss[1:])
    plt.ylabel('loss')
    plt.title('Validation loss per iteration')
    plt.savefig(approxplotdir + '/Train_loss_per_iteration_{}'.format(model.unroll_count), bbox_inches='tight')

    plt.figure()
    plt.subplot(211)
    plt.plot(Z_star_sparcity)
    plt.title('ZStar sparsity')
    plt.subplot(212)
    plt.plot(Z_sparcity)
    plt.title('Z sparsity iteration {}'.format(model.unroll_count))
    plt.savefig(approxplotdir + '/Sparsity_{}'.format(model.unroll_count), bbox_inches='tight')


def test(sess, model, test_gen, iter_count, Wd, sparse_coder='cod', test_size=20):

    approx_sc_err = 0
    sc_err = 0

    if sparse_coder == 'lcod':
        sparse_coder = cod.CoD(Wd=Wd, max_iter=iter_count)
    elif sparse_coder == 'lista' or sparse_coder == 'lista_cov' or  sparse_coder == 'lista_convdict':
        sparse_coder = ista.ISTA(Wd=Wd, max_iter=iter_count, verbose=False)
    else:
        raise NameError('sparse_coder should be ether "ista" or "cod"')
    input_shape = (1, model.input.get_shape().as_list()[-1])
    output_shape = (1, model.output[0].get_shape().as_list()[-1])
    for i in range(test_size):
        X_test, Z_test = next(test_gen)

        X_test = np.reshape(np.array(X_test), input_shape)
        Z_test = np.reshape(np.array(Z_test), output_shape)
        # #
        # run approx SC
        Z_approx= sess.run(model.output,
                           {model.input: X_test, model.target: Z_test})

        # Z_approx = Z_approx.T
        approx_sc_err += np.sum((Z_approx - Z_test) ** 2)
        #
        # run ista/cod with specified iterations
        Z_sc, _ = sparse_coder.fit(X_test.T)
        sc_err += np.sum((Z_sc.T - Z_test) ** 2)

    approx_sc_err /= test_size
    sc_err /= test_size

    return approx_sc_err, sc_err




import argparse 
sys.path.append(DIR_PATH + '/..')
from Utils import db_tools

if __name__ == '__main__':

    defualt_iters = [1, 2, 5, 10, 20, 50, 100, 200]

    parser = argparse.ArgumentParser(description='Train approximate sparse code learnign network based on paper \
    Learning Fast Approximations of Sparse Coding - \
    http://yann.lecun.com/exdb/publis/pdf/gregor-icml-10.pdf')

    parser.add_argument('-m', '--model', default='lista_convdict', type=str,
                        choices=['lcod', 'lista', 'lista_conv', 'lista_convdict'],
                        help='input mode')

    parser.add_argument('-b', '--batch_size', default=1,
                        type=float, help='size of train batches')

    parser.add_argument('-tr', '--train_path',
                        default='/../../covdict_data/trainset.npz', type=str,
                        help='Path to train data')

    parser.add_argument('-ts', '--test_path',
                        default='/../../covdict_data/testset.npz', type=str,
                        help='Load train data')

    parser.add_argument('-r', '--ratio', default=0.2, type=float,
                        help='amount of train set for validation check')

    parser.add_argument('-os', '--output_size', default=100,
                        help='Size of sparse representation Z')

    parser.add_argument('-nw', '--not_warm_start', action='store_true',
                        help='if specified initlize ver randomly')

    parser.add_argument('-u', '--unroll_count', default=[2],
                        type=int, nargs='+',
                        help='Amount of times to run lcod/list block')

    parser.add_argument('-n', '--num_steps', default=0, type=int,
                        help='number of training steps')

    parser.add_argument('-vs', '--num_validsteps', default=50, type=int,
                        help='Number of validation step to run every\
                        X training steps')

    parser.add_argument('-ks', '--kernal_size', default=3, type=int,
                        help='kernal size to be used in lista_conv')

    parser.add_argument('-kc', '--kernal_count', default=16, type=int,
                        help='amount of kernal to use in lista_conv')

    parser.add_argument('-dp', '--dictionary_path',
                        default='/../../covdict_data/conv_dict.npy')

    parser.add_argument('-l', '--log_dir_path',
                        default='../../lcod_logdir/logdir', type=str,
                        help='output directory for log files can be \
                        used with tensor board')

    parser.add_argument('-o', '--output_dir_path', default='', type=str,
                        help='output directory to save model if non is given\
                              model wont be saved')

    args = parser.parse_args()

    Wd = np.load(DIR_PATH + args.dictionary_path)
    We_shape = Wd.T.shape
    approx_error = []
    sc_error = []

    for unroll_count in args.unroll_count:

        print("*"*30 + 'unroll amount {}'.format(unroll_count) + "*"*30)

        data_gens = db_tools.trainset_gen(DIR_PATH + args.train_path, args.ratio,
                                          args.batch_size)

        if args.not_warm_start is False:
            We = Wd.T
            # We = np.random.normal(size=We_shape)
            # We /= np.linalg.norm(We, axis=1)
        else:
            We = None

        if args.model == 'lcod':
            X, Z = next(data_gens.train_gen)
            model = lcod.LCoD(We_shape=We_shape, unroll_count=unroll_count,
                              We=We, batch_size=args.batch_size)
            tst = 'lcod'
        elif args.model == 'lista':
            model = lista.LISTA(We_shape=We_shape, unroll_count=unroll_count,
                                We=We, batch_size=args.batch_size)
            tst = 'lista'
        elif args.model == 'lista_convdict':
            tst = 'lista_convdict'
            filter_arr = np.load(DIR_PATH + '/../../covdict_data/filter_arr.npy')
            L = max(abs(np.linalg.eigvals(np.matmul(We, We.T))))
            model = lista_convdict.LISTAConvDict(We_shape=We_shape,
                                                 unroll_count=unroll_count,
                                                 filter_arr=filter_arr, L=L,
                                                 batch_size=args.batch_size,
                                                 kernal_size=args.kernal_size,
                                                 amount_of_kernals=args.kernal_count)
        else:
            tst = 'lista_cov'
            model = lista_conv.LISTAConv(We_shape=We_shape,
                                         unroll_count=unroll_count,
                                         We=We, batch_size=args.batch_size,
                                         kernal_size=args.kernal_size,
                                         amount_of_kernals=args.kernal_count)

        with tf.Session() as sess:
            model.build_model()
            train(sess=sess, model=model,
                  train_gen=data_gens.train_gen,
                  num_optimization_steps=args.num_steps,
                  valid_gen=data_gens.valid_gen,
                  valid_steps=args.num_validsteps)

            test_gen = db_tools.testset_gen(DIR_PATH + args.test_path)
            aperr, scerr = test(sess=sess, model=model,
                                iter_count=unroll_count,
                                test_gen=test_gen, Wd=Wd, sparse_coder=tst)

        tf.reset_default_graph()
        approx_error.append(aperr)
        sc_error.append(scerr)

    if args.model == 'lcod':
        lb1 = 'LCoD'
        lb2 = 'CoD'
    elif args.model == 'lista':
        lb1 = 'LISTA'
        lb2 = 'ISTA'
    elif args.model == 'lista_convdict':
        lb1 = 'LISTAConvDict'
        lb2 = 'ISTA'
    else:
        lb1 = 'LISTAConv'
        lb2 = 'ISTA'

    file_name = DIR_PATH + '/logdir/data_result/approx_sc/error_{}'.format(args.model)
    np.savez(file_name, approx=approx_error, sc=sc_error)

    plt.figure()
    plt.plot(args.unroll_count, approx_error, 'ro', label=lb1)
    plt.plot(args.unroll_count, sc_error, 'g^', label=lb2)
    plt.ylabel('error')
    plt.xlabel('iter')
    plt.legend(loc='upper right')
    plt.show()

