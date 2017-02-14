import lcod
import tensorflow as tf
import shutil

def tf_sparse_count(tf_vec):
    non_zeros = tf.not_equal(tf_vec, 0)
    as_ints = tf.cast(non_zeros, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def train(sess, model, train_gen, num_optimization_steps, valid_gen=None, valid_steps=0, logdir='./logdir'):
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

    if valid_gen is None and not valid_steps == 0:
        raise ValueError('no valid generator was given bad valid steps is not zero.')
    
    if os.path.exists(logdir + '/train'):
        shutil.rmtree(logdir)
    if os.path.exists(logdir + '/validation'):
        shutil.rmtree(logdir)
    #
    # optimize graph with gradient decent with LR of 1/t
    global_step   = tf.Variable(0, trainable=False)
    learning_rate = 0.1
    k             = 0.5
    decay_rate    = 1
    learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, k, decay_rate)
    
    optimizer     = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss, global_step=global_step)
    #
    # for tensor-board logging
    with tf.name_scope('summary'):
        tf.scalar_summary('loss', model.loss) 
        sparsity  = tf_sparse_count(model.output)
        tf.scalar_summary('sparsity', sparsity)

    summary_op = tf.merge_all_summaries()
    train_summary_writer = tf.train.SummaryWriter(logdir=logdir + '/train', graph=sess.graph)
    valid_summary_writer = tf.train.SummaryWriter(logdir=logdir + '/validation')

    print(30*'='+'Restart' + 30*'=')

    tf.global_variables_initializer().run(session=sess)
    print('Initialized')

    for step in range(num_optimization_steps):
        X_train, Z_train = next(train_gen)
        _, loss, summary = sess.run( [optimizer, model.loss, summary_op], 
                                    {model.input: X_train, model.target: Z_train})
        train_summary_writer.add_summary(summary, global_step=step)
        print('\rTrain step: %d. Loss %.6f.' % (step+1, loss))

        if (step % 50 == 0) and valid_steps != 0:
            val_avg_loss = 0
            for val_step in  range(valid_steps):
                X_val, Z_val = next(valid_gen)
                loss, summary = sess.run( [model.loss, summary_op], 
                                    {model.input: X_val, model.target: Z_val})
                val_avg_loss += loss
                valid_summary_writer.add_summary(summary, global_step=step)
            val_avg_loss /= valid_steps
            print('Valid Loss Avg loss: %.6f.'%val_avg_loss)


import argparse 
import approx_sparselearnig_dataset as db_set
import os
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train approximate sparse code learnign network based on paper \
    Learning Fast Approximations of Sparse Coding - http://yann.lecun.com/exdb/publis/pdf/gregor-icml-10.pdf')

    parser.add_argument('-m', '--model', default='lcod', type=str, choices=['lcod', 'lista'], \
        help='input mode to use valid options are - "lcod" or "lista"')

    parser.add_argument('-t', '--train_path', default='/../../lcod_trainset/trainset.npy', type=str,\
        help='output directory for log files can be used with tensor board')

    parser.add_argument('-r', '--ratio', default=0.2, type=float,\
        help='amount of train set for validation check')

    parser.add_argument('-os', '--output_size', default=100, help='Size of sparse representation Z')

    parser.add_argument('-is', '--input_size', default=100, help='Size of dense vector X')

    parser.add_argument('-u', '--unroll_count', default=7, type=int,\
        help='Amount of times to run lcod/list block')

    parser.add_argument('-s', '--num_steps', default=1000, type=int,\
        help='number of training steps')

    parser.add_argument('-vs', '--num_validsteps', default=5, type=int,\
        help='Number of validation step to run every X training steps')

    parser.add_argument('-l', '--log_dir_path', default='./logdir', type=str,\
        help='output directory for log files can be used with tensor board')

    parser.add_argument('-o', '--output_dir_path', default='', type=str,\
        help='output directory to save model if non is given model wont be saved')

    args = parser.parse_args()

    We_shape = (args.input_size, args.output_size)

    if args.model == 'lcod':
        import lcod
        model = lcod.Lcod(We_shape=We_shape, unroll_count=args.unroll_count)
    else:
        raise NotImplementedError('lista is not implemented yet...')

    model.build_model(args.unroll_count)

    data_gens = db_set.trainset_gen(os.path.dirname(os.path.realpath(__file__)) + args.train_path, args.ratio)
    sess = tf.Session()
    train(sess=sess, model=model, \
          train_gen=data_gens.train_gen, \
          num_optimization_steps=args.num_steps, \
          valid_gen=data_gens.valid_gen, \
          valid_steps=args.num_validsteps,\
          logdir=args.log_dir_path)


