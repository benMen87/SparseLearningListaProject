import os
import tensorflow as tf
    

def tf_sparse_count(tf_vec):
    non_zeros = tf.not_equal(tf_vec, 0)
    as_ints = tf.cast(non_zeros, tf.int32)
    count = tf.reduce_mean(tf.reduce_sum(as_ints, axis=[1,2,3]))
    return count

def lerning_rate_timedecay(learning_rate, decay_rate, global_step, decay_steps):
    learning_rate = tf.train.inverse_time_decay(
            learning_rate=args.learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
            )
    return learning_rate

def clip_grad(optimizer, loss,  val, tr_vars, global_step):
    gvs = optimizer.compute_gradients(loss, tr_vars)
    capped_gvs = [(tf.clip_by_value(grad, -val, val), var) for grad, var in gvs]
    optimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    return optimizer

def get_optimizer(optimizer, lr=0.001, momentum=0.9):
    opt_name = optimizer
    if  opt_name == 'adam':
        return tf.train.AdamOptimizer(lr)
    elif opt_name == 'momentum':
        return  tf.train.MomentumOptimizer(lr, momentum, use_nesterov=True)

def change_lr_val(sess, tf_lr, factor):
    sess.run(tf_lr.assign(tf_lr * factor))

class Saver():
    """Help handle save/restore logic"""
    def __init__(self, **kwargs):
        self._save = kwargs.get('save_model', False)
        self._load = kwargs.get('load_model', False)
        self._name = kwargs['name']
        self._path = kwargs['dir_path'] + '/logdir/models/' + self._name + '/'
        self._dir_path = kwargs['dir_path']
        self._saver = None

        if self._save  or self._load:
            print('#'*15+'\n'+'checkpoint dir %s\n'%self._path + '#'*15)

    def  __call__(self): 
        if self._save or self._load:
            self._saver = tf.train.Saver(max_to_keep=1)
            if not os.path.isdir(self._path):
                os.mkdir(self._path)

    def maybe_load(self, load_name, sess):
        if self._load:
            if load_name != '':
                LOAD_DIR = self._dir_path + '/logdir/models/' + load_name + '/'
            else:
                LOAD_DIR = self._path
            if os.listdir(LOAD_DIR):
                print('loading model')
                self._saver.restore(sess, tf.train.latest_checkpoint(LOAD_DIR))
                return True
            else:
                print('no cpk to load running with random init')
                return False

    def save(self, sess, global_step):
        self._saver.save(sess, self._path, global_step=global_step)

    def restore(self, sess):
        self._saver.restore(sess, tf.train.latest_checkpoint(self._path))

    @property
    def saver(self):
        return self._saver
 
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def config_train_tb(_model, tb_name, loss=None, add_stats=False):
    tensorboard_path = '/data/hillel/tb/' + tb_name + '/'
    if not os.path.isdir(tensorboard_path):
        os.mkdir(tensorboard_path)

    with tf.variable_scope('variables_norm'):
        for var in tf.trainable_variables(): # TODO: see if this is better for debug than bellow
            tf.summary.scalar(var.name, tf.norm(var))
    
    if loss is not None:
        grads = tf.gradients(loss, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))
        with tf.variable_scope('gradiant_norm'):
            for grad, var in grads:
                print(var.name)
                tf.summary.scalar(var.name, tf.norm(grad))


    if add_stats:
        with tf.name_scope('encoder'):
            with tf.name_scope('We'):
                variable_summaries(_model._encoder[-1]._We)
            with tf.name_scope('Wd'):
                variable_summaries(_model._encoder[-1]._Wd)
            with tf.name_scope('threshold'):
                variable_summaries(_model._encoder[-1]._theta)
        with tf.name_scope('decoder'):
            variable_summaries(_model._decoder[-1].convdict)
        with tf.name_scope('sparse_code'):
            variable_summaries(_model._encoder[-1].output)
        tf.summary.scalar('encoded_sparsity',
             tf.reduce_mean(tf.count_nonzero(_model._encoder.output, axis=[1,2,3])))

    if 'dynamicthrsh' in  _model.type:
        tf.summary.image('input', _model.encoder.inputs_noisy)
    elif 'adaptive_deblur' in _model.type:
        tf.summary.image('input', _model.encoder.inputs_blur)
    else:
        tf.summary.image('input', _model.input)

    tf.summary.image('output', _model.output)
    tf.summary.image('target', _model.target)
    return tensorboard_path

