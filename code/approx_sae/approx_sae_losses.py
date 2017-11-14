import tensorflow as tf

def _smooth_l1(self, _x, _thrsh=1, _theta=0.5):
        """
        smooth l1 funciton as proposed in:
        Faster R-CNN: https://arxiv.org/pdf/1504.08083.pdf
        """
        less = tf.to_float(tf.less(_x, _thrsh))
        dist = less * (_theta * tf.square(_x)) + (1 - less) * (tf.abs(_x) - _theta)
        return dist

def dist(_x, _y, _distfn, _boundry, _name='recon_loss'):

    bound_row = _boundry[0]
    bound_col = _boundry[1]
    dist = _distfn(
        _x[:,bound_row:-bound_row,bound_row:-bound_row,:] -
        _y[:,bound_row:-bound_row,bound_row:-bound_row,:]
        )
    return tf.reduce_mean(dist, name=_name)

def l1(_out, _target, _boundry=(0,0)):
    return dist(_out, _target, tf.abs, _boundry)

def l2(_out, _target, _boundry=(0,0)):
    return dist(_out, _target, tf.square, _boundry)

def smooth_l1(_out, _target, _boundry=(0,0)):
    return dist(_out, _target, _smooth_l1, _boundry)

def sc_similarity(_x, _chunk_size, _chunk_count):
    """
    _chunk_size - the amount of time an image is repeated.
    smiilarity is messured whithin each chunk. 
    """
    var = tf.concat(
        [tf.nn.moments(chunk, axes=0)[1] for chunk in tf.split(_x, _chunk_count, axis=0)],
        axis=0
        )
    return tf.reduce_mean(var)
