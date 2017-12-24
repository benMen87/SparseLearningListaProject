import tensorflow as tf
from Utils import ms_ssim

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



def reconstruction_loss(_model, disttyp, ms_ssim):
    dist_loss = 0
    _ms_ssim = 0

    out = _model.decoder.output
    target = _model.decoder.target
    boundry = (_model.encoder.kernel_size // 2,_model.encoder.kernel_size // 2)

    if disttyp == 'l1':
        print('out: {} target: {}'.format(out, target))
        dist_loss = l1(_out=out, _target=target, _boundry=boundry)
        tf.summary.scalar('dist_l1', dist_loss, collections=['TB_LOSS'])
    elif disttyp == 'smooth_l1':
        dist_loss = smooth_l1(_out=_out, _target=_target, _boundry=boundry)
        tf.summary.scalar('dist_smoothl1', dist_loss, collections=['TB_LOSS'])
    elif disttyp == 'l2':
        dist_loss = l2(_out=out, _target=target, _boundry=boundry)
        tf.summary.scalar('dist_l2', dist_loss, collections=['TB_LOSS'])

    if ms_ssim:
        half_ker = _model.encoder.kernel_size // 2
        _ms_ssim = ms_ssim.tf_ms_ssim(
            target[:,half_ker:-half_ker,half_ker:-half_ker,...],
            out[:,half_ker:-half_ker,half_ker:-half_ker,...],
            level=4
        )
        tf.summary.scalar('ms_ssim', _ms_ssim, collections=['TB_LOSS'])
    return dist_loss, (1 - _ms_ssim)

def sparsecode_loss(_model, sparse_sim_factor=False):
    sparse_loss = 0
    similarity_loss = 0

    sparse_loss = tf.reduce_mean(tf.abs(_model.sparsecode))
    tf.summary.scalar('l1_sparse', sparse_loss, collections=['TB_LOSS'])
    if sparse_sim_factor and False:
        raise NotImplementedError('Missing argument to funtion')
        similarity_loss = approx_sae_losses.sc_similarity(
            _x=_model.sparsecode,
            _chunk_size=5, # HYPR_PARAMS['dup_count']
            _chunk_count=5 # HYPR_PARAMS['batch_size']
        )
        tf.summary.scalar('sc_sim', similarity_loss, collections=['TB_LOSS'])
    return sparse_loss, similarity_loss
 