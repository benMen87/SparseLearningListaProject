import argparse

def args(train_mode=False):
    parser = argparse.ArgumentParser(description='Sparse encoder decoder model')
    parser.add_argument('-ks', '--kernel_size', default=3, type=int,
                                help='kernel size to be used in lista_conv')
    parser.add_argument('-kc', '--kernel_count', default=1, type=int,
                                help='amount of kernel to use in lista_conv')
    parser.add_argument('--dilate', '-dl', action='store_true')
    parser.add_argument('-u', '--unroll_count', default=1,
         type=int, help='Amount of Reccurent timesteps for decoder')
    parser.add_argument('--shrinkge_type', default='soft thresh',
                            choices=['soft thresh', 'smooth soft thresh'])
    parser.add_argument('--task',  default='doc_clean', choices=['denoise',
                                                              'denoise_dynamicthrsh',
                                                              'inpaint',
                                                              'doc_clean',
                                                              'deblur'], 
            help='task to train lista on')
    parser.add_argument('--grayscale',  action='store_true', help='converte RGB images to grayscale')
    parser.add_argument('--inpaint_keep_prob', '-p', type=float, default=0.5,
            help='probilty to sample pixel')
    parser.add_argument('--noise_sigma', '-ns', type=float, default=20,
            help='noise magnitude')
    parser.add_argument('--sae_type', '-st', default='classic_sae',
            choices=['classical_sae', 'resudual_sae', 'ms_sae'], help='diffrent ways to connect encoder_decoder')
    parser.add_argument('--model_type', '-mt', default='convdict', choices=['convdict',
                                                                           'convdict_alt', 
                                                                            'convmultidict',
                                                                            'untied',
                                                                            'dynamicthrsh',
                                                                            'dynamicthrsh_untied',
                                                                            'adaptive_deblur',
                                                                            'adaptive_deblur_untied'])

    parser.add_argument('--norm_kernal',  action='store_true', help='keep kernals with unit kernels')
    parser.add_argument('--psf_id',  default=1, type=int, help='psf to use -1 for random comb (-1 is only for training)')
    parser.add_argument('--pyramid_depth',  default=1, type=int, help='depth\
                        of resudual pyramid')
    if train_mode:
        parser.add_argument('--name', default='lista_conv2d', type=str, help='used for\
            creating load/store log dir names')
        parser.add_argument('-b', '--batch_size', default=5,
                                    type=int, help='size of train batches')
        parser.add_argument('-n', '--num_epochs', default=1, type=int,
                                    help='number of epochs steps')
        parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='learning rate')
        parser.add_argument('--save_model', dest='save_model', action='store_true')
        parser.add_argument('--load_model', dest='load_model', action='store_true')
        parser.add_argument('--debug', dest='debug', action='store_true')
        parser.add_argument('--load_name', default='', type=str, help='used to\
            load from a model with "name" diffrent from this model name')
        parser.add_argument('--dataset', default='docs', choices=['mnist',
            'docs', 'stl10', 'cifar10', 'pascal', 'pascal_small', 'berkeley'])
        parser.add_argument('--sparse_factor', '-sf',  default=0.0, type=float)
        parser.add_argument('--sparse_sim_factor',  default=0, type=float)
        parser.add_argument('--recon_factor', '-rf',  default=0.2, type=float)
        parser.add_argument('--ms_ssim', '-ms',  default=0.8, type=float)
        parser.add_argument('--dup_count', '-dc',  default=1, type=int)
        parser.add_argument('--load_pretrained_dict', action='store_true', help='inilize dict with pre traindict in "./pretrained_dict" dir')
        parser.add_argument('--dont_train_dict', action='store_true',  help='how many epochs to wait train dict 0 means dont train')
        parser.add_argument('--disttyp', '-dt', default='l1', type=str, choices=['l2', 'l1', 'smoothl1'])
    else:
        parser.add_argument('--checkpoint_path', help='path to saved checkpoint file with trained weights')
        parser.add_argument('--image_path', help='path to images')
        parser.add_argument('--patt', default='png', help='add image name or pattern of images to run modle on')
        parser.add_argument('--savepath', '-s', default=' ', help='path to save results')

    args = parser.parse_args()

    return args
 

