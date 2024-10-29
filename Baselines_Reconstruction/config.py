# ------------------------------------------------------------------
"""
Main config file

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import argparse
import pickle
import os
import datetime

# ------------------------------------------------------------------


def add_all_arguments(parser):

    # --- general options ---
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_workers', type=int, default=8, help='number of workers for multiprocessing')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='allocate the loaded samples in GPU memory. Use it with training on GPU')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--name', type=str, default='test', help='name of the experiment')
    parser.add_argument('--dir_log', type=str, default=r'./log', help='log folder')

    parser.add_argument('--root_synthetic', type=str, default=r'../Synthetic/synthetic_CERRA',
                        help='root of the synthetic dataset')

    parser.add_argument('--gpu_id', type=str, default="0, 1, 2, 3", help='gpu ids: i.e. 0  (0,1,2, use -1 for CPU)')
    parser.add_argument('--nan_fill', type=float, default=0., help='a value to fill missing values')

    parser.add_argument('--in_channels_dynamic', type=int, default=6, help='number of input dynamic variables')

    # --- STEAL NET ---
    parser.add_argument('--en_embed_dim_steal', type=int, default=[96, 128, 256], help='hidden dimensions in STEALNET encoder')
    parser.add_argument('--de_embed_dim_steal', type=int, default=[256, 128, 96], help='hidden dimensions in STEALNET decoder')

    # --- UniAD ---
    parser.add_argument('--inplanes', type=int, default=6, help='in planes in UniAD')
    parser.add_argument('--instrides', type=int, default=2, help='strides in UniAD')
    parser.add_argument('--feature_size', type=tuple, default=(100, 100), help='size of feature maps')
    parser.add_argument('--feature_jitter_scale', type=int, default=0.01, help='scale for feature jitter')
    parser.add_argument('--feature_jitter_prob', type=float, default=.0, help='probability to features jitter')
    parser.add_argument('--neighbor_size', type=tuple, default=(9, 9), help='neighbourhood size or UniAD')
    parser.add_argument('--neighbor_mask', type=list, default=[True, True, True],
                        help='whether use mask in [enc, dec1, dec2]')
    parser.add_argument('--hidden_dim', type=int, default=96, help='hidden dimensions in the model')
    parser.add_argument('--pos_embed_type', type=str, default='learned', help='type for positional embedding')
    parser.add_argument('--initializer', type=str, default='xavier_uniform', help='type of initializer')
    # parser.add_argument('--save_recon', type=str, default='False')
    parser.add_argument('--nhead', type=int, default=3, help='number of heads in UniAD')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='number of layer in the UniAD encoder')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='number of layer in the UniAD decoder')
    parser.add_argument('--dim_feedforward', type=int, default=96 * 4, help='dimension of the feedforward mlp in the UniAD')
    parser.add_argument('--dropout', type=float, default=0.1, help='drop out ratio for UniAD')
    parser.add_argument('--activation', type=str, default='relu', help='activation function for UniAD')
    parser.add_argument('--normalize_before', type=bool, default=False, help='')
    parser.add_argument('--return_intermediate_dec', type=bool, default=False, help='')

    parser.add_argument('--en_de_pretrained', type=str,
                        default=None,
                        help='pretrained model i.e. a trained model with best loss')

    parser.add_argument('--times_train', type=tuple, default=(1, 52 * 34), help='time steps for training on synthetic data')
    parser.add_argument('--times_val', type=tuple, default=(52 * 34 + 1, 52 * 40), help='time steps for validation on synthetic data')
    parser.add_argument('--times_test', type=tuple, default=(52 * 40 + 1, 52 * 46), help='time steps for testing on synthetic data' )

    parser.add_argument('--delta_t', type=int, default=1, help='number of weeks or time steps')
    parser.add_argument('--window_size', type=int, default=1, help='scaling factor for resolution (2 means half)')

    parser.add_argument('--x_min', type=int, default=0, help='start of grid extension in x direction')
    parser.add_argument('--x_max', type=int, default=200, help='end of grid extension in x direction')
    parser.add_argument('--y_min', type=int, default=0, help='start of grid extension in y direction')
    parser.add_argument('--y_max', type=int, default=200, help='start of grid extension in y direction')

    parser.add_argument('--is_shuffle', type=bool, default=False, help='if True, apply data shuffling')
    parser.add_argument('--is_aug', type=bool, default=True, help='if True, apply data augmentation')
    parser.add_argument('--is_norm', type=bool, default=True, help='if True, apply data normalization')
    parser.add_argument('--is_clima_scale', type=bool, default=True,
                        help='if True, apply data normalization with climatology')
    parser.add_argument('--is_replace_anomaly', type=bool, default=True,
                        help='if True, replace anomalies with normal data from climatology')

    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.003, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 momentum term for Adam/AdamW')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 momentum term for Adam/AdamW')

    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler')
    parser.add_argument('--lr_warmup', type=int, default=1e-6, help='learning rate for warmup')
    parser.add_argument('--lr_warmup_epochs', type=int, default=2, help='number of epochs for warmup')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='minimum learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate step decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='learning rate decay')

    # input variables
    parser.add_argument('--variables', type=str,
                        default=[
                            'var_01',
                            'var_02',
                            'var_03',
                            'var_04',
                            'var_05',
                            'var_06'
                        ]
                        , help='input dynamic variables')

    parser.add_argument('--variables_static', type=str,
                        default=[
                                 'latitude',
                                 'longitude'],

                        help='input static variables')
    return parser


def read_arguments(train=True, print=True, save=True):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser)
    parser.add_argument('--phase', type=str, default='train')
    config = parser.parse_args()
    config.phase = 'train' if train else 'test'
    if print:
        print_options(config, parser)
    if save:
        save_options(config, parser)

    return config


def save_options(config, parser):

    if config.name is None or len(config.name) == 0:
        config.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    dir_log = os.path.join(config.dir_log, config.name)
    os.makedirs(dir_log, exist_ok=True)

    with open(dir_log + '/config.txt', 'wt') as config_file:
        message = ''
        message += '----------------- Options ---------------       -------------------\n\n'
        for k, v in sorted(vars(config).items()):
            if k in ['variables', 'times_train', 'times_val', 'times_test', 'dir_log']:
                continue
            # comment = ''
            default = parser.get_default(k)
            # if v != default:
            comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<20}{}\n'.format(str(k), str(v), comment)

        comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
        message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

        message += '\n----------------- Input Variables -------      -------------------'
        message += '\n\n{}\n'.format(str(config.variables))

        message += '\n----------------- Times -----------------      -------------------'
        if config.phase == 'train':
            message += '\n\nTraining: {}'.format(str(config.times_train))
            message += '\nValidation: {}\n'.format(str(config.times_val))
        else:
            message += '\n\nTesting: {}\n'.format(str(config.times_test))

        message += '\n----------------- End -------------------      -------------------'
        config_file.write(message)

    with open(dir_log + '/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)


def print_options(config, parser):
    message = ''
    message += '----------------- Options ---------------       -------------------\n\n'
    for k, v in sorted(vars(config).items()):
        if k in ['variables', 'times_train', 'times_val', 'times_test', 'dir_log']:
            continue
        # comment = ''
        default = parser.get_default(k)
        # if v != default:
        comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<20}{}\n'.format(str(k), str(v), comment)

    comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
    message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

    message += '\n----------------- Input Variables -------      -------------------'
    message += '\n\n{}\n'.format(str(config.variables))

    message += '\n----------------- Times -----------------      -------------------'
    if config.phase == 'train':
        message += '\n\nTraining: {}'.format(str(config.times_train))
        message += '\nValidation: {}\n'.format(str(config.times_val))
    else:
        message += '\n\nTesting: {}\n'.format(str(config.times_test))

    message += '\n----------------- End -------------------      -------------------'
    print(message)


if __name__ == '__main__':

    config = read_arguments(train=True, print=True, save=False)


