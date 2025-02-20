# ------------------------------------------------------------------
"""
Main config file for Multiple Instance Learning (MIL)

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
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--name', type=str, default='test', help='name of the experiment')
    parser.add_argument('--dir_log', type=str, default=r'./log', help='log folder')

    parser.add_argument('--root_synthetic', type=str, default=r'../Synthetic/synthetic_CERRA',
                        help='root of the synthetic dataset')

    parser.add_argument('--encoder', type=str, default='Swin_3D', help='name of the encoder model')
    parser.add_argument('--agent', type=str, default='Swin_3D', help='name of the encoder model')
    parser.add_argument('--classifier', type=str, default='ARNet', help='name of the classifier model')

    parser.add_argument('--gpu_id', type=str, default="0, 1, 2, 3", help='gpu ids: i.e. 0  (0,1,2, use -1 for CPU)')

    parser.add_argument('--nan_fill', type=float, default=0., help='a value to fill missing values')

    # --- encoder ---
    parser.add_argument('--in_channels_dynamic', type=int, default=6, help='number of input dynamic variables')
    parser.add_argument('--in_channels', type=int, default=1, help='number of input channels per variable')  # 2 for Era5 and CERRA, 1 for synthetic
    parser.add_argument('--en_embed_dim', type=int, default=[16, 16], help='hidden dimensions in the encoder model')
    parser.add_argument('--en_depths', type=int, default=[2, 1], help='number transformer blocks inside each layer')
    parser.add_argument('--en_patch_size', type=int, default=(1, 1, 1),
                        help='patch size inside transformer. Keep it 1 for regression tasks')
    parser.add_argument('--en_window_size', type=int, default=[(2, 4, 4), (8, 1, 1)], help='window size for self-attention/scanning')
    parser.add_argument('--en_mlp_ratio', type=float, default=4., help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--en_drop_rate', type=float, default=0., help='dropout rate')
    parser.add_argument('--en_drop_path_rate', type=float, default=0., help='stochastic depth rate')
    parser.add_argument('--en_patch_norm', type=bool, default=False,
                        help='if True, add normalization after patch embedding')
    parser.add_argument('--en_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')

    # encoder Swin
    parser.add_argument('--en_n_heads', type=int, default=[2, 2], help='number of heads for self-attention')
    parser.add_argument('--en_attn_drop_rate', type=float, default=0.0, help='attention dropout rate')
    parser.add_argument('--en_qkv_bias', type=bool, default=True,
                        help='if True, add a learnable bias to query, key, value')
    parser.add_argument('--en_qk_scale', type=float, default=None,
                        help='override default qk scale of head_dim ** -0.5 if set')

    # encoder Mamba
    parser.add_argument('--d_state', type=int, default=[1, 1], help='SSM state expansion factor')
    parser.add_argument('--d_conv', type=int, default=[3, 3], help='local convolution width')
    parser.add_argument('--expand', type=int, default=[1, 1], help='d_inner expansion factor')
    parser.add_argument('--dt_min', type=int, default=0.01, help='SSM dt_min')
    parser.add_argument('--dt_max', type=int, default=0.1, help='SSM dt_max')

    # --- agent ---
    parser.add_argument('--agent_embed_dim', type=int, default=[16], help='hidden dimensions in the agent model')
    parser.add_argument('--agent_depths', type=int, default=[1], help='number transformer blocks inside each layer')
    parser.add_argument('--agent_patch_size', type=int, default=(1, 1, 1), help='patch size inside transformer. keep it 1 for regression tasks')
    parser.add_argument('--agent_n_heads', type=int, default=[2], help='number of heads for self-attention')
    parser.add_argument('--agent_window_size', type=int, default=[(1, 1, 1)], help='window size for self-attention')
    parser.add_argument('--agent_mlp_ratio', type=float, default=4., help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--agent_drop_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--agent_attn_drop_rate', type=float, default=0.0, help='attention dropout rate')
    parser.add_argument('--agent_drop_path_rate', type=float, default=0.1, help='stochastic depth rate')
    parser.add_argument('--agent_qkv_bias', type=bool, default=True,
                        help='if True, add a learnable bias to query, key, value')
    parser.add_argument('--agent_qk_scale', type=float, default=None,
                        help='override default qk scale of head_dim ** -0.5 if set')
    parser.add_argument('--agent_patch_norm', type=bool, default=False,
                        help='if True, add normalization after patch embedding')
    parser.add_argument('--agent_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')

    # --- classifier ---
    parser.add_argument('--cls_dim', type=int, default=[512, 256, 1], help='input dimension for the classifier')
    parser.add_argument('--cls_drop_rate', type=float, default=0.5, help='drop rate for classification layer')

    parser.add_argument('--instance_drop_rate', type=float, default=0.5, help='dropout for instances/pixels ')

    parser.add_argument('--en_de_pretrained', type=str, default=None,
                        help='pretrained model i.e. a trained model with best loss')

    parser.add_argument('--times_train', type=tuple, default=(1, 52 * 34), help='time steps for training on synthetic data')
    parser.add_argument('--times_val', type=tuple, default=(52 * 34 + 1, 52 * 40), help='time steps for validation on synthetic data')
    parser.add_argument('--times_test', type=tuple, default=(52 * 40 + 1, 52 * 46), help='time steps for testing on synthetic data' )

    parser.add_argument('--delta_t', type=int, default=8, help='number of weeks or time steps')
    parser.add_argument('--window_size', type=int, default=1, help='scaling factor for resolution (2 means half)')

    parser.add_argument('--x_min', type=int, default=0, help='start of grid extension in x direction')
    parser.add_argument('--x_max', type=int, default=200, help='end of grid extension in x direction')
    parser.add_argument('--y_min', type=int, default=0, help='start of grid extension in y direction')
    parser.add_argument('--y_max', type=int, default=200, help='start of grid extension in y direction')

    parser.add_argument('--is_shuffle', type=bool, default=False, help='if True, apply data shuffling')
    parser.add_argument('--is_aug', type=bool, default=True, help='if True, apply data augmentation')
    parser.add_argument('--is_norm', type=bool, default=True, help='if True, apply data normalization')
    parser.add_argument('--is_clima_scale', type=bool, default=True, help='if True, apply data normalization with climatology')
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

    # --- DeepMIL hyperparameters ---
    parser.add_argument('--loss_lambda1', type=float, default=8e-5, help='lambda for smoothness loss')
    parser.add_argument('--loss_lambda2', type=float, default=8e-5, help='lambda for sparsity loss')
    parser.add_argument('--loss_k_deepmil', type=int, default=100, help='top k for DeepMIL')

    # --- ARNet hyperparameters ---
    parser.add_argument('--loss_alpha_arnet', type=float, default=400, help='alpha parameter for ARNet model')
    parser.add_argument('--loss_lambda_c_arnet', type=float, default=20, help='lambda c for ARNet model')

    # --- RTFM hyperparameters ---
    parser.add_argument('--dim_mtn_rtfm', type=int, default=32, help='dimension for multi-scale temporal network for RTFM model')
    parser.add_argument('--loss_alpha_rtfm', type=float, default=0.0001, help='alpha parameter for RTFM model')
    parser.add_argument('--loss_margin_rtfm', type=float, default=100, help='margin for RTFM model')
    parser.add_argument('--loss_k_rtfm', type=int, default=100, help='top k for RTFM')

    # --- MGFN hyperparameters ---
    parser.add_argument('--loss_k_mgfn', type=int, default=100, help='top k for MGFN')
    parser.add_argument('--loss_lambda_mgfn', type=float, default=0.0001, help='lambda parameter for MGFN model')
    parser.add_argument('--loss_margin_mgfn', type=float, default=100, help='margin for MGFN model')
    parser.add_argument('--alpha_mgfn', type=float, default=0.1, help='alpha parameter for MGFN model')
    parser.add_argument('--dim_head_mgfn', type=list, default=[16, 96], help='rhead dimension for MGFN model')
    parser.add_argument('--depths_mgfn', type=list, default=[1, 1], help='depths for MGFN model')
    parser.add_argument('--types_mgfn', type=list, default=['fb', 'fb'], help='types of layers for MGFN model')
    parser.add_argument('--lokernel_mgfn', type=int, default=5, help='local kernel for MGFN model')
    parser.add_argument('--ff_repe_mgfn', type=int, default=4, help='mlp ratio for MGFN model')
    parser.add_argument('--attention_drop_rate_mgfn', type=float, default=0., help='attention drop rate for MGFN model')

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


