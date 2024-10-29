# ------------------------------------------------------------------
"""
Script for testing UniAD on the Synthetic dataset

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
import utils.utils_train as utils
from models.build_uniad import UniAD
import time
from torch.utils.tensorboard import SummaryWriter
from dataset.Synthetic_dataset import Synthetic_Dataset
import config as config_file

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

# torch.autograd.set_detect_anomaly(True)

# ------------------------------------------------------------------


def test(config_file):

    # read config arguments
    config = config_file.read_arguments(train=False)

    # get logger
    logger = utils.get_logger(config)

    # fix random seed
    utils.fix_seed(config.seed)

    # dataloader
    utils.log_string(logger, "loading testing dataset ...")

    test_dataset = Synthetic_Dataset(
        root_datacube=config.root_synthetic,
        is_aug=False,
        is_norm=config.is_norm,
        is_clima_scale=config.is_clima_scale,
        is_replace_anomaly=False,
        variables=config.variables,
        variables_static=config.variables_static,
        times=config.times_test,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        delta_t=1,
        window_size=config.window_size
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=config.batch_size,
                                                  drop_last=False,
                                                  shuffle=True,
                                                  pin_memory=config.pin_memory,
                                                  num_workers=config.n_workers)

    utils.log_string(logger, "# testing samples: %d" % len(test_dataset))

    # get models
    utils.log_string(logger, "\nloading the model ...")

    if config.gpu_id != -1:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    model = UniAD(inplanes=config.inplanes, instrides=config.instrides, feature_size=config.feature_size,
                  feature_jitter_scale=config.feature_jitter_scale, feature_jitter_prob=config.feature_jitter_prob,
                  neighbor_size=config.neighbor_size, neighbor_mask=config.neighbor_mask, hidden_dim=config.hidden_dim,
                  pos_embed_type=config.pos_embed_type, initializer=config.initializer,
                  nhead=config.nhead, num_encoder_layers=config.num_encoder_layers,
                  num_decoder_layers=config.num_decoder_layers, dim_feedforward=config.dim_feedforward,
                  dropout=config.dropout, activation=config.activation, normalize_before=config.normalize_before,
                  return_intermediate_dec=config.return_intermediate_dec, en_de_pretrained=config.en_de_pretrained
                  )

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "encoder parameters: %d" % utils.count_parameters(model.transformer.encoder))
    utils.log_string(logger, "decoder parameters: %d" % utils.count_parameters(model.transformer.decoder))
    utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    # training loop
    utils.log_string(logger, 'testing on Artificial dataset ...\n')

    eval_test_anomaly = utils.evaluator_anomaly_synthetic(logger, 'Testing', config)
    test_anomaly_collector = utils.anomaly_collector(test_dataset.anomaly, test_dataset.timestep, config)

    time.sleep(1)

    # initialize the best values
    #best_loss_test = np.inf

    # validation
    with torch.no_grad():

        model = model.eval()
        #loss_test = 0

        time.sleep(1)
        extreme_score_mean = 0  # [0 for _ in range(config.inplanes)]

        for i, (data_d, _, _, mask_extreme_loss, mask_anomaly, timestep) in tqdm(enumerate(test_dataloader),
                                                                                 total=len(test_dataloader),
                                                                                 smoothing=0.9,
                                                                                 postfix="  validation"):

            mask_extreme_loss = mask_extreme_loss[:, 0, :, :]
            p_scores = model(data_d[:, :, 0, 0, :, :].to(device), mask_extreme_loss)
            #loss_test += p_scores.mean().item()

            p_scores = p_scores.permute(1, 0, 2, 3)
            p_scores[:, mask_extreme_loss == 1] = p_scores[:, mask_extreme_loss == 1] * -1

            p_scores_n = torch.mean(p_scores[:, mask_extreme_loss == 0])
            p_scores_p = torch.mean(p_scores[:, mask_extreme_loss == 1])
            extreme_score = (p_scores_n + p_scores_p) / 2

            p_scores = p_scores.permute(1, 0, 2, 3)
            anomaly = p_scores.clone()
            anomaly[p_scores > extreme_score] = 1
            anomaly[p_scores <= extreme_score] = 0

            anomaly = anomaly.unsqueeze(2)

            extreme_score_mean += extreme_score
            test_anomaly_collector(anomaly.cpu().numpy(), timestep.cpu().numpy())

        #mean_loss_test = loss_test / float(len(test_dataloader))

        test_anomaly_collector.majority_vote()

        eval_test_anomaly(np.swapaxes(test_anomaly_collector.anomaly, 0, 1), np.swapaxes(test_dataset.anomaly, 0, 1))
        eval_test_anomaly.get_results()


if __name__ == '__main__':

    test(config_file)

