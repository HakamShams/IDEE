# ------------------------------------------------------------------
"""
Script for testing on the Synthetic dataset for DeepMIL, ARNet, and RTFM models

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
import Baselines_MIL.utils.utils_train as utils
#from Baselines_MIL.models.losses import RankingLoss, DMIL_RankingLoss, CenterLoss, RTFMLoss
from Baselines_MIL.models import build_deepmil, build_arnet, build_rtfm
import time
import os
from torch.utils.tensorboard import SummaryWriter
from Baselines_MIL.dataset.Synthetic_dataset import Synthetic_Dataset
import Baselines_MIL.config as config_file

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

# ------------------------------------------------------------------

def test(config_file):
    # read config arguments
    config = config_file.read_arguments(train=True)

    # get logger
    logger = utils.get_logger(config)

    # fix random seed
    utils.fix_seed(config.seed)

    # dataloader
    utils.log_string(logger, "loading testing dataset ...")

    test_dataset = Synthetic_Dataset(
        root_datacube=config.root_synthetic,
        times=config.times_test,
        is_aug=False,
        is_norm=config.is_norm,
        is_clima_scale=config.is_clima_scale,
        variables=config.variables,
        variables_static=config.variables_static,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        delta_t=config.delta_t,
        window_size=config.window_size
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=config.batch_size,
                                                  drop_last=False,
                                                  shuffle=False,
                                                  pin_memory=config.pin_memory,
                                                  num_workers=config.n_workers)

    # train_dataloader = val_dataloader
    utils.log_string(logger, "# testing samples: %d" % len(test_dataset))

    # get models
    utils.log_string(logger, "\nloading the model and get criterion ...")

    if config.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    if config.classifier == 'DeepMIL':
        model = build_deepmil.MIL_model(config)
     #   criterion_ranking = RankingLoss(drop_rate=config.instance_drop_rate, k=config.loss_k_deepmil).to(device)
    elif config.classifier == 'ARNet':
        model = build_arnet.MIL_model(config)
      #  criterion_ranking = DMIL_RankingLoss(drop_rate=config.instance_drop_rate, alpha=config.loss_alpha_arnet,
       #                                      t=test_dataset.n_lat_window * test_dataset.n_lon_window).to(device)
      #  criterion_center = CenterLoss(lambda_c=config.loss_lambda_c_arnet).to(device)
    elif config.classifier == 'RTFM':
        model = build_rtfm.MIL_model(config)
      #  criterion_rtfm = RTFMLoss(drop_rate=config.instance_drop_rate,
       #                           alpha=config.loss_alpha_rtfm,
        #                          margin=config.loss_margin_rtfm,
         #                         k=config.loss_k_rtfm).to(device)
    else:
        raise NotImplementedError(f"Classifier {config.classifier} not implemented")

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "encoder parameters: %d" % utils.count_parameters(model.encoder))
    if config.classifier == 'RTFM':
        utils.log_string(logger, "agent parameters: %d" % utils.count_parameters(model.agent))
    utils.log_string(logger, "classifier parameters: %d" % utils.count_parameters(model.classifier))
    utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    # if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)

    model.to(device)

    utils.log_string(logger, 'testing on Artificial dataset ...\n')

    eval_test_anomaly = utils.evaluator_anomaly_synthetic(logger, 'Testing', config)
    test_anomaly_collector = utils.anomaly_collector(test_dataset.anomaly, test_dataset.timestep, config)

    time.sleep(1)

    # initialize the best values
    #best_loss_test = np.inf

    # testing
    with torch.no_grad():

        model = model.eval()
        #loss_test = 0
        time.sleep(1)

        for i, (data_d, _, _, _, mask_extreme_loss, mask_anomaly, timestep) in tqdm(enumerate(test_dataloader),
                                                                                    total=len(test_dataloader),
                                                                                    smoothing=0.9,
                                                                                    postfix="  validation"):

            if config.classifier == 'RTFM':
                z_n, z_p, z_n_feature, z_p_feature = model(data_d.to(device), mask_extreme_loss)
            else:
                z_n, z_p = model(data_d.to(device), mask_extreme_loss)
            """
            loss_ranking = torch.zeros(1, device=device)
            loss_centering = torch.zeros(1, device=device)    
            for k in range(len(z_n)):
                for v in range(config.in_channels_dynamic):
                    if config.classifier == 'RTFM':
                        loss_ranking += criterion_rtfm(z_p[k][:, v, ...], z_n[k][:, v, ...], z_p_feature[k][:, v, ...],
                                                       z_n_feature[k][:, v, ...], is_training=False)
                    else:
                        loss_ranking += criterion_ranking(z_p[k][:, v, ...], z_n[k][:, v, ...], is_training=False)
                        if config.classifier == 'ARNet':
                            loss_centering += criterion_center(z_n[k][:, v, ...])

            loss = (loss_ranking + loss_centering) / len(z_n)
            loss_test += loss.item()
            """

            anomaly = torch.zeros(mask_anomaly.shape, requires_grad=False, device=device)
            # anomaly  N, V, T, H, W
            # z_p  Np, V, T, C

            anomaly = anomaly.permute(1, 2, 0, 3, 4)  # anomaly  V, T, N, H, W

            for k in range(len(z_n)):
                anomaly[:, :, k, mask_extreme_loss[k] == 0] = z_n[k][:, :, :, 0].permute(1, 2, 0)
                anomaly[:, :, k, mask_extreme_loss[k] != 0] = z_p[k][:, :, :, 0].permute(1, 2, 0)

            anomaly = anomaly.permute(2, 0, 1, 3, 4)  # anomaly  V, T, N, H, W

            anomaly[anomaly > 0.5] = 1
            anomaly[anomaly <= 0.5] = 0

            test_anomaly_collector(anomaly.cpu().numpy(), timestep.cpu().numpy())

        #mean_loss_test = loss_test / float(len(test_dataloader))

        test_anomaly_collector.majority_vote()

        eval_test_anomaly(np.swapaxes(test_anomaly_collector.anomaly, 0, 1), np.swapaxes(test_dataset.anomaly, 0, 1))
        eval_test_anomaly.get_results()


if __name__ == '__main__':

    test(config_file)

