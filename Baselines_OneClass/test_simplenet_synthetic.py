# ------------------------------------------------------------------
"""
Script for testing on the Synthetic dataset

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
import Baselines_OneClass.utils.utils_train as utils
from Baselines_OneClass.models.build_simplenet import SimpleNet, Backbone
import time
from torch.utils.tensorboard import SummaryWriter
from Baselines_OneClass.dataset.Synthetic_dataset import Synthetic_Dataset
import Baselines_OneClass.config as config_file

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

#torch.autograd.set_detect_anomaly(True)

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
        delta_t=config.delta_t,
        window_size=config.window_size
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=config.batch_size,
                                                 drop_last=False,
                                                 shuffle=False,
                                                 pin_memory=config.pin_memory,
                                                 num_workers=config.n_workers)

    utils.log_string(logger, "# evaluation samples: %d" % len(test_dataset))

    # get models
    utils.log_string(logger, "\nloading the model ...")

    if config.gpu_id != -1:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    backbone = Backbone(config).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    model = SimpleNet(config).to(device)

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "encoder parameters: %d" % utils.count_parameters(backbone.encoder))
    utils.log_string(logger, "pre_projection parameters: %d" % utils.count_parameters(model.pre_projection))
    utils.log_string(logger, "discriminator parameters: %d" % utils.count_parameters(model.discriminator))
    utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    # if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)

    model.to(device)

    utils.log_string(logger, 'testing on Artificial dataset ...\n')

    eval_test_anomaly = utils.evaluator_anomaly_synthetic(logger, 'Validation', config)
    test_anomaly_collector = utils.anomaly_collector(test_dataset.anomaly, test_dataset.timestep, config)

    time.sleep(1)

    # validation
    with torch.no_grad():

        model = model.eval()

        time.sleep(1)

        for i, (data_d, _, _, mask_extreme_loss, mask_anomaly, timestep) in tqdm(enumerate(test_dataloader),
                                                                                 total=len(test_dataloader),
                                                                                 smoothing=0.9,
                                                                                 postfix="  validation"):

            z = backbone(data_d.to(device))  # B, V, C, T, H , W
            z_n_scores = model(z).squeeze(-1)
            z_n_scores_tmp = z_n_scores.permute(1, 0, 2, 3, 4)[:, mask_extreme_loss == 0]

            anomaly = z_n_scores.clone()

            for v in range(config.in_channels_dynamic):
                extreme_thr = torch.median(z_n_scores_tmp[v]) - 0.001
                anomaly[:, v, ...][z_n_scores[:, v, ...] >= extreme_thr] = 0
                anomaly[:, v, ...][z_n_scores[:, v, ...] < extreme_thr] = 1

            test_anomaly_collector(anomaly.cpu().numpy(), timestep.cpu().numpy())

        test_anomaly_collector.majority_vote()

        eval_test_anomaly(np.swapaxes(test_anomaly_collector.anomaly, 0, 1), np.swapaxes(test_dataset.anomaly, 0, 1))
        eval_test_anomaly.get_results()


if __name__ == '__main__':

    test(config_file)

