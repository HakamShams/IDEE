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
import utils.utils_train as utils
from models.build import VQ_model
import time
import os
from torch.utils.tensorboard import SummaryWriter
from dataset.Synthetic_dataset import Synthetic_Dataset
import config as config_file

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

    utils.log_string(logger, "# testing samples: %d" % len(test_dataset))

    # get models
    utils.log_string(logger, "\nloading the model ...")

    if config.gpu_id != "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    model = VQ_model(config)

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "encoder parameters: %d" % utils.count_parameters(model.encoder))
    utils.log_string(logger, "classifier parameters: %d" % utils.count_parameters(model.cls))
    utils.log_string(logger, "vq parameters: %d" % utils.count_parameters(model.vq))
    utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    # DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    utils.log_string(logger, 'testing on Artificial dataset ...\n')

    # get evaluator for extreme events prediction
    eval_test = utils.evaluator_synthetic(logger, 'Testing')
    # get collector for anomalous events prediction. Used in the evaluator
    test_anomaly_collector = utils.anomaly_collector(test_dataset.anomaly, test_dataset.timestep, config)
    # get evaluator for anomalous events prediction
    eval_test_anomaly = utils.evaluator_anomaly_synthetic(logger, 'Validation', config)

    time.sleep(1)

    # testing
    with torch.no_grad():

        model = model.eval()
        time.sleep(1)

        for i, (data_d, _, _,
                mask_extreme, mask_extreme_loss, mask_anomaly, timestep) in tqdm(enumerate(test_dataloader),
                                                                                 total=len(test_dataloader),
                                                                                 smoothing=0.9,
                                                                                 postfix="  validation"):

            mask_extreme = mask_extreme.unsqueeze(1)

            pred, _, anomaly, _, _ = model(data_d.to(device))

            anomaly = anomaly.float()

            pred = torch.sigmoid(pred.detach().cpu())
            pred_c = pred.clone()
            pred_c[pred > 0.5] = 1
            pred_c[pred <= 0.5] = 0

            eval_test(pred_c.numpy(), mask_extreme.cpu().numpy())
            test_anomaly_collector(anomaly.cpu().numpy(), timestep.cpu().numpy())

        test_anomaly_collector.majority_vote()
        eval_test_anomaly(np.swapaxes(test_anomaly_collector.anomaly, 0, 1), np.swapaxes(test_dataset.anomaly, 0, 1))
        eval_test_anomaly.get_results()
        eval_test.get_results(0, 0)


if __name__ == '__main__':

    test(config_file)

