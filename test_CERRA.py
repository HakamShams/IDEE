# ------------------------------------------------------------------
"""
Script for testing on CERRA dataset

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
from dataset.CERRA_dataset import CERRA_Dataset
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
    utils.log_string(logger, "loading test dataset ...")

    test_dataset = CERRA_Dataset(
        root_CERRA=config.root_CERRA,
        root_NOAA=config.root_NOAA_CERRA,
        nan_fill=config.nan_fill,
        is_aug=False,
        is_shuffle=False,
        is_norm=config.is_norm,
        is_clima_scale=config.is_clima_scale,
        variables=config.variables,
        years=config.years_test,
        threshold=config.threshold,
        alpha=config.alpha,
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

    # testing loop
    utils.log_string(logger, 'testing on CERRA dataset ...\n')

    eval_test = utils.evaluator(logger, 'Testing', config)

    time.sleep(1)

    # testing
    with torch.no_grad():

        model.eval()
        time.sleep(1)

        for i, (data_d, data_drought, data_drought_loss, data_cold_surface, data_cold_surface_loss,
                data_sea, data_no_vegetation, file_name) in tqdm(enumerate(test_dataloader),
                                                         total=len(test_dataloader),
                                                         smoothing=0.9,
                                                         postfix="  validation"):

            mask_valid = 1 - data_sea - data_cold_surface - data_no_vegetation
            mask_valid[mask_valid < 0] = 0

            pred, _, _, _, _ = model(data_d.to(device))

            pred = torch.sigmoid(pred.detach().cpu())
            pred_c = pred.clone()
            pred_c[pred > 0.35] = 1
            pred_c[pred <= 0.35] = 0

            eval_test(pred_c[:, 0, :, :].numpy(), data_drought.cpu().numpy(), mask_valid.cpu().numpy())

        eval_test.get_results(0, 0)


if __name__ == '__main__':

    test(config_file)

