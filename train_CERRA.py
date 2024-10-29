# ------------------------------------------------------------------
"""
Script for training and validating on CERRA dataset

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
from models.losses import BCE_loss, Anomaly_L1_loss
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

# torch.autograd.set_detect_anomaly(True)
# ------------------------------------------------------------------


def train(config_file):
    # read config arguments
    config = config_file.read_arguments(train=True)

    # get logger
    logger = utils.get_logger(config)

    # get tensorboard writer
    writer = SummaryWriter(os.path.join(config.dir_log, config.name))

    # fix random seed
    utils.fix_seed(config.seed)

    # dataloader
    utils.log_string(logger, "loading training dataset ...")

    train_dataset = CERRA_Dataset(
        root_CERRA=config.root_CERRA,
        root_NOAA=config.root_NOAA_CERRA,
        nan_fill=config.nan_fill,
        is_aug=config.is_aug,
        is_shuffle=False,
        is_norm=config.is_norm,
        is_clima_scale=config.is_clima_scale,
        variables=config.variables,
        years=config.years_train,
        threshold=config.threshold,
        alpha=config.alpha,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        delta_t=config.delta_t,
        window_size=config.window_size
    )

    utils.log_string(logger, "loading validation dataset ...")
    val_dataset = CERRA_Dataset(
        root_CERRA=config.root_CERRA,
        root_NOAA=config.root_NOAA_CERRA,
        nan_fill=config.nan_fill,
        is_aug=False,
        is_shuffle=False,
        is_norm=config.is_norm,
        is_clima_scale=config.is_clima_scale,
        variables=config.variables,
        years=config.years_val,
        threshold=config.threshold,
        alpha=config.alpha,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        delta_t=config.delta_t,
        window_size=config.window_size
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   pin_memory=config.pin_memory,
                                                   num_workers=config.n_workers)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=config.batch_size,
                                                 drop_last=False,
                                                 shuffle=True,
                                                 pin_memory=config.pin_memory,
                                                 num_workers=config.n_workers)

    utils.log_string(logger, "# training samples: %d" % len(train_dataset))
    utils.log_string(logger, "# evaluation samples: %d" % len(val_dataset))

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

    # get losses
    utils.log_string(logger, "get criterion ...")

    criterion = BCE_loss().to(device)
    criterion_anomaly = Anomaly_L1_loss(n_dynamic=config.in_channels_dynamic,
                                        delta_t=config.delta_t,
                                        dim=config.en_embed_dim[-1]).to(device)

    # get optimizer
    utils.log_string(logger, "get optimizer and learning rate scheduler ...")

    optimizer = torch.optim.Adam([x for x in model.parameters() if x.requires_grad],
                                 lr=config.lr, weight_decay=config.weight_decay,
                                 )

    lr_scheduler = utils.get_learning_scheduler(optimizer, config)

    # DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    # training loop
    utils.log_string(logger, 'training on CERRA dataset ...\n')

    eval_train = utils.evaluator(logger, 'Training', config)
    eval_val = utils.evaluator(logger, 'Validation', config)

    time.sleep(1)

    # initialize the best values
    best_loss_train = np.inf
    best_loss_val = np.inf
    best_F1_train = 0
    best_F1_val = 0

    for epoch in range(config.n_epochs):
        utils.log_string(logger, '################# Epoch (%s/%s) #################' % (epoch + 1, config.n_epochs))

        # train
        model = model.train()
        loss_train = 0

        time.sleep(1)

        for i, (data_d, data_drought, data_drought_loss, data_cold_surface, data_cold_surface_loss,
                data_sea, data_no_vegetation, _) in tqdm(enumerate(train_dataloader),
                                                         total=len(train_dataloader),
                                                         smoothing=0.9,
                                                         postfix="  training"):

            optimizer.zero_grad(set_to_none=True)

            mask_valid = 1 - data_cold_surface  #- data_no_vegetation
            mask_valid[mask_valid < 0] = 0

            pred, pred_y, _, z_q, loss_z_q = model(data_d.to(device))

            # compute extreme loss
            loss = criterion(pred[:, 0, ...], data_drought.to(device), mask_valid.to(device))

            # compute anomaly loss
            if torch.cuda.device_count() > 1:
                loss_anomaly = criterion_anomaly(z_q,
                                                 data_drought_loss.to(device).float(),
                                                 data_cold_surface_loss.to(device),
                                                 model.module.vq.indices_to_codes(
                                                     torch.Tensor([0]).long().to(device)).detach())
            else:
                loss_anomaly = criterion_anomaly(z_q,
                                                 data_drought_loss.to(device).float(),
                                                 data_cold_surface_loss.to(device),
                                                 model.vq.indices_to_codes(
                                                     torch.Tensor([0]).long().to(device)).detach())

            # compute multi-head extreme loss
            loss_var = 0
            for k in range(config.in_channels_dynamic):
                loss_var += criterion(pred_y[k][:, 0, ...], data_drought.to(device).float(), mask_valid.to(device))

            loss = loss + loss_anomaly * config.lambda_anomaly + loss_var + loss_z_q

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()

            loss_train += loss.item()

            pred = torch.sigmoid(pred.detach().cpu())
            pred_c = pred.clone()
            pred_c[pred > 0.35] = 1
            pred_c[pred <= 0.35] = 0

            eval_train(pred_c[:, 0, ...].numpy(), data_drought.cpu().numpy(), mask_valid.cpu().numpy())

        mean_loss_train = loss_train / float(len(train_dataloader))
        eval_train.get_results(mean_loss_train, best_loss_train)

        if mean_loss_train <= best_loss_train:
            best_loss_train = mean_loss_train
        if eval_train.F1[1] >= best_F1_train:
            best_F1_train = eval_train.F1[1]

        #utils.save_model(model, optimizer, epoch, mean_loss_train, np.nan, logger, config, 'train')

        # validation
        with torch.no_grad():

            model = model.eval()
            loss_val = 0

            time.sleep(1)

            for i, (data_d, data_drought, data_drought_loss, data_cold_surface, data_cold_surface_loss,
                    data_sea, data_no_vegetation, _) in tqdm(enumerate(val_dataloader),
                                                             total=len(val_dataloader),
                                                             smoothing=0.9,
                                                             postfix="  validation"):

                optimizer.zero_grad(set_to_none=True)

                mask_valid = 1 - data_cold_surface  #- data_no_vegetation
                mask_valid[mask_valid < 0] = 0

                pred, pred_y, anomaly, z_q, loss_z_q = model(data_d.to(device))

                anomaly = anomaly.float()

                # compute extreme loss
                loss = criterion(pred[:, 0, ...], data_drought.to(device), mask_valid.to(device))

                # compute anomaly loss
                if torch.cuda.device_count() > 1:
                    loss_anomaly = criterion_anomaly(z_q,
                                                     data_drought_loss.to(device).float(),
                                                     data_cold_surface_loss.to(device),
                                                     model.module.vq.indices_to_codes(
                                                     torch.Tensor([0]).long().to(device)).detach())
                else:
                    loss_anomaly = criterion_anomaly(z_q,
                                                     data_drought_loss.to(device).float(),
                                                     data_cold_surface_loss.to(device),
                                                     model.vq.indices_to_codes(
                                                     torch.Tensor([0]).long().to(device)).detach())

                # compute multi-head extreme loss
                loss_var = 0
                for k in range(config.in_channels_dynamic):
                    loss_var += criterion(pred_y[k][:, 0, ...], data_drought.to(device).float(), mask_valid.to(device))

                loss = loss + loss_anomaly * config.lambda_anomaly + loss_var + loss_z_q

                loss_val += loss.item()

                pred = torch.sigmoid(pred.detach().cpu())
                pred_c = pred.clone()
                pred_c[pred > 0.35] = 1
                pred_c[pred <= 0.35] = 0

                eval_val(pred_c[:, 0, ...].numpy(), data_drought.cpu().numpy(), mask_valid.cpu().numpy())

                if i == len(val_dataloader) - 1:
                    # plot results
                    im_pred, im_pred_c, im_target = utils.generate_images(pred[:, 0, :, :].cpu().numpy(),
                                                                          pred_c[:, 0, :, :].numpy(),
                                                                          data_drought.cpu().numpy(),
                                                                          data_cold_surface.cpu().numpy(),
                                                                          data_sea.numpy(),
                                                                          data_no_vegetation.numpy(),
                                                                          mask_valid.numpy())

                    im_anomaly = utils.generate_anomaly(anomaly.cpu().numpy())

            mean_loss_val = loss_val / float(len(val_dataloader))
            eval_val.get_results(mean_loss_val, best_loss_val)

            if mean_loss_val <= best_loss_val:
                best_loss_val = mean_loss_val
                utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'loss')

            if eval_val.F1[1] >= best_F1_val:
                best_F1_val = eval_val.F1[1]
                utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'F1')

        writer.add_images('probability', im_pred, epoch + 1, dataformats='NHWC')
        writer.add_images('prediction', im_pred_c, epoch + 1, dataformats='NHWC')
        writer.add_images('target', im_target, epoch + 1, dataformats='NHWC')

        for v, var in enumerate(val_dataset.variables_dynamic):
            writer.add_images(var, im_anomaly[0, v, ...], epoch + 1, dataformats='HWC')

        writer.add_scalars("Loss", {'train': mean_loss_train, 'val': mean_loss_val}, epoch + 1)
        writer.add_scalars("IOU", {'train': eval_train.iou[1], 'val': eval_val.iou[1]}, epoch + 1)
        writer.add_scalars("F1", {'train': eval_train.F1[1], 'val': eval_val.F1[1]}, epoch + 1)

        eval_train.reset()
        eval_val.reset()

        lr_scheduler.step_update(epoch)


if __name__ == '__main__':
    train(config_file)

