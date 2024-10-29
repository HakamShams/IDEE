# ------------------------------------------------------------------
"""
Script for training and validating STEALNET on the Synthetic dataset

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
from Baselines_Reconstruction.models.losses import STEALLoss
import Baselines_Reconstruction.utils.utils_train as utils
from Baselines_Reconstruction.models.build_steal import Rec_model
import time
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from Baselines_Reconstruction.dataset.Synthetic_dataset import Synthetic_Dataset
import Baselines_Reconstruction.config as config_file

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

#torch.autograd.set_detect_anomaly(True)

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

    train_dataset = Synthetic_Dataset(
        root_datacube=config.root_synthetic,
        times=config.times_train,
        is_aug=config.is_aug,
        is_shuffle=config.is_shuffle,
        is_norm=config.is_norm,
        is_clima_scale=config.is_clima_scale,
        is_replace_anomaly=False,
        variables=config.variables,
        variables_static=config.variables_static,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        delta_t=config.delta_t,
        window_size=config.window_size
    )

    utils.log_string(logger, "loading validation dataset ...")
    val_dataset = Synthetic_Dataset(
        root_datacube=config.root_synthetic,
        times=config.times_val,
        is_aug=False,
        is_shuffle=False,
        is_norm=config.is_norm,
        is_clima_scale=config.is_clima_scale,
        is_replace_anomaly=False,
        variables=config.variables,
        variables_static=config.variables_static,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max,
        delta_t=config.delta_t,
        window_size=config.window_size
    )

    # random_sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=len(val_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.batch_size,
                                                   # sampler=random_sampler,
                                                   shuffle=True,
                                                   pin_memory=config.pin_memory,
                                                   num_workers=config.n_workers)
    # persistent_workers=True)

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

    if config.gpu_id != -1:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    model = Rec_model(config)

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "encoder parameters: %d" % utils.count_parameters(model.encoder))
    utils.log_string(logger, "decoder parameters: %d" % utils.count_parameters(model.decoder))
    utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    # get losses
    utils.log_string(logger, "get criterion ...")

    criterion = STEALLoss(n_dynamic=config.in_channels_dynamic).to(device)

    # get optimizer
    utils.log_string(logger, "get optimizer and learning rate scheduler ...")

    #  optim_groups = utils.get_optimizer_groups_vq(model, config)
    #  optimizer = utils.get_optimizer(optim_groups, config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                                 betas=(config.beta1, config.beta2))

    lr_scheduler = utils.get_learning_scheduler(optimizer, config)

    # if torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model)

    model.to(device)

    # training loop
    utils.log_string(logger, 'training on Artificial dataset ...\n')

    # get collector for anomalous events prediction. Used in the evaluator
    train_anomaly_collector = utils.anomaly_collector(train_dataset.anomaly, train_dataset.timestep, config)
    val_anomaly_collector = utils.anomaly_collector(val_dataset.anomaly, val_dataset.timestep, config)

    # get evaluator for anomalous events prediction
    eval_train_anomaly = utils.evaluator_anomaly_synthetic(logger, 'Training', config)
    eval_val_anomaly = utils.evaluator_anomaly_synthetic(logger, 'Validation', config)

    time.sleep(1)

    # initialize the best values
    best_loss_train = np.inf
    best_loss_val = np.inf

    for epoch in range(config.n_epochs):
        utils.log_string(logger, '################# Epoch (%s/%s) #################' % (epoch + 1, config.n_epochs))

        # train
        model = model.train()
        loss_train = 0

        time.sleep(1)

        for i, (data_d, _, _, mask_extreme_loss, _, timestep) in tqdm(enumerate(train_dataloader),
                                                                      total=len(train_dataloader),
                                                                      smoothing=0.9,
                                                                      postfix="  training"):

            optimizer.zero_grad(set_to_none=True)
            data_d = data_d.squeeze(2).to(device)
            target = data_d.clone().detach().requires_grad_(False)

            pred = model(data_d)
            loss = criterion(pred, target, mask_extreme_loss.to(device))

            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        #    p_scores = 10 * torch.log10(100 / F.mse_loss(pred, target, reduction='none'))
            p_scores = F.mse_loss(pred, target, reduction='none')
            p_scores_n = torch.mean(p_scores.permute(1, 0, 2, 3, 4)[:, mask_extreme_loss == 0])
            p_scores_p = torch.mean(p_scores.permute(1, 0, 2, 3, 4)[:, mask_extreme_loss != 0])
            extreme_score = (p_scores_n + p_scores_p) / 2

            anomaly = p_scores.clone()
            #for v in range(config.inplanes):
            #    p_scores_n = torch.mean(p_scores.permute(1, 0, 2, 3, 4)[v, mask_extreme_loss == 0])
            #    p_scores_p = torch.mean(p_scores.permute(1, 0, 2, 3, 4)[v, mask_extreme_loss != 0])
            #    extreme_score = (p_scores_n + p_scores_p) / 2
            #    anomaly[:, v, ...][p_scores[:, v, ...] > extreme_score] = 1
            #    anomaly[:, v, ...][p_scores[:, v, ...] <= extreme_score] = 0

            anomaly[p_scores > extreme_score] = 1
            anomaly[p_scores <= extreme_score] = 0

            train_anomaly_collector(anomaly.detach().cpu().numpy(), timestep.detach().cpu().numpy())

        mean_loss_train = loss_train / float(len(train_dataloader))
        train_anomaly_collector.majority_vote()
        eval_train_anomaly(np.swapaxes(train_anomaly_collector.anomaly, 0, 1), np.swapaxes(train_dataset.anomaly, 0, 1))
        eval_train_anomaly.get_results()

        utils.log_string(logger, '%s mean loss     : %.4f' % ('Training', mean_loss_train))
        utils.log_string(logger, '%s best mean loss: %.4f\n' % ('Training', best_loss_train))

        if mean_loss_train <= best_loss_train:
            best_loss_train = mean_loss_train

        #utils.save_model(model, optimizer, epoch, mean_loss_train, np.nan, logger, config, 'train')

        # validation
        with torch.no_grad():

            model = model.eval()
            loss_val = 0

            time.sleep(1)

            extreme_score_mean = 0  #[0 for _ in range(config.inplanes)]

            for i, (data_d, _, _, mask_extreme_loss, mask_anomaly, timestep) in tqdm(enumerate(val_dataloader),
                                                                                     total=len(val_dataloader),
                                                                                     smoothing=0.9,
                                                                                     postfix="  validation"):

                data_d = data_d.squeeze(2).to(device)
                target = data_d.clone()

                pred = model(data_d)
                loss = criterion(pred, target, mask_extreme_loss.to(device))

                loss_val += loss.item()

            #    p_scores = 10 * torch.log10(100 / F.mse_loss(pred, target, reduction='none'))
                p_scores = F.mse_loss(pred, target, reduction='none')
                p_scores_n = torch.mean(p_scores.permute(1, 0, 2, 3, 4)[:, mask_extreme_loss == 0])
                p_scores_p = torch.mean(p_scores.permute(1, 0, 2, 3, 4)[:, mask_extreme_loss != 0])
                extreme_score = (p_scores_n + p_scores_p) / 2

                anomaly = p_scores.clone()
                anomaly[p_scores > extreme_score] = 1
                anomaly[p_scores <= extreme_score] = 0

                #for v in range(config.inplanes):
                #    p_scores_n = torch.mean(p_scores.permute(1, 0, 2, 3, 4)[v, mask_extreme_loss == 0])
                #    p_scores_p = torch.mean(p_scores.permute(1, 0, 2, 3, 4)[v, mask_extreme_loss != 0])
                #    extreme_score = (p_scores_n + p_scores_p) / 2

                #    anomaly[:, v, ...][p_scores[:, v, ...] > extreme_score] = 1
                #    anomaly[:, v, ...][p_scores[:, v, ...] <= extreme_score] = 0

                #    extreme_score_mean[v] += extreme_score

                extreme_score_mean += extreme_score

                val_anomaly_collector(anomaly.cpu().numpy(), timestep.cpu().numpy())

                if i == len(val_dataloader) - 1:
                    # plot results
                    im_anomaly = utils.generate_anomaly(anomaly.cpu().numpy())
                    im_anomaly_gt = utils.generate_anomaly(mask_anomaly.cpu().numpy())
                    im_anomaly = np.concatenate((im_anomaly, im_anomaly_gt), axis=2)

            mean_loss_val = loss_val / float(len(val_dataloader))

            val_anomaly_collector.majority_vote()
            eval_val_anomaly(np.swapaxes(val_anomaly_collector.anomaly, 0, 1), np.swapaxes(val_dataset.anomaly, 0, 1))
            eval_val_anomaly.get_results()

            utils.log_string(logger, '%s mean loss     : %.4f' % ('Validation', mean_loss_val))
            utils.log_string(logger, '%s best mean loss: %.4f\n' % ('Validation', best_loss_val))

            if mean_loss_val <= best_loss_val:
                best_loss_val = mean_loss_val
                utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'loss')
                utils.log_string(logger, 'thresholds % .4f' % (extreme_score_mean/float(len(val_dataloader))))

                #for v in range(config.inplanes):
                #    utils.log_string(logger, 'thresholds variable % s % .4f' % (v+1, extreme_score_mean[v]/float(len(val_dataloader))))

        for v, var in enumerate(val_dataset.variables_dynamic):
            writer.add_images(var, im_anomaly[0, v, ...], epoch + 1, dataformats='HWC')

        writer.add_scalars("Loss", {'train': mean_loss_train, 'val': mean_loss_val}, epoch + 1)

        eval_train_anomaly.reset()
        eval_val_anomaly.reset()

        train_anomaly_collector.reset()
        val_anomaly_collector.reset()

        # lr_scheduler.step()
        lr_scheduler.step_update(epoch)

        del mask_extreme_loss, mask_anomaly, im_anomaly, im_anomaly_gt, p_scores, anomaly


if __name__ == '__main__':

    train(config_file)

