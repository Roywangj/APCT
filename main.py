import os, argparse, yaml, numpy as np
from openpcmae.utils import parser, dist_utils, misc
from openpcmae.utils.logger import *
from openpcmae.utils.config import *
import time
import os
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from tools.classification.runner_pretrain import run_net as pretrain
from tools.classification.runner_end2endtrain_valmodelnetc import run_net_valmodelnetc
from tools.classification.runner_end2endtrain_valscanobjectnnc import run_net_valscanobjectnnc
from tools.classification.runner_end2endtrain_valscanobjectnnc_3predicts_1 import run_net_valscanobjectnnc_3preds_1
from tools.classification.runner_end2endtrain_valmodelnetc_3predicts_1 import run_net_valmodelnetc_3preds_1



def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
            # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank,
                             deterministic=args.deterministic)  # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    # if not args.finetune_model:
    #     print_log(f'pretraining mode:', logger=logger)
    #     pretrain(args, config, train_writer, val_writer)
    # elif args.finetune_model:   #   e2etrain and finetune use shared training process, w/wo args.ckpts differs
    if config.mode == 'modelnetc':
        print_log(f'modelnetc mode', logger=logger)
        run_net_valmodelnetc(args, config, train_writer, val_writer)
    elif config.mode == 'scanobjectnnc':
        print_log(f'scanobjectnnc mode', logger=logger)
        run_net_valscanobjectnnc(args, config, train_writer, val_writer)
    elif config.mode == 'modelnetc3preds_1':
        print_log(f'modelnetc mode', logger=logger)
        run_net_valmodelnetc_3preds_1(args, config, train_writer, val_writer)
    elif config.mode == 'scanobjectnnc3preds_1':
        print_log(f'scanobjectnnc mode', logger=logger)
        run_net_valscanobjectnnc_3preds_1(args, config, train_writer, val_writer)

    pass


if __name__ == '__main__':
    main()
    print('==>ending..')