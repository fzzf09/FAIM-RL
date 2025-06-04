import os
from os.path import join
import torch
import multiprocessing

# 设置默认参数（替代 argparse）
class Args:
    bpr_batch = 4096
    recdim = 64
    layer = 3
    lr = 0.001
    decay = 1e-4
    dropout = 0
    keepprob = 0.6
    a_fold = 100
    testbatch = 100
    dataset = 'gowalla'
    path = "./checkpoints"
    topks = "[20]"
    tensorboard = 1
    comment = "lgn"
    load = 0
    epochs = 2000
    multicore = 0
    pretrain = 0
    seed = 2020
    model = 'lgn'

args = Args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')

import sys
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
config['train_epochs'] =100
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

GPU = torch.cuda.is_available()
device_id = 7
device=torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
