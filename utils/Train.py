import os.path as osp
import yaml
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.nn.parallel import DataParallel
from utils.Auxiliary import move2device
from datasets.Datasets import My_Dataset
import random


# Train
# Read training parameters
def conform_args(model_name, dataset_name, epochs, batchsize, more_gpu, debug):
    args = yaml.safe_load(open(f'cfgs/TRAIN.yaml'))
    args['model_name'] = model_name
    args['dataset_name'] = dataset_name
    args['more_gpu'] = more_gpu 
    if args['exp_name_mode'] == 'mape':
        args['exp_name'] = 'temporarily'
    elif args['exp_name_mode'] == 'model_name':
        args['exp_name'] = model_name
    else:
        args['exp_name'] = args['exp_name_mode']
    args['epochs'] = epochs
    args['batch_size'] = batchsize
    args['datasets'] = yaml.safe_load(open(f'cfgs/datasets/{dataset_name}.yaml'))['dataset']
    args['model'] = yaml.safe_load(open(f'cfgs/model/{model_name}.yaml'))['model']
    args['model']['num_features'] = len(args['datasets']['features'])
    args['model']['which_transition_matrices'] = args['datasets']['which_transition_matrices']
    args['model']['in_len'] = args['datasets']['in_len']
    args['model']['out_len'] = args['datasets']['out_len']
    args['model']['top_k'] = args['datasets']['top_k']
    args['model']['num_nodes'] = args['datasets']['num_nodes']
    args['model']['use_curriculum_learning'] = args['use_curriculum_learning']
    args['model']['cl_decay_steps'] = args['cl_decay_steps']
    args['model']['temporal_num_embeddings'] = args['datasets']['temporal_num_embeddings']
    args['model']['spatial_num_embeddings'] = args['datasets']['spatial_num_embeddings']
    args['model']['eigenmaps_k'] = args['datasets']['eigenmaps_k']
    if debug == True:
        args['datasets']['debug'] = True
        args['batch_size'] = batchsize
        args['epochs'] = epochs
    else:
        args['datasets']['debug'] = False

    return args


# Visualization of forecast results
def tensorboard(write, target, output, out_len=0):
    B, T, N, C = target.shape
    B = min(B, 1000)
    index = random.sample(range(0, N), 10)
    for b in range(B):
        for c in range(C):
            for node in index:
                write.add_scalars(main_tag=f'result-feature{c}/node{node}',
                                  tag_scalar_dict={'target': target[b, 0, node, c]},
                                  global_step=b)
                for lenl in range(out_len):
                    write.add_scalars(main_tag=f'result-feature{c}/node{node}',
                                      tag_scalar_dict={f'output-{lenl + 1}': output[b, lenl, node, c]},
                                      global_step=b + lenl)


# gen_train_val_data
def gen_train_val_data(args):
    train_set = My_Dataset(args['datasets'], split='train')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                                               pin_memory=True, drop_last=True)

    val_set = My_Dataset(args['datasets'], split='val')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args['batch_size'], shuffle=False,
                                             pin_memory=True, drop_last=False)

    return (train_set, train_loader), (val_set, val_loader)


# gen_test_data
def gen_test_data(args):
    test_set = My_Dataset(args['datasets'], split='test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, pin_memory=True,
                                              drop_last=False)
    return test_set, test_loader


# build_model
def build_model(args, mode, device, state_dict=None, **kwargs):
    '''

    :param args:
    :param mode:
    :param device:
    :param state_dict: 历史参数
    :param kwargs:
    :return:
    '''
    exec('from models.{0} import {0}'.format(args['model_name']), globals())
    cfgs = args['model']
    model = eval(args['model_name'])(cfgs)
    if torch.cuda.device_count() > 1 and args['more_gpu']:
        model = MyDataParallel(model)
    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict(state_dict_2_1(state_dict))

    exec(f'model.{mode}()')  # net.train /net.eval
    model.to(device)

    return model


# load_model
def load_model(file):
    save_dict = torch.load(file, map_location='cpu')
    statics = save_dict['statics']
    state_dict = save_dict['model']

    return statics, state_dict


# ancillary
# Error Aggregate Averager
class Average(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def add(self, value, count):
        self._sum += value
        self._count += count

    def average(self):
        return self._sum / self._count


# Dataset model parameter loading
def get_dataset_model_args(dataset, model):
    '''
    数据库相关参数
    :param dataset:
    :param model:
    :return:
    '''
    dataset_model_args = yaml.safe_load(open(osp.join('cfgs', f'{dataset}_{model}.yaml')))
    return dataset_model_args


# Multi-GPU data segmentation
class MyDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        kwargs2 = {1:torch.rand(size=[6,6],dtype=torch.float32)}
        inputs, _ = scatter_kwargs(inputs, kwargs2, device_ids, dim=self.dim)
        # 不划分 'statics' 参数
        kwargs = (move2device(kwargs, f'cuda:{0}'), move2device(kwargs, f'cuda:{1}'))
        return inputs, kwargs


# Multi-GPU parameter conversion
def state_dict_2_1(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict
