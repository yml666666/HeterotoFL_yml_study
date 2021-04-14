import copy
from collections import OrderedDict

import numpy as np
import torch

from config import cfg


# Federation类里实现了联邦评估，联邦训练（全局模型训练参数的分发，和本地模型训练参数的汇总等功能。
class Federation:
    def __init__(self, global_parameters, rate, label_split):
        self.global_parameters = global_parameters
        self.rate = rate
        self.label_split = label_split
        self.make_model_rate()

    # 根据框架是否可动态调整分配本地模型计算复杂度rate
    def make_model_rate(self):
        if cfg['model_split_mode'] == 'dynamic':
            rate_idx = torch.multinomial(torch.tensor(cfg['proportion']), num_samples=cfg['num_users'],
                                         replacement=True).tolist()
            self.model_rate = np.array(self.rate)[rate_idx]
        elif cfg['model_split_mode'] == 'fix':
            self.model_rate = np.array(self.rate)
        else:
            raise ValueError('Not valid model split mode')
        return

    # 根据模型划分本地模型
    def split_model(self, user_idx):
        if cfg['model_name'] == 'conv':
            idx_i = [None for _ in range(len(user_idx))]
            idx = [OrderedDict() for _ in range(len(user_idx))]
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                for m in range(len(user_idx)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                input_size = v.size(1)
                                output_size = v.size(0)
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                if k == output_weight_name:
                                    output_idx_i_m = torch.arange(output_size, device=v.device)
                                else:
                                    scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                                idx[m][k] = output_idx_i_m, input_idx_i_m
                                idx_i[m] = output_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                        else:
                            if k == output_bias_name:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                            else:
                                input_idx_i_m = idx_i[m]
                                idx[m][k] = input_idx_i_m
                    else:
                        pass
        else:
            raise ValueError('Not valid model name')
        return idx

    # 给本地模型更新分配参数
    def distribute(self, user_idx):
        self.make_model_rate()
        param_idx = self.split_model(user_idx)
        local_parameters = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters, param_idx

    # 聚合本地模型参数
    def combine(self, local_parameters, param_idx, user_idx):
        count = OrderedDict()
        if cfg['model_name'] == 'conv':
            output_weight_name = [k for k in self.global_parameters.keys() if 'weight' in k][-1]
            output_bias_name = [k for k in self.global_parameters.keys() if 'bias' in k][-1]
            for k, v in self.global_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if k == output_weight_name:
                                    label_split = self.label_split[user_idx[m]]
                                    param_idx[m][k] = list(param_idx[m][k])
                                    param_idx[m][k][0] = param_idx[m][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                                    count[k][torch.meshgrid(param_idx[m][k])] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                        else:
                            if k == output_bias_name:
                                label_split = self.label_split[user_idx[m]]
                                param_idx[m][k] = param_idx[m][k][label_split]
                                tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                                count[k][param_idx[m][k]] += 1
                            else:
                                tmp_v[param_idx[m][k]] += local_parameters[m][k]
                                count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v += local_parameters[m][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        else:
            raise ValueError('Not valid model name')

        return
