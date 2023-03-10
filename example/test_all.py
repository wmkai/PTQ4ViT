# from timm.models.layers import config
from torch.nn.modules import module
from test_vit import *
# from quant_layers.conv import MinMaxQuantConv2d
# from quant_layers.linear import MinMaxQuantLinear, PTQSLQuantLinear
# from quant_layers.matmul import MinMaxQuantMatMul, PTQSLQuantMatMul
import matplotlib.pyplot as plt
# from utils.net_wrap import wrap_certain_modules_in_net
from tqdm import tqdm
import torch.nn.functional as F
import pickle as pkl
from itertools import product
import types
from utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from utils.models import get_net
import time
import pdb
from torchstat import stat

def test_all(name, cfg_modifier=lambda x: x, calib_size=32, config_name="PTQ4ViT"):
    quant_cfg = init_config(config_name) # <module 'configs.PTQ4ViT' from '/hdd1/wangmingkai/codes/PTQ4ViT/./configs/PTQ4ViT.py'>
    quant_cfg = cfg_modifier(quant_cfg) # 调用cfg_modifier类的__call__函数

    net = get_net(name)
    # stat(net.to(torch.device('cpu')), (3, 224, 224)) # 计算param和flops

    wrapped_modules=net_wrap.wrap_modules_in_net(net,quant_cfg)

    g=datasets.ViTImageNetLoaderGenerator('/hdd1/data/imagenet','imagenet',32,32,16, kwargs={"model":net})
    test_loader=g.test_loader()
    calib_loader=g.calib_loader(num=calib_size)

    # add timing
    calib_start_time = time.time()
    quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
    quant_calibrator.batching_quant_calib()
    calib_end_time = time.time()

    acc = test_classification(net,test_loader, description=quant_cfg.ptqsl_linear_kwargs["metric"])

    print(f"model: {name} \n")
    print(f"calibration size: {calib_size} \n")
    print(f"bit settings: {quant_cfg.bit} \n")
    print(f"config: {config_name} \n")
    print(f"ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs} \n")
    print(f"ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs} \n")
    print(f"ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs} \n")
    print(f"calibration time: {(calib_end_time-calib_start_time)/60}min \n")
    print(f"accuracy: {acc} \n\n")

class cfg_modifier():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self,name,value)

    def __call__(self, cfg):
        # bit setting
        cfg.bit = self.bit_setting
        cfg.w_bit = {name: self.bit_setting[0] for name in cfg.conv_fc_name_list}
        cfg.w_bit["qconv"], cfg.w_bit["qlinear_qkv"], cfg.w_bit["qlinear_proj"], cfg.w_bit["qlinear_MLP_1"], cfg.w_bit["qlinear_MLP_2"], cfg.w_bit["qlinear_classifier"], cfg.w_bit["qlinear_reduction"] = \
            self.bit_setting[0][0], self.bit_setting[0][1], self.bit_setting[0][2], self.bit_setting[0][3], self.bit_setting[0][4], self.bit_setting[0][5], self.bit_setting[0][6],
        cfg.a_bit = {name: self.bit_setting[1] for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        # pdb.set_trace()

        # conv2d configs
        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        # linear configs
        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        # matmul configs
        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        return cfg

if __name__=='__main__':
    args = parse_args()

    names = [
        # "vit_tiny_patch16_224",
        # "vit_small_patch32_224",
        # "vit_small_patch16_224",
        # "vit_base_patch16_224",
        # "vit_base_patch16_384",

        # "deit_tiny_patch16_224",
        # "deit_small_patch16_224",
        # "deit_base_patch16_224",
        # "deit_base_patch16_384",

        # "swin_tiny_patch4_window7_224",
        # "swin_small_patch4_window7_224",
        "swin_base_patch4_window7_224",
        # "swin_base_patch4_window12_384",
        ]
    metrics = ["hessian"]
    linear_ptq_settings = [(1,1,1)] # n_V, n_H, n_a
    # calib_sizes = [32,128]
    # bit_settings = [(8,8), (6,6)] # weight, activation
    # config_names = ["PTQ4ViT", "BasePTQ"]

    calib_sizes = [32]
    bit_settings = [ # weight, activation
                    # (8,8),
                    # (6,6),
                    # vit weight:["qconv", "qlinear_qkv", "qlinear_proj", "qlinear_MLP_1", "qlinear_MLP_2", "qlinear_classifier", "qlinear_reduction"], activation
                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8]], 8), # vit_tiny_patch16_224
                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8]], 8), # vit_small_patch32_224
                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8]], 8), # vit_small_patch16_224
                    # SOTA: vit_base_patch16_224 6bit & 8bit
                    # ([6, [6, 5, 5, 6, 5, 4, 5, 5, 6, 5, 5, 6], [6, 5, 5, 6, 5, 6, 4, 5, 6, 5, 5, 6], [6, 5, 4, 6, 5, 5, 4, 5, 4, 5, 6, 6], [6, 4, 5, 6, 5, 6, 4, 5, 5, 6, 5, 6], 6, [6]], 6), # vit_base_patch16_224 6 bit
                    # ([8, [7, 6, 5, 7, 6, 7, 7, 7, 6, 7, 6, 7], [7, 6, 7, 5, 7, 6, 7, 6, 7, 6, 7, 7], [7, 6, 7, 5, 7, 6, 7, 6, 7, 6, 7, 7], [7, 6, 7, 7, 7, 7, 6, 7, 6, 7, 7, 6], 8, [8]], 8), # vit_base_patch16_224 8 bit

                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8]], 8), # vit_base_patch16_384

                    # deit
                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8]], 8), # deit_tiny_patch16_224
                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8]], 8), # deit_small_patch16_224
                    # SOTA: deit_base_patch16_224 8bit
                    # ([8, [8, 6, 5, 7, 6, 7, 8, 7, 6, 7, 6, 8], [8, 6, 7, 5, 7, 6, 7, 6, 7, 6, 7, 8], [8, 6, 7, 5, 7, 6, 7, 6, 7, 6, 7, 8], [8, 6, 8, 7, 8, 7, 6, 7, 6, 7, 8, 8], 8, [8]], 8), # deit_base_patch16_224

                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8]], 8), # deit_base_patch16_384

                    # swin
                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8, 8, 8]], 8), # swin_tiny_patch4_window7_224
                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8, 8, 8]], 8), # swin_small_patch4_window7_224

                    # SOTA: deit_base_patch16_224 8bit
                    # ([8, [6, 5, 6, 6, 6, 5, 6, 6, 5, 6, 6, 4, 6, 6, 6, 6, 5, 6, 6, 5, 6, 6, 6, 6], [6, 6, 6, 5, 6, 6, 6, 4, 6, 6, 6, 5, 5, 6, 6, 6, 6, 6, 6, 5, 6, 6, 5, 6], [6, 6, 5, 6, 6, 5, 6, 6, 6, 5, 6, 5, 6, 6, 6, 6, 5, 5, 6, 6, 6, 5, 6, 6], [6, 6, 5, 6, 6, 6, 6, 6, 5, 6, 6, 6, 5, 6, 6, 6, 5, 6, 6, 5, 5, 6, 6, 6], 6, [6, 6, 8]], 6), # swin_base_patch4_window7_224 6bit
                    ([8, [8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8], [8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 7, 8, 8, 7, 8, 8, 8], [8, 8, 8, 8, 8, 8, 7, 8, 8, 7, 8, 8, 8, 7, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8], [8, 8, 7, 8, 7, 8, 8, 8, 8, 8, 7, 8, 8, 7, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8], 8, [8, 8, 8]], 8), # swin_base_patch4_window7_224 8bit

                    # ([8, [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], 8, [8, 8, 8]], 8), # swin_base_patch4_window12_384
                    ]
    config_names = ["PTQ4ViT"]

    cfg_list = []
    for name, metric, linear_ptq_setting, calib_size, bit_setting, config_name in product(names, metrics, linear_ptq_settings, calib_sizes, bit_settings, config_names):
        cfg_list.append({
            "name": name,
            "cfg_modifier":cfg_modifier(linear_ptq_setting=linear_ptq_setting, metric=metric, bit_setting=bit_setting),
            "calib_size":calib_size,
            "config_name": config_name
        })

    if args.multiprocess:
        multiprocess(test_all, cfg_list, n_gpu=args.n_gpu) # 意思应该是每个GPU处理一个cfg_list，cfg_list作为experiment_process函数的参数
    else:
        for cfg in cfg_list:
            test_all(**cfg)