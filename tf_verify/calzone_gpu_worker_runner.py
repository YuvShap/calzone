import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
import argparse
from calzone_gpu_worker import LZeroGpuWorker
from read_net_file import read_onnx_net
from onnx_translator import ONNXTranslator
from optimizer import Optimizer
from analyzer import layers
from config import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calzone Example', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', type=int, default=None, help='the port calzone\'s main is listening on')
    parser.add_argument('--netname', type=str, default=None, help='the network name, the extension can be only .onnx')
    parser.add_argument('--dataset', type=str, default=None, help='the dataset, can be either mnist, cifar10, or fashion')
    parser.add_argument('--means', nargs='+', type=float, default=None, help='the mean used to normalize the data with')
    parser.add_argument('--stds', nargs='+', type=float, default=None, help='the standard deviation used to normalize the data with')

    args = parser.parse_args()
    port = args.port
    netname = args.netname
    dataset = args.dataset
    means = args.means
    stds = args.stds


    # set ERAN config values
    config.netname = netname
    config.dataset = dataset
    config.domain = 'gpupoly'

    model, is_conv = read_onnx_net(netname)
    translator = ONNXTranslator(model, True)
    operations, resources = translator.translate()
    optimizer = Optimizer(operations, resources)
    nn = layers()
    network, relu_layers, num_gpu_layers = optimizer.get_gpupoly(nn)

    os.sched_setaffinity(0, cpu_affinity)

    l_zero_gpu_worker = LZeroGpuWorker(port=port, network=network, means=means, stds=stds, is_conv=is_conv, dataset=dataset)
    l_zero_gpu_worker.work()
