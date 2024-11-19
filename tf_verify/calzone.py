import sys
import os
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, '../ELINA/python_interface/')
import argparse
from calzone_config import CalzoneConfig
from pprint import pprint
import csv
from calzone_robustness_analyzer import LZeroRobustnessAnalyzer
from read_net_file import read_onnx_net
from onnx_translator import ONNXTranslator
from optimizer import Optimizer
from analyzer import layers
from multiprocessing.connection import Listener
from subprocess import Popen
import numpy as np
import time
import json
from calzone_utils import normalize
import itertools
from config import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calzone Example', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--netname', type=str, default=CalzoneConfig.netname, help='the network name, the extension can be only .onnx')
    parser.add_argument('--dataset', type=str, default=CalzoneConfig.dataset, help='the dataset, can be either mnist, cifar10, or fashion')
    parser.add_argument('--num_tests', type=int, default=CalzoneConfig.num_tests, help='the number of images to test')
    parser.add_argument('--from_test', type=int, default=CalzoneConfig.from_test, help='the first image index')
    parser.add_argument("--t", type=int, default=CalzoneConfig.t, help="the maximal number of perturbed pixels")
    parser.add_argument("--rep_num", type=int, default=CalzoneConfig.rep_num, help="the number of samples for each subset size")
    parser.add_argument("--timeout", type=int, default=CalzoneConfig.timeout, help="the analysis timeout for a single image (in seconds)")
    parser.add_argument("--gpu_num", type=int, default=CalzoneConfig.gpu_num, help="the number of GPUs used in the analysis")
    parser.add_argument("--milp_num", type=int, default=CalzoneConfig.milp_num, help="the number of MILP instances used in the analysis")
    parser.add_argument("--logname", type=str, default=None, help="the name of the log file (.json will be added), if not specified timestamp will be used")
    parser.add_argument('--mean', nargs='+', type=float, default=CalzoneConfig.mean, help='the mean used to normalize the data with, ignored if extracted from the network, using a deafault if not provided')
    parser.add_argument('--std', nargs='+', type=float, default=CalzoneConfig.std, help='the standard deviation used to normalize the data with, ignored if extracted from the network, using a deafault if not provided')

    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(CalzoneConfig, k, v)

    CalzoneConfig.json = vars(args)
    pprint(CalzoneConfig.json)

    assert CalzoneConfig.netname, 'a network has to be provided for analysis.'
    netname = CalzoneConfig.netname
    assert os.path.isfile(netname), f'Model file not found. Please check \'{netname}\' is correct.'
    filename, file_extension = os.path.splitext(netname)
    assert file_extension == ".onnx", "file extension not supported"

    dataset = CalzoneConfig.dataset
    assert dataset in ['mnist', 'cifar10', 'fashion'], "only mnist, cifar10 and fashion datasets are supported"

    assert CalzoneConfig.num_tests >= 0 and CalzoneConfig.from_test >= 0, 'num_tests and from_test must be non-negative integers'
    assert CalzoneConfig.t in [1, 2, 3, 4, 5], 't must be 1,2,3,4 or 5'
    assert CalzoneConfig.rep_num >= 1, 'rep_num must be positive integer'
    assert CalzoneConfig.timeout >= 1, 'timeout must be positive integer'
    assert CalzoneConfig.gpu_num >= 1, 'gpu_num must be positive integer'
    assert CalzoneConfig.milp_num >= 1, 'milp_num must be positive integer'

    # set ERAN config values
    config.netname = netname
    config.dataset = dataset
    config.domain = 'gpupoly'

    domain = 'gpupoly'

    model, is_conv = read_onnx_net(netname)
    translator = ONNXTranslator(model, True)
    operations, resources = translator.translate()
    optimizer = Optimizer(operations, resources)
    nn = layers()
    network, relu_layers, num_gpu_layers = optimizer.get_gpupoly(nn)

    # Extracted from the network
    if config.mean is not None:
        means = config.mean
        stds = config.std
    # Specified by the user
    elif CalzoneConfig.mean is not None:
        assert CalzoneConfig.std is not None, 'if the mean is provided, the standard deviation must also be provided'
        means = CalzoneConfig.mean
        stds = CalzoneConfig.std
    # Default values
    elif dataset == 'mnist':
        means = [0.1307]
        stds = [0.30810001]
    elif dataset == 'fashion':
        means = [0.5]
        stds = [0.5]
    elif dataset == 'cifar10':
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2470, 0.2435, 0.2616]

    assert means is not None and stds is not None
    assert len(means) == len(stds), 'means len and stds len must be the same'
    if dataset =='cifar10':
        assert len(means) == 3, 'cifar10 means and stds len must be 3'
    else:
        assert len(means) == 1, 'mnist or fashion means and stds len must be 1'

    os.sched_setaffinity(0, cpu_affinity)

    correctly_classified_images = 0
    verified_images = 0
    unsafe_images = 0
    cum_time = 0

    csvfilename = '../data/{}_test.csv'.format(dataset if dataset != 'fashion' else 'fashion-mnist')
    csvfile = open(csvfilename, 'r')
    tests = csv.reader(csvfile, delimiter=',')
    if dataset == 'fashion':
        next(tests)

    address = ('localhost', 0)
    with Listener(address) as gpu_listener:
        with Listener(address) as cpu_listener:
            for worker in range(CalzoneConfig.gpu_num):
                my_env = os.environ.copy()
                my_env["CUDA_VISIBLE_DEVICES"] = str(worker % CalzoneConfig.gpu_num)
                Popen(["python3.8", "calzone_gpu_worker_runner.py", "--port", str(gpu_listener.address[1]), "--dataset", CalzoneConfig.dataset,
                       "--netname", CalzoneConfig.netname, "--means", *map(str, means), "--stds", *map(str, stds)], env=my_env)
            for worker in range(CalzoneConfig.milp_num):
                my_env = os.environ.copy()
                my_env["CUDA_VISIBLE_DEVICES"] = str(worker % CalzoneConfig.gpu_num)
                Popen(["python3.8", "calzone_cpu_worker_runner.py", "--port", str(cpu_listener.address[1]), "--dataset", CalzoneConfig.dataset,
                       "--netname", CalzoneConfig.netname, "--means", *map(str, means), "--stds", *map(str, stds),
                       "--timeout", str(CalzoneConfig.timeout)], env=my_env)

            gpu_workers = []
            print("Waiting for gpu workers")
            for i in range(CalzoneConfig.gpu_num):
                gpu_workers.append(gpu_listener.accept())

            cpu_workers = []
            print("Waiting for cpu workers")
            for i in range(CalzoneConfig.milp_num):
                cpu_workers.append(cpu_listener.accept())

    image_results_by_image_index = dict()
    for i, test in enumerate(tests):
        if CalzoneConfig.from_test and i < CalzoneConfig.from_test:
            continue

        if CalzoneConfig.num_tests is not None and i >= CalzoneConfig.from_test + CalzoneConfig.num_tests:
            break

        image = np.float64(test[1:len(test)]) / np.float64(255)
        specLB = np.copy(image)
        specUB = np.copy(image)
        normalize(specLB, means, stds, dataset, domain, is_conv)
        normalize(specUB, means, stds, dataset, domain, is_conv)
        start = time.time()
        is_correctly_classified = network.test(specLB, specUB, int(test[0]), True)
        image_results = dict()
        image_results['is_correctly_classified'] = is_correctly_classified
        label = int(test[0])
        image_results['true_label'] = label
        if is_correctly_classified:
            correctly_classified_images += 1

            l_zero_robustness_analyzer = LZeroRobustnessAnalyzer(image_index=i, image=image,
                                                                 label=label,
                                                                 gpu_workers=gpu_workers,
                                                                 cpu_workers=cpu_workers,
                                                                 t=CalzoneConfig.t,
                                                                 sampling=CalzoneConfig.rep_num,
                                                                 timeout=CalzoneConfig.timeout,
                                                                 dataset=dataset)
            analysis_summary = l_zero_robustness_analyzer.analyze()
            image_results['analysis_summary'] = analysis_summary

            if analysis_summary['verified']:
                verified_images += 1
            elif not analysis_summary['timed_out']:
                unsafe_images += 1

            end = time.time()
            cum_time += end - start  # only count samples where we did try to certify
        else:
            print("img", i, "not considered, incorrectly classified")
            end = time.time()

        image_results['running_time'] = end - start
        image_results_by_image_index[i] = image_results

        print(f"progress: {1 + i - CalzoneConfig.from_test}/{CalzoneConfig.num_tests}, "
              f"correct:  {correctly_classified_images}/{1 + i - CalzoneConfig.from_test}, "
              f"verified: {verified_images}/{correctly_classified_images}, "
              f"unsafe: {unsafe_images}/{correctly_classified_images}, ",
              f"time: {end - start:.3f}; {0 if cum_time == 0 else cum_time / correctly_classified_images:.3f}; {cum_time:.3f}")

    for worker in itertools.chain(gpu_workers, cpu_workers):
        worker.send('terminate')

    results = {'netname': CalzoneConfig.netname,
               'dataset': CalzoneConfig.dataset,
               't': CalzoneConfig.t,
               'rep_num': CalzoneConfig.rep_num,
               'timeout': CalzoneConfig.timeout,
               'from_test': CalzoneConfig.from_test,
               'num_tests': CalzoneConfig.num_tests,
               'cumulative_time': cum_time,
               'images_results_by_index': image_results_by_image_index}

    print('analysis precision ', verified_images, '/ ', correctly_classified_images)
    logname = CalzoneConfig.logname if CalzoneConfig.logname is not None else time.strftime('%d%m%Y-%H:%M:%S')
    logfile = logname + '.json'
    with open(logfile, 'w') as fp:
        json.dump(results, fp)
    print(f'Results saved to {logfile}')
