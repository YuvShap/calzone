import time
from multiprocessing.connection import Client
from random import shuffle, sample
import numpy as np
from calzone_utils import normalize


class LZeroGpuWorker:
    def __init__(self, port, network, means, stds, is_conv, dataset):
        self.__port = port
        self.__network = network
        self.__means = means
        self.__stds = stds
        self.__is_conv = is_conv
        self.__dataset = dataset
        if dataset == 'cifar10':
            self.__number_of_pixels = 1024
        else:
            self.__number_of_pixels = 784

    def work(self):
        address = ('localhost', self.__port)
        with Client(address) as conn:
            # Every iteration of this loop is one image
            message = conn.recv()
            while message != 'terminate':
                image, label, sampling_lower_bound, sampling_upper_bound, repetitions = message
                sampling_successes, sampling_time = self.__sample(image, label, sampling_lower_bound, sampling_upper_bound, repetitions)
                conn.send((sampling_successes, sampling_time))
                image, label, strategy, worker_index, number_of_workers = conn.recv()
                coverings = self.__load_coverings(strategy)
                self.__prove(conn, image, label, strategy, worker_index, number_of_workers, coverings)
                message = conn.recv()

    def __sample(self, image, label, sampling_lower_bound, sampling_upper_bound, repetitions):
        population = list(range(0, self.__number_of_pixels))
        sampling_successes = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        sampling_time = [0] * (sampling_upper_bound - sampling_lower_bound + 1)
        for size in range(sampling_lower_bound, sampling_upper_bound + 1):
            for i in range(0, repetitions):
                pixels = sample(population, size)
                start = time.time()
                verified = self.verify_group(image, label, pixels)
                duration = time.time() - start
                sampling_time[size - sampling_lower_bound] += duration
                if verified:
                    sampling_successes[size - sampling_lower_bound] += 1

        return sampling_successes, sampling_time

    def __load_coverings(self, strategy):
        t = strategy[-1]
        coverings = dict()
        for size, broken_size in zip(strategy, strategy[1:]):
            covering = []
            with open(f'coverings/({size},{broken_size},{t}).txt',
                      'r') as coverings_file:
                for line in coverings_file:
                    block = tuple(int(item) for item in line.split(','))
                    covering.append(block)
                coverings[size] = covering
        return coverings

    def __prove(self, conn, image, label, strategy, worker_index, number_of_workers, coverings):
        t = strategy[-1]
        with open(f'coverings/({self.__number_of_pixels},{strategy[0]},{t}).txt',
                  'r') as shared_covering:
            for line_number, line in enumerate(shared_covering):
                if conn.poll() and conn.recv() == 'stop':
                    conn.send('stopped')
                    return
                if line_number % number_of_workers == worker_index:
                    pixels = tuple(int(item) for item in line.split(','))
                    start = time.time()
                    verified = self.verify_group(image, label, pixels)
                    duration = time.time() - start
                    if verified:
                        conn.send((True, len(pixels), duration))
                    else:
                        conn.send((False, len(pixels), duration))
                        if len(pixels) not in coverings:
                            conn.send('adversarial-example-suspect')
                            conn.send(pixels)
                        else:
                            groups_to_verify = self.__break_failed_group(pixels, coverings[len(pixels)])
                            while len(groups_to_verify) > 0:
                                if conn.poll() and conn.recv() == 'stop':
                                    conn.send('stopped')
                                    return
                                group_to_verify = groups_to_verify.pop(0)
                                start = time.time()
                                verified = self.verify_group(image, label, group_to_verify)
                                duration = time.time() - start
                                if verified:
                                    conn.send((True, len(group_to_verify), duration))
                                else:
                                    conn.send((False, len(group_to_verify), duration))
                                    if len(group_to_verify) in coverings:
                                        groups_to_verify = self.__break_failed_group(group_to_verify, coverings[len(group_to_verify)]) + groups_to_verify
                                    else:
                                        conn.send('adversarial-example-suspect')
                                        conn.send(group_to_verify)
                    conn.send('next')
        conn.send("done")
        message = conn.recv()
        if message != 'stop':
            raise Exception('This should not happen')
        conn.send('stopped')

    def __break_failed_group(self, pixels, covering):
        permutation = list(pixels)
        shuffle(permutation)
        return [tuple(sorted(permutation[item] for item in block)) for block in covering]

    def verify_group(self, image, label, pixels_group):
        specLB = np.copy(image)
        specUB = np.copy(image)
        for pixel_index in self.get_indexes_from_pixels(pixels_group):
            specLB[pixel_index] = 0
            specUB[pixel_index] = 1
        normalize(specLB, self.__means, self.__stds, self.__dataset, 'gpupoly', self.__is_conv)
        normalize(specUB, self.__means, self.__stds, self.__dataset, 'gpupoly', self.__is_conv)

        return self.__network.test(specLB, specUB, label)

    def get_indexes_from_pixels(self, pixels_group):
        if self.__dataset != 'cifar10':
            return pixels_group
        indexes = []
        for pixel in pixels_group:
            indexes.append(pixel * 3)
            indexes.append(pixel * 3 + 1)
            indexes.append(pixel * 3 + 2)
        return indexes