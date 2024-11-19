import math
import time
import numpy as np
from itertools import chain
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def decreasing_exp(x, a, b):
    return 1 - np.exp(a*(x-b))


class LZeroRobustnessAnalyzer:
    def __init__(self, image_index, image, gpu_workers, cpu_workers, label, t, sampling, timeout, dataset):
        self.__image_index = image_index
        self.__image = image
        self.__label = label
        self.__gpu_workers = gpu_workers
        self.__cpu_workers = cpu_workers
        self.__t = t
        self.__sampling = sampling
        self.__timeout = timeout
        if dataset == 'cifar10':
            self.__number_of_pixels = 1024
        else:
            self.__number_of_pixels = 784

    def analyze(self):
        analyzer_start_time = time.time()
        print('*******************************************************')
        print(f'Starting to analyze image {self.__image_index}')

        print('Starting to estimate by sampling')
        estimation_start_time = time.time()
        p_vector, w_vector = self.__estimate_p_and_w()
        estimation_duration = time.time() - estimation_start_time
        print(f'Estimation took {estimation_duration:.3f}')

        print('Choosing strategy')
        choosing_strategy_start_time = time.time()
        covering_sizes, fraction_robust = self.__load_covering_sizes_and_approximate_fraction_robust(p_vector)
        strategy, T_k = self.__choose_strategy(covering_sizes, fraction_robust, w_vector)
        choosing_strategy_duration = time.time() - choosing_strategy_start_time
        estimated_verification_time = T_k[self.__number_of_pixels][0] / len(self.__gpu_workers)
        print(f'Chosen strategy is {strategy}, estimated verification time is {estimated_verification_time:.3f} sec')

        self.__release_workers(strategy)

        gpupoly_stats_by_size = {size: {'runs': 0, 'successes': 0, 'total_duration': 0} for size in strategy}
        waiting_adversarial_example_suspects = set()
        innocent_adversarial_example_suspects = set()

        results = {'verified': True, 'timed_out': False, 'p_vector': p_vector.tolist(), 'w_vector': w_vector.tolist(),
                   'estimation_duration': estimation_duration,
                   'fraction_robust': fraction_robust,
                   'T_k': T_k,
                   'strategy': strategy,
                   'choosing_strategy_duration': choosing_strategy_duration,
                   'estimated_verification_time': estimated_verification_time,
                   'gpupoly_stats_by_size': gpupoly_stats_by_size,
                   'time_waiting_for_milp_after_covering': 0}

        iterating_covering_start_time = time.time()
        timed_out, found_adversarial, adversarial_pixels, adversarial_example, adversarial_label = self.__wait_while_iterating_covering(
            analyzer_start_time, gpupoly_stats_by_size, waiting_adversarial_example_suspects,
            innocent_adversarial_example_suspects, covering_sizes[(self.__number_of_pixels, strategy[0])])
        iterating_covering_duration = time.time() - iterating_covering_start_time
        results['iterating_covering_duration'] = iterating_covering_duration

        if timed_out:
            results['verified'] = False
            results['timed_out'] = True
            results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
            results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
            return results

        if found_adversarial:
            results['verified'] = False
            results['adversarial_pixels'] = adversarial_pixels
            results['adversarial_example'] = adversarial_example
            results['adversarial_label'] = adversarial_label
            results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
            results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
            return results

        timed_out, found_adversarial, adversarial_pixels, adversarial_example, adversarial_label, waiting = self.__wait_for_cpu_workers_if_needed(
            analyzer_start_time,
            innocent_adversarial_example_suspects, waiting_adversarial_example_suspects)
        results['time_waiting_for_milp_after_covering'] = waiting

        if timed_out:
            results['verified'] = False
            results['timed_out'] = True
            results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
            results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
            return results

        if found_adversarial:
            results['verified'] = False
            results['adversarial_pixels'] = adversarial_pixels
            results['adversarial_example'] = adversarial_example
            results['adversarial_label'] = adversarial_label
            results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
            results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
            return results

        print('\nVerified!')
        print('*******************************************************')
        results['proved_with_milp'] = len(innocent_adversarial_example_suspects)
        results['groups_waiting_for_milp'] = len(waiting_adversarial_example_suspects)
        return results

    def __estimate_p_and_w(self):
        sampling_lower_bound = self.__t
        sampling_upper_bound = 99
        repetitions = math.ceil(self.__sampling / len(self.__gpu_workers))
        for gpu_worker in self.__gpu_workers:
            gpu_worker.send((self.__image, self.__label, sampling_lower_bound,
                             sampling_upper_bound, repetitions))

        sampling_successes_vector = np.zeros(sampling_upper_bound - sampling_lower_bound + 1)
        sampling_time_vector = np.zeros(sampling_upper_bound - sampling_lower_bound + 1)
        for gpu_worker in self.__gpu_workers:
            sampling_successes, sampling_time = gpu_worker.recv()
            sampling_successes_vector += np.array(sampling_successes)
            sampling_time_vector += np.array(sampling_time)

        success_ratio_vector = sampling_successes_vector / (repetitions * len(self.__gpu_workers))
        average_time_vector = sampling_time_vector / (repetitions * len(self.__gpu_workers))

        p_vector = savgol_filter(success_ratio_vector, 15, 2)
        w_vector = np.copy(average_time_vector)

        indexes = sorted([index for index, sample in enumerate(success_ratio_vector) if sample < 0.97])
        stop_index = min(indexes[0] + 1, len(success_ratio_vector)) if len(indexes) > 0 else len(success_ratio_vector)
        if stop_index > 1:
            popt, pcov = curve_fit(decreasing_exp, range(sampling_lower_bound, sampling_lower_bound + stop_index), p_vector[:stop_index], maxfev=5000)
            print(f'a={popt[0]}, b={popt[1]}')
            p_vector[:stop_index] = decreasing_exp(np.asarray(range(sampling_lower_bound, sampling_lower_bound + stop_index)), *popt)

        return p_vector, w_vector

    def __load_covering_sizes_and_approximate_fraction_robust(self, p_vector):
        fraction_robust = dict()
        covering_table_file = np.genfromtxt(f'coverings/{self.__t}-table.csv', delimiter=',')
        covering_sizes = dict()
        for v in range(self.__t, 100):
            fraction_robust[v] = dict()
            for k in range(self.__t, v):
                estimated_value = (p_vector[k - self.__t] - p_vector[v - self.__t]) / (1 - p_vector[v - self.__t])
                fraction_robust[v][k] = min(1, max(estimated_value, 0))
                covering_sizes[(v, k)] = covering_table_file[v - self.__t + 1][k - self.__t + 1]
        v = self.__number_of_pixels
        fraction_robust[v] = dict()
        for k in range(self.__t, 100):
            fraction_robust[v][k] = min(1, max(p_vector[k - self.__t], 0))
            if v == 784:
                covering_sizes[(v, k)] = covering_table_file[100 - self.__t + 1][k - self.__t + 1]
            else:
                covering_sizes[(v, k)] = covering_table_file[101 - self.__t + 1][k - self.__t + 1]

        covering_sizes = {key: value for key, value in covering_sizes.items() if value < 10 * math.pow(10, 6)}

        return covering_sizes, fraction_robust

    def __choose_strategy(self, covering_sizes, fraction_robust, w_vector):
        T_k = dict()
        T_k[self.__t] = (0, None)
        for v in chain(range(self.__t + 1, 100), [self.__number_of_pixels]):
            best_k = None
            best_k_value = None
            for k in range(self.__t, min(v, 100)):
                if (v, k) not in covering_sizes:
                    continue
                k_value = covering_sizes[(v, k)] * (w_vector[k - self.__t] + (1 - fraction_robust[v][k]) * T_k[k][0])
                if best_k_value is None or k_value < best_k_value:
                    best_k = k
                    best_k_value = k_value
            T_k[v] = (best_k_value, best_k)
        strategy = []
        move_to = T_k[self.__number_of_pixels][1]
        while move_to is not None:
            strategy.append(move_to)
            move_to = T_k[move_to][1]
        return strategy, T_k

    def __release_workers(self, strategy):
        for worker_index, gpu_worker in enumerate(self.__gpu_workers):
            gpu_worker.send((self.__image, self.__label, strategy, worker_index, len(self.__gpu_workers)))
        for cpu_worker in self.__cpu_workers:
            cpu_worker.send((self.__image, self.__label))

    def __wait_while_iterating_covering(self, analyzer_start_time, statistics_by_size,
                                        waiting_adversarial_example_suspects,
                                        innocent_adversarial_example_suspects, initial_covering_size):
        number_of_groups_finished_from_initial_covering = 0
        next_cpu_worker = 0
        done_count = 0
        last_print_time = time.time()
        while done_count < len(self.__gpu_workers):
            if time.time() - analyzer_start_time >= self.__timeout:
                print('\nTimed out!')
                print('*******************************************************')
                self.__stop_workers()
                return True, False, None, None, None
            last_print_time = self.__print_progress(statistics_by_size, initial_covering_size,
                                                    innocent_adversarial_example_suspects,
                                                    last_print_time, number_of_groups_finished_from_initial_covering,
                                                    waiting_adversarial_example_suspects)

            found_adversarial, pixels, adversarial_example, adversarial_label = self.__pool_cpu_workers_messages(
                innocent_adversarial_example_suspects, waiting_adversarial_example_suspects)

            if found_adversarial:
                return False, found_adversarial, pixels, adversarial_example, adversarial_label

            for gpu_worker in self.__gpu_workers:
                if gpu_worker.poll():
                    message = gpu_worker.recv()
                    if message == 'adversarial-example-suspect':
                        adversarial_example_suspect = gpu_worker.recv()
                        if adversarial_example_suspect not in innocent_adversarial_example_suspects \
                                and adversarial_example_suspect not in waiting_adversarial_example_suspects:
                            waiting_adversarial_example_suspects.add(adversarial_example_suspect)
                            cpu_worker = self.__cpu_workers[next_cpu_worker]
                            cpu_worker.send(adversarial_example_suspect)
                            next_cpu_worker = (next_cpu_worker + 1) % len(self.__cpu_workers)

                    elif message == 'done':
                        done_count += 1

                    elif message == 'next':
                        number_of_groups_finished_from_initial_covering += 1

                    else:
                        verified, number_of_pixels, duration = message
                        size_statistics = statistics_by_size[number_of_pixels]
                        size_statistics['runs'] += 1
                        size_statistics['total_duration'] += duration
                        if verified:
                            size_statistics['successes'] += 1
        return False, False, None, None, None

    def __print_progress(self, gpu_statistics_by_size, initial_covering_size, innocent_adversarial_example_suspects,
                         last_print_time, number_of_groups_finished_from_initial_covering,
                         waiting_adversarial_example_suspects):
        current = time.time()
        if current - last_print_time < 2:
            return last_print_time
        initial_covering_string = f'progress: ' \
                                  f'{number_of_groups_finished_from_initial_covering}/{int(initial_covering_size)}=' \
                                  f'{100 * number_of_groups_finished_from_initial_covering / initial_covering_size:.3f}%'
        sizes_and_stats = ((size, size_stat['runs'], size_stat['successes'], size_stat['total_duration']) for
                           (size, size_stat) in sorted(gpu_statistics_by_size.items(), reverse=True))
        sizes_string = 'gpupoly: ' + '. '.join(
            f'{succ}/{runs}={100 * succ / runs:.3f}%, {1000 * duration / runs:.1f} ms' if runs != 0 else f'NONE'
            for (size, runs, succ, duration) in sizes_and_stats)
        MILP_string = f'MILP: {len(innocent_adversarial_example_suspects)} verified, ' \
                      f'{len(waiting_adversarial_example_suspects)} waiting'
        print('\r' + initial_covering_string, sizes_string, MILP_string, sep='; ', end='')
        return current

    def __pool_cpu_workers_messages(self, innocent_adversarial_example_suspects, waiting_adversarial_example_suspects):
        for cpu_worker in self.__cpu_workers:
            if cpu_worker.poll():
                pixels, verified, adversarial_example, adversarial_label, timeout = cpu_worker.recv()
                if verified:
                    waiting_adversarial_example_suspects.remove(pixels)
                    innocent_adversarial_example_suspects.add(pixels)
                elif not timeout:
                    self.__handle_adversarial(pixels, adversarial_example, adversarial_label)
                    return True, pixels, adversarial_example, adversarial_label
        return False, None, None, None

    def __handle_adversarial(self, pixels, adversarial_example, adversarial_label):
        self.__stop_workers()

        print(f'\nAdversarial example found, changing label "{self.__label}" to "{adversarial_label}" '
              f'while perturbing pixels: {pixels}')
        print(adversarial_example)
        print('*******************************************************')

    def __wait_for_cpu_workers_if_needed(self, analyzer_start_time, innocent_adversarial_example_suspects,
                                         waiting_adversarial_example_suspects):
        waiting = 0
        if len(waiting_adversarial_example_suspects) > 0:
            start_time = time.time()
            while len(waiting_adversarial_example_suspects) > 0:
                if time.time() - analyzer_start_time >= self.__timeout:
                    print('\nTimed out!')
                    print('*******************************************************')
                    self.__stop_workers()
                    return True, False, None, None, None, time.time() - start_time

                found_adversarial, pixels, adversarial_example, adversarial_label = self.__pool_cpu_workers_messages(
                    innocent_adversarial_example_suspects, waiting_adversarial_example_suspects)

                if found_adversarial:
                    return False, found_adversarial, pixels, adversarial_example, adversarial_label, time.time() - start_time
            waiting = time.time() - start_time

        self.__stop_workers()
        return False, False, None, None, None, waiting

    def __stop_workers(self):
        for worker in chain(self.__gpu_workers, self.__cpu_workers):
            worker.send('stop')
        for worker in chain(self.__gpu_workers, self.__cpu_workers):
            message = worker.recv()
            while message != 'stopped':
                message = worker.recv()
