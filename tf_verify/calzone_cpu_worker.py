import contextlib
from os import devnull
from multiprocessing.connection import Client
from constraint_utils import get_constraints_for_dominant_label
from ai_milp import verify_network_with_milp
import numpy as np
from calzone_utils import normalize
from config import config


class LZeroCpuWorker:
    def __init__(self, port, eran, means, stds, is_conv, dataset):
        self.__port = port
        self.__eran = eran
        self.__means = means
        self.__stds = stds
        self.__is_conv = is_conv
        self.__dataset = dataset

    def work(self):
        address = ('localhost', self.__port)
        with Client(address) as conn:
            # Every iteration of this loop is one image
            message = conn.recv()
            while message != 'terminate':
                image, label = message
                self.__handle_image(conn, image, label)
                conn.send('stopped')
                message = conn.recv()

    def __handle_image(self, conn, image, label):
        jobs = []
        while True:
            while conn.poll() or len(jobs) == 0:
                message = conn.recv()
                if message != 'stop':
                    jobs.append(message)
                else:
                    return
            pixels = jobs.pop(0)
            with contextlib.redirect_stdout(open(devnull, 'w')):
                verified, adv_image, adv_label, timeout = self.verify_group(image, label, pixels)
            conn.send((pixels, verified, adv_image, adv_label, timeout))

    def verify_group(self, image, label, pixels_group):
        specLB = np.copy(image)
        specUB = np.copy(image)
        for pixel_index in self.get_indexes_from_pixels(pixels_group):
            specLB[pixel_index] = 0
            specUB[pixel_index] = 1
        normalize(specLB, self.__means, self.__stds, self.__dataset, 'deeppoly', self.__is_conv)
        normalize(specUB, self.__means, self.__stds, self.__dataset, 'deeppoly', self.__is_conv)

        prop = -1
        retry = False
        try:
            verified, adv_image, adv_label, timeout = self.use_milp(specLB, specUB, label, prop)
        except:
            # Probably Gurobi numerical errors, try to relax deeppoly bounds
            retry = True

        if retry:
            verified, adv_image, adv_label, timeout = self.use_milp(specLB, specUB, label, prop, relax_bounds=True)

        if verified or timeout:
            return verified, adv_image, adv_label, timeout

        real_adv_image = np.copy(image)
        for index in self.get_indexes_from_pixels(pixels_group):
            real_adv_image[index] = adv_image[index]
        spec_LB_adv = np.copy(real_adv_image)
        spec_UB_adv = np.copy(real_adv_image)
        normalize(spec_LB_adv, self.__means, self.__stds, self.__dataset, 'deeppoly', self.__is_conv)
        normalize(spec_UB_adv, self.__means, self.__stds, self.__dataset, 'deeppoly', self.__is_conv)
        perturbed_label, nn, nlb, nub, failed_labels, x = self.__eran.analyze_box(spec_LB_adv, spec_UB_adv, "deeppoly",
                                                                                  config.timeout_lp,
                                                                                  config.timeout_milp,
                                                                                  config.use_default_heuristic,
                                                                                  label=adv_label, prop=prop, K=0, s=0,
                                                                                  timeout_final_lp=config.timeout_final_lp,
                                                                                  timeout_final_milp=config.timeout_final_milp,
                                                                                  use_milp=False,
                                                                                  complete=False,
                                                                                  terminate_on_failure=not config.complete,
                                                                                  partial_milp=0,
                                                                                  max_milp_neurons=0,
                                                                                  approx_k=0)
        assert perturbed_label == adv_label, 'Adv example problem, probably due to numerical difficulties in normalization and denormalization'
        return verified, real_adv_image.tolist(), adv_label, False


    def use_milp(self, specLB, specUB, label, prop, relax_bounds=False):
        perturbed_label, nn, nlb, nub, failed_labels, x = self.__eran.analyze_box(specLB, specUB, "deeppoly",
                                                                                      config.timeout_lp,
                                                                                      config.timeout_milp,
                                                                                      config.use_default_heuristic,
                                                                                      label=label, prop=prop, K=0, s=0,
                                                                                      timeout_final_lp=config.timeout_final_lp,
                                                                                      timeout_final_milp=config.timeout_final_milp,
                                                                                      use_milp=False,
                                                                                      complete=False,
                                                                                      terminate_on_failure=False,
                                                                                      partial_milp=0,
                                                                                      max_milp_neurons=0,
                                                                                      approx_k=0)

        if relax_bounds:
            for i in range(len(nlb)):
                for j in range(len(nlb[i])):
                    if nub[i][j] - nlb[i][j] < 1e-2:
                        nlb[i][j] -= 1e-2
                        nub[i][j] += 1e-2


        if (perturbed_label == label):
            # Should not happen
            return True, None, None, False

        if failed_labels is not None:
            failed_labels = list(set(failed_labels))
            constraints = get_constraints_for_dominant_label(label, failed_labels)
            verified_flag, adv_image, adv_val = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
            if (verified_flag == True):
                return True, None, None, False
            else:
                if adv_image != None:
                    cex_label, _, _, _, _, _ = self.__eran.analyze_box(adv_image[0], adv_image[0], 'deepzono',
                                                                config.timeout_lp, config.timeout_milp,
                                                                config.use_default_heuristic, approx_k=config.approx_k)
                    if (cex_label != label):
                        self.denormalize(adv_image[0], self.__means, self.__stds, self.__dataset)
                        return False, adv_image[0], cex_label, False
                    else:
                        assert False, 'This should not happen'
                else:
                    # Timeout
                    return False, None, None, True

    def denormalize(self, image, means, stds, dataset):
        if dataset == 'mnist' or dataset == 'fashion':
            for i in range(len(image)):
                image[i] = image[i] * stds[0] + means[0]
        elif (dataset == 'cifar10'):
            count = 0
            tmp = np.zeros(3072)
            for i in range(1024):
                tmp[count] = image[count] * stds[0] + means[0]
                count = count + 1
                tmp[count] = image[count] * stds[1] + means[1]
                count = count + 1
                tmp[count] = image[count] * stds[2] + means[2]
                count = count + 1

            domain = 'deeppoly'
            is_gpupoly = (domain == 'gpupoly' or domain == 'refinegpupoly')
            if self.__is_conv and not is_gpupoly:
                for i in range(3072):
                    image[i] = tmp[i]
            else:
                count = 0
                for i in range(1024):
                    image[i] = tmp[count]
                    count = count + 1
                    image[i + 1024] = tmp[count]
                    count = count + 1
                    image[i + 2048] = tmp[count]
                    count = count + 1

    def get_indexes_from_pixels(self, pixels_group):
        if self.__dataset != 'cifar10':
            return pixels_group
        indexes = []
        for pixel in pixels_group:
            indexes.append(pixel * 3)
            indexes.append(pixel * 3 + 1)
            indexes.append(pixel * 3 + 2)
        return indexes

