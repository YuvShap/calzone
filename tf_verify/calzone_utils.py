import numpy as np


def normalize(image, means, stds, dataset, domain, is_conv):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds != None:
                image[i] /= stds[i]
    elif dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0]) / stds[0]
    elif (dataset == 'cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0]) / stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1]) / stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2]) / stds[2]
            count = count + 1

        is_gpupoly = (domain == 'gpupoly' or domain == 'refinegpupoly')
        if is_conv and not is_gpupoly:
            for i in range(3072):
                image[i] = tmp[i]
            # for i in range(1024):
            #    image[i*3] = tmp[i]
            #    image[i*3+1] = tmp[i+1024]
            #    image[i*3+2] = tmp[i+2048]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count + 1
                image[i + 1024] = tmp[count]
                count = count + 1
                image[i + 2048] = tmp[count]
                count = count + 1
