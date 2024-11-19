Calzone
========
Calzone (<strong>c</strong>ertification <strong>a</strong>nalyzer for <strong>L</strong>-<strong>z</strong>ero <strong>ne</strong>ighborhoods) is a sound and complete L0 robustness verifier for neural networks.
To the best of our knowledge, Calzone is the first deterministic L0 robustness verifier. For more information, refer to the paper [Deep Learning Robustness Verification for Few-Pixel Attacks [OOPSLA'23]](https://dl.acm.org/doi/abs/10.1145/3586042). <br/>

Overview
========
Calzone is implemented as a module of [ERAN](https://github.com/eth-sri/ERAN). The files and folder associated with Calzone start with the prefix 'calzone' and can be found in the `tf_verify` directory.<br/>

Requirements
========
Calzone's requirements are similar to ERAN. Note that, Calzone rely on GPUPoly, therefore to run Calzone a GPU is required.

Installation
------------
<strong>Clone Calzone:</strong><br />
```
git clone https://github.com/YuvShap/calzone.git
cd calzone
```

<strong>Install ERAN's dependencies:</strong><br />
Follow [Eran's installation instructions](https://github.com/eth-sri/eran?tab=readme-ov-file#installation) (skip `git clone https://github.com/eth-sri/ERAN.git` and `cd ERAN`, make sure ELINA and Gurobi are installed and obtain an academic license for Gurobi).

<strong>Setup coverings database:</strong>
1. Create a subdirectory named `coverings` inside `tf_verify`.
2. Download all the [covering files](https://technionmail-my.sharepoint.com/:f:/g/personal/ece_safe_technion_ac_il/EmuSjTmzfNxNgdaMBPzaibUBDpODm5nejYJw-zA-Wr_rzA?e=brKPHx) into `coverings`.
3. Unzip the downloaded zip files.
* As described in our paper, the covering database is built on the [La Jolla Covering Repository Tables](https://ljcr.dmgordon.org/cover/table.html).

<strong>Setup datasets:</strong><br/>
100 test samples of MNIST and CIFAR-10 datasets are provided by ERAN.<br/> 
To run Calzone on Fashion MNIST, download fashion-mnist_test.csv from [here](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and copy it to `data` directory.

Usage
-------------
1. Change dir: `cd tf_verify`.
2. Run Calzone.

Examples:
````
python3 calzone.py --netname calzone_models/FASHION_ConvSmallPGD.onnx --dataset fashion --t 3 --timeout 1800  --gpu_num 8 --milp_num 5
python3 calzone.py --netname calzone_models/MNIST_ConvMedPGD.onnx --dataset mnist --t 4 --timeout 3600 --gpu_num 8 --milp_num 5 
python3 calzone.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 5 --timeout 18000 --gpu_num 8 --milp_num 5  
````

**Calzone supports the following parameters:**

* `netname`: the network name, the extension must be .onnx (required).<br/>
* `dataset`: the dataset, can be either mnist, fashion or cifar10 (required).<br/>
* `t`: the maximal number of perturbed pixels, can be either 1,2,3,4 or 5 (default is 3).<br/>
* `timeout`: the analysis timeout in seconds for a single image (default is 1800).<br/>
* `rep_num`: the number of sampled subsets of pixels for each size k (default is 400).<br/>
* `gpu_num`: the number of GPUs to be used in the analysis (default is 8).<br/>
* `milp_num`: the number of MILP verifier instances to use (default is 5).<br/>
* `num_tests`: the number of images to test (default is 100).<br/>
* `from_test`: the index to start testing from within the test set (default is 0).<br/>
* `logname`: the name of the log file (a .json extension will be added), if not specified, a timestamp will be used.<br/>
* `mean`: the mean used to normalize the data. Must be one value for mnist/fashion (e.g --mean 0.5) and three values for cifar10 (e.g --mean 0.4914 0.4822 0.4465).<br/> If normalization is extracted from the network, this argument will be ignored. If not specified, default values will be used.
* `std`: the standard deviation used to normalize the data. Must be one value for mnist/fashion (e.g --std 0.5) and three values for cifar10 (e.g --std 0.2470 0.2435 0.2616).<br/> If normalization is extracted from the network, this argument will be ignored. If not specified, default values will be used.

Note:
* Sampling is distributed across `gpu_num` GPUs, so in case `rep_num` is not divisible by `gpu_num` the number of samples for each size will be `gpu_num * ceil(rep_num/gpu_num)`.
* The timeout of each MILP verification task in the analysis is the global timeout for a single image (specified by `timeout`). In case of a timeout or a detected adversarial example, Calzone sends a stop message to the MILP verifier. The MILP verifier stops only after completing its running tasks or reaching their timeout. This might sometimes delay the overall termination. In our evaluated networks, MILP tasks typically complete quickly, so it is generally unnoticeable.    


Networks and Experiments
-------------
The networks evaluated in the paper can be found in the directory `tf_verify/calzone_models`.<br/> 
Their architectures are adopted from the networks in ERAN repository, we trained all of them except for CIFAR_ConvSmallPGD.onnx which is provided by ERAN's repository [here](https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convSmallRELU__PGDK.onnx). <br/>
There is no need to specify `mean` and `std` when running these networks, normalization is either extracted from the onnx file or the default values.<br/>
We provide the running configurations for the experiments described in Table 2 in the paper: 

| Dataset  | Network | t  | Configuration |
| ------------- | ------------- | ------------- | ------------- |
| MNIST  | 6x200_PGD  | 1 | `python3 calzone.py --netname calzone_models/MNIST_6x200_PGD.onnx --dataset mnist --t 1 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-6x200-t-1` |
| | | 2 | `python3 calzone.py --netname calzone_models/MNIST_6x200_PGD.onnx --dataset mnist --t 2 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-6x200-t-2` |
| | | 3 | `python3 calzone.py --netname calzone_models/MNIST_6x200_PGD.onnx --dataset mnist --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-6x200-t-3`|
| | | 4 | `python3 calzone.py --netname calzone_models/MNIST_6x200_PGD.onnx --dataset mnist --t 4 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname mnist-6x200-t-4`|
| | ConvSmall  | 1 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmall.onnx --dataset mnist --t 1 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvSmall-t-1` |
| | | 2 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmall.onnx --dataset mnist --t 2 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvSmall-t-2`|
| | | 3 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmall.onnx --dataset mnist --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvSmall-t-3`|
| | | 4 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmall.onnx --dataset mnist --t 4 --timeout 3600  --gpu_num 8 --milp_num 5 --num_tests 50 --logname mnist-ConvSmall-t-4`|
| | ConvSmallPGD  | 1 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 1 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvSmallPGD-t-1`|
| | | 2 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 2 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvSmallPGD-t-2`|
| | | 3 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvSmallPGD-t-3`|
| | | 4 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 4 --timeout 3600  --gpu_num 8 --milp_num 5 --num_tests 50 --logname mnist-ConvSmallPGD-t-4`|
| | | 5 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset mnist --t 5 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname mnist-ConvSmallPGD-t-5`|
| | ConvMedPGD  | 1 | `python3 calzone.py --netname calzone_models/MNIST_ConvMedPGD.onnx --dataset mnist --t 1 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvMedPGD-t-1`|
| | | 2 | `python3 calzone.py --netname calzone_models/MNIST_ConvMedPGD.onnx --dataset mnist --t 2 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvMedPGD-t-2`|
| | | 3 | `python3 calzone.py --netname calzone_models/MNIST_ConvMedPGD.onnx --dataset mnist --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvMedPGD-t-3`|
| | | 4 | `python3 calzone.py --netname calzone_models/MNIST_ConvMedPGD.onnx --dataset mnist --t 4 --timeout 3600  --gpu_num 8 --milp_num 5 --num_tests 50 --logname mnist-ConvMedPGD-t-4`|
| | | 5 | `python3 calzone.py --netname calzone_models/MNIST_ConvMedPGD.onnx --dataset mnist --t 5 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname mnist-ConvMedPGD-t-5`|
| | ConvBig  | 1 | `python3 calzone.py --netname calzone_models/MNIST_ConvBig.onnx --dataset mnist --t 1 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvBig-t-1`|
| | | 2 | `python3 calzone.py --netname calzone_models/MNIST_ConvBig.onnx --dataset mnist --t 2 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname mnist-ConvBig-t-2`|
| | | 3 | `python3 calzone.py --netname calzone_models/MNIST_ConvBig.onnx --dataset mnist --t 3 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname mnist-ConvBig-t-3`|
| F-MNIST | ConvSmallPGD  | 1 | `python3 calzone.py --netname calzone_models/MNIST_ConvSmallPGD.onnx --dataset fashion --t 1 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname fashion-ConvSmallPGD-t-1`|
| | | 2 | `python3 calzone.py --netname calzone_models/FASHION_ConvSmallPGD.onnx --dataset fashion --t 2 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname fashion-ConvSmallPGD-t-2`|
| | | 3 | `python3 calzone.py --netname calzone_models/FASHION_ConvSmallPGD.onnx --dataset fashion --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname fashion-ConvSmallPGD-t-3`|
| | | 4 | `python3 calzone.py --netname calzone_models/FASHION_ConvSmallPGD.onnx --dataset fashion --t 4 --timeout 3600  --gpu_num 8 --milp_num 5 --num_tests 50 --logname fashion-ConvSmallPGD-t-4`|
| | ConvMedPGD  | 1 | `python3 calzone.py --netname calzone_models/FASHION_ConvMedPGD.onnx --dataset fashion --t 1 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname fashion-ConvMedPGD-t-1`|
| | | 2 | `python3 calzone.py --netname calzone_models/FASHION_ConvMedPGD.onnx --dataset fashion --t 2 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname fashion-ConvMedPGD-t-2`|
| | | 3 | `python3 calzone.py --netname calzone_models/FASHION_ConvMedPGD.onnx --dataset fashion --t 3 --timeout 1800  --gpu_num 8 --milp_num 5 --num_tests 100 --logname fashion-ConvMedPGD-t-3`|
| | | 4 | `python3 calzone.py --netname calzone_models/FASHION_ConvMedPGD.onnx --dataset fashion --t 4 --timeout 18000  --gpu_num 8 --milp_num 5 --num_tests 10 --logname fashion-ConvMedPGD-t-4`|
| CIFAR-10  | ConvSmallPGD  | 1 |`python3 calzone.py --netname calzone_models/CIFAR_ConvSmallPGD.onnx --dataset cifar10 --t 1 --timeout 1800  --gpu_num 8 --milp_num 50 --num_tests 100 --logname cifar-ConvSmallPGD-t-1` |
| | | 2 | `python3 calzone.py --netname calzone_models/CIFAR_ConvSmallPGD.onnx --dataset cifar10 --t 2 --timeout 1800  --gpu_num 8 --milp_num 50 --num_tests 100 --logname cifar-ConvSmallPGD-t-2`|
| | | 3 | `python3 calzone.py --netname calzone_models/CIFAR_ConvSmallPGD.onnx --dataset cifar10 --t 3 --timeout 18000  --gpu_num 8 --milp_num 50 --num_tests 10 --logname cifar-ConvSmallPGD-t-3`|