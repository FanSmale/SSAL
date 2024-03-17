# SSAL
The proposed self-supervised active learning approach.

## Usage 
Please create  `models/CIFAR10`, `models/CIFAR100`, `models/FashionMNIST`, `models/SVHN`, `models/TinyImageNet` folders to store the training model and the middle results.
The middle results consist of `queried sample idx`, `rho`, `delta`, `features`, `logits`, `The number of samples of each class`, `The number of queried samples of each class` in each AL round.

Please Run `train.py` or `run.sh` in each folder.

The code for sample selection will be uploaded soon.

Model Structure:
![alt text](https://github.com/FanSmale/SSAL/blob/main/image.png)
This is Our SSAL framework. In the deep training phase, two linear classifiers are trained simultaneously to classify the image rotation degree and the ground truth labels, whilst optimizing the representation learner. In the AL phase, three key values are primarily calculated for the representation of each sample: the classification uncertainty measurement, i.e., Entropy, the local density $\gamma$, and the minimum distance $\delta$. The sample information $I_{umf}$ (Eq. (7)) is measured by fusing these values. To maintain class balance, the samples are grouped by category and sorted in descending order based on the information. Next, the theoretical optimal number of samples to be queried for each category is determined. Both stages are interdependent iterative processes. 


Code Structure:
1. Five folders `CIFAR10, CIFAR100,FashionMNIST,SVHN,TinyImageNet` are the code for running on different benchmarks.
2. `dataloader.py` is the data loading approach with learning image rotation.
3. `resnet.py` is the popular resnet structure.
4. `ssal.py` is the core sample selection approach, which will be uploaded soon, or contact us for acquiring ``minfan@swpu.edu.cn;yanxuewu2023@163.com''
