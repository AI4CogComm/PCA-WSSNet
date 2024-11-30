# PCA-WSSNet

Source code for the paper: 

Peihao Dong, Jibin Jia, Shen Gao, Fuhui Zhou and Qihui Wu, “Pruned convolutional attention network based wideband spectrum sensing with sub-Nyquist sampling,” IEEE Transactions on Vehicular Technology, Accepted for publication, 2024.

Please cite this paper when using the codes.

## Instructions

* The `generate_dataset.py` script is designed to generate datasets with varying occupancy rates. It includes the `MP()` function, which processes the generated signals using Multicoset sampling.
* The `CA_WSSNet.py` script defines the architecture of the neural network, specifying the layers, activation functions, and connections required to build the model.
* The `utils.py` script is a utility module that contains practical functions for data processing and performance evaluation. The `data_processing()` function splits raw data into training, validation, and test sets. The `performance_evaluation()` function provides commonly used metrics for model performance evaluation, such as detection probability, false alarm probability, and prediction accuracy.
* The `train_CA_WSSNet.py` script is responsible for training and testing the model.
* The `weight_pruning.py` script is utilized for model pruning, and it includes the `apply_pruning_to_CA_WSSNet()` function, which specifically prunes the designated layers of the model.
* The `pruned_transfer_learning.py` script uses samples from the target domain to fine-tune the pruned model that was initially trained on the source domain.
* The `direct_transfer_learning.py` script uses samples from the target domain to fine-tune the model that was initially trained on the source domain, without applying any pruning.

## Environment

* Python 3.7.16
* TensorFlow-gpu 2.7.0
