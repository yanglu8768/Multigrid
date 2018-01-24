# Multigrid

This code is an tensorflow based implementation for the paper [Learning Multi-grid Generative ConvNets by Minimal Contrastive Divergence](https://arxiv.org/abs/1709.08868)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

What things you need to install the software and how to install them
tensorflow >= 1.2
Plase refer to the tensorflow [install instruction](https://www.tensorflow.org/install/)

skimage
pprint
scipy
sklearn


## Run the code

(1) Copy the training data into ./data/dataset_name
(2) Change the input parameter in the main.py at line 9.

```
flags.DEFINE_string('dataset_name', 'new', 'Folder name which stored in ./data/dataset_name')
```

(3) Run the code
```
python main.py
```



## Authors

* **Yang Lu** 
* **Ruiqi Gao**


