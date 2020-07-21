## BioS2Net (Biological Sequence and Structure Network)

![architecture](https://github.com/tryptofanik/bios2net/blob/master/doc/architecture.jpg)

### Introduction
This repository contains the source code of BioS2Net (Biological Sequence and Structure Network), a novel deep neural network architecture that extracts both sequential and structural information of biomolecules. Our architecture consists of four main parts:

1. sequence convolutional extractor,

2. 3D structure extractor,

3. 3D-structure-aware sequence temporal network,

4. fusion classification network.

Overall workflow of BioS2Net is as follows. At first, the sequence extractor assigns to each of the input point features representing its sequential context. Then enriched point cloud is directed to 3D structure extractor, which is meant to learn the context of the structural information of a given biomolecule. Then, from one of the layers of it, the 3D-structureaware
sequence temporal convolutional network (which inputs comprise of both sequential and structural information learnt by both extractors) is created. Finally, at the end of the learning, all branches are merged to produce a global feature vector from which classification of proteins
starts. This vector may be further used to represent protein in a fixed and compressed way. It is worth to emphasise that each point in the input of temporal network is aware of its structural and sequential surrounding, thus significantly leveraging learning capabilities of BioS2Net. Importantly, our architecture can be easily extended to face regression or semantic segmentation problems in different fields of natural sciences.

### Installation

For running BioS2Net you will need Tensorflow 2.0 installed on your computer. Our experiment has been performed on TF 2.0., Python 3.6.9, CUDA 10.0., gcc 7.4, Ubuntu 18.04 LTS.



#### Compile Customized TF Operators
The TF operators are included under `tf_ops` and you need to compile them.

Update path to the CUDA within `tf_ops/compile_ops.sh` file. Then run the file. It shoud compile files and produce .so files in each subdirectory.

### Usage

#### Classification

To train BioS2Net you have to at first create a dataset with point clouds on which training should be run.
Dataset directory should be in data directory. Each group to which you want to classify must be in separate subdirectory and it should be splitted into train and test subsubdirectory within which examples should be stored. As for now, current dataset manager accept only `.npy` files, but it can be easily changed and adopted to user needs.

To run training BioS2Net model you have to run:

        python train_bios2net.py --dataset_path data data/eDD

If any help is needed with selection of parameters, like number of points in point cloud, or learning rate just type:


        python train_bios2net.py -h


You can run BioS2Net with [wandb](https://www.wandb.com/) to see real-time visualization of progess of the model.
