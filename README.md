# The color out of space: learning self-supervised representations for Earth Observation imagery
This repository contains the PyTorch code for the colorization phase explained in the paper:

**<a href="https://arxiv.org/abs/2006.12119">The color out of space: learning self-supervised representations for Earth Observation imagery</a>**  

## Model architecture
![Colorization & Multi-label classification - overview](/models/colorization_framework.png)

## Prerequisites
* Python >= 3.7
* PyTorch >= 1.5
* CUDA 10.0

## Dataset
We adopt the BigEarthNet Dataset. Refer to the README in the dataset folder for further information.

## Models
Different Encoder-Decoder combinations are available
- *Encoder ResNet18 - Decoder ResNet18*
- *Encoder ResNet50 - Decoder ResNet50*
- *Encoder ResNet50 - Decoder ResNet18*

## Training 
Before running the file ``main.py`` you can set the desired parameters in the file ``job_config.py``, which modify the ones contained in ``config/configuration.json``.

## Cite
If you have any questions, please contact [stefano.vincenzi@unimore.it](mailto:stefano.vincenzi@unimore.it), or open an issue on this repo. 

If you find this repository useful for your research, please cite the following paper:
```bibtex
  @article{vincenzi2020color,
  title={The color out of space: learning self-supervised representations for Earth                            Observation imagery},
  author={Vincenzi, Stefano and Porrello, Angelo and Buzzega, Pietro and Cipriano, Marco and Fronte,     Pietro and Cuccu, Roberto and Ippoliti, Carla and Conte, Annamaria and Calderara, Simone},
  journal={arXiv preprint arXiv:2006.12119},
  year={2020}
}
```
