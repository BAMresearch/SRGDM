# Super-Resolution for Gas Distribution Mapping

## Introduction
![Architecture of Network](images/architecture.png)
This repository contains the code for our paper **Super-Resolution for Gas Distribution Mapping**, in which we provide:

1. Gas Distribution Decoder (GDD): A CNN-based **method for spatiotemporal interpolation** of spatially sparse sensor measurements.
2. An extensive **dataset of synthetic gas distribution maps based on actual airflow measurements**. As generating ground truth maps is nearly impossible, this dataset provides a valuable resource for researchers in this field. It is available online, along with the code for our neural network model. 
3. A detailed comparative evaluation of GDD with state-of-the-art models on synthesized and real gas distribution data.

## Usage
GDD and the implementations of the state-of-the-art models can be found in the folder "models". Pre-trained GDD models are saved as PyTorch *.pth files. Model parameters can be found in the associated *.yaml file.

Our datasets can be found in the folder "data". Each file contains the gas distribution maps in a Tensor object and can be loaded with torch.load(*file*). Additionally, you can use the specified PyTorch dataset and PyTorch Lightning datamodule to conveniently load samples. Their usage can be found in the different Jupyter notebook files (*.ipynb).

## License
This software is released under the MIT license. See the [LICENSE](LICENSE.md) file for more details.

## Acknowledgements
This research was funded by BAM, SAF€RA (project RASEM) and JSPS (KAKENHI 474 Grant Number 22H04952 and 22K12124).

- The GMRF implementation is heavily based on the [MAPIRlab's implementation](https://github.com/MAPIRlab/gdm)
- The Kernel DM+V implementation is heavily based on the [Stephan Müller's implementation](https://gitlab.com/smueller18/TDKernelDMVW)

## Contact Information
Please contact us either via Github or via mro[at]bam.de

If you find this code useful, please cite our paper:
```
@inproceedings{winkler2022super,
  title={Super-Resolution for Gas Distribution Mapping: Convolutional Encoder-Decoder Network},
  author={Winkler, Nicolas P and Matsukura, Haruka and Neumann, Patrick P and Schaffernicht, Erik and Ishida, Hiroshi and Lilienthal, Achim J},
  booktitle={2022 IEEE International Symposium on Olfaction and Electronic Nose (ISOEN)},
  pages={1--3},
  year={2022},
  organization={IEEE}
}
```
