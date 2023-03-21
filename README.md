# Super-Resolution for Gas Distribution Mapping

![Architecture of Network](images/architecture.png)

This repository contains the code for our paper **Super-Resolution for Gas Distribution Mapping**, in which we provide:

1. Gas Distribution Decoder (GDD): A CNN-based **method for spatiotemporal interpolation** of spatially sparse sensor measurements.
2.  An extensive **dataset of synthetic gas distribution maps based on actual airflow measurements**. As generating ground truth maps is nearly impossible, this dataset provides a valuable resource for researchers in this field. It is available online, along with the code for our neural network model. 
3.  A detailed comparative evaluation of GDD with state-of-the-art models on synthesized and real gas distribution data.

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

