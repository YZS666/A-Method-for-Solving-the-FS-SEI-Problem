# Few-Shot Specific Emitter Identification Using Asymmetric Masked Auto-Encoder

<p align="center">
  <img src="https://github.com/YZS666/A-Method-for-Solving-the-FS-SEI-Problem/tree/main/Visualization/AMAE_FS_SEI.jpg?raw=true" width="480">
</p>

This is a PyTorch/GPU implementation of the paper [Few-Shot Specific Emitter Identification Using Asymmetric Masked Auto-Encoder](https://ieeexplore.ieee.org/document/10243409). If using relevant content, please cite this paper:
```
@article{yao2023few,
  title={Few-Shot Specific Emitter Identification Using Asymmetric Masked Auto-Encoder},
  author={Yao, Zhisheng and Fu, Xue and Guo, Lantu and Wang, Yu and Lin, Yun and Shi, Shengnan and Gui, Guan},
  journal={IEEE Communications Letters},
  year={2023},
  publisher={IEEE}
}
```

* Attention, you need to manually create three folders and name them as model_weight, test_result and Visualization.
* In addition, the dataset link used in this demo is https://ieee-dataport.org/open-access/lorarffidataset.

### Catalog

- [x] Pre-training and Visualization code
- [x] Fine-tuning code

### Pre-training
* Start pre-training by running the pretrain.py file.
* After the pre-training is completed, the Visualization.py file can be run to visualize the features of the pre-trained model, and the pre-trained model can be evaluated using unsupervised clustering indicators.

### Fine-tuning with pre-trained checkpoints
* Start fine-tuning by running the fine-tuning.py file.

# New result （Experimental results under power normalization on each dataset）

### Clustering performance of pre-training based on AMAE
*  Semantic feature visualization after pre-training on LoRa dataset with 30 categories (left: feature visualization of AMAE on LoRa dataset; right: feature visualization of AMAE on WiFi dataset)
![image](https://github.com/YZS666/A-Method-for-Solving-the-FS-SEI-Problem/blob/main/Visualization/t-SNE_LoRa.jpg)
![image](https://github.com/YZS666/A-Method-for-Solving-the-FS-SEI-Problem/blob/main/Visualization/t-SNE_WiFi.jpg)
*  Cluster performance indicators of silhouette coefficient (SC), clustering accuracy (AC), normalized mutual information (NMI) and adjusting mutual information (AMI)
|	Dataset	|	LoRa	|	WiFi	|
|	----	|	----	|	----	|
|	SC	|	-0.0999	|	-0.0030	|
|	AC	|	0.0860	|	0.4277	|
|	NMI	|	0.0939	|	0.5092	|
|	AMI	|	-0.0002	|	0.3142	|

### Few-shot fine-tuning results using AMAE method under LoRa and WiFi datasets after 100 Monte Carlo experiments
|	Dataset	|	LoRa	|	WiFi	|
|	----	|	----	|	----	|
|	10-shot	|	4.80%	|	60.39%	|
|	20-shot	|	15.30%	|	91.92%	|



