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
|	----	|	----	|	----	|	----	|
|	SC	|	-0.0999	|	-0.0030	|
|	AC	|	0.0860	|	0.4277	|
|	NMI	|	0.0939	|	0.5092	|
|	AMI	|	-0.0002	|	0.3142	|

### Few-shot fine-tuning results using AMAE method under LoRa and WiFi datasets after 100 Monte Carlo experiments
|	Dataset	|	LoRa	|	WiFi	|
|	----	|	----	|	----	|	----	|
|	10-shot	|	4.80%	|	60.39%	|
|	20-shot	|	15.30%	|	91.92%	|


|	C-K	|	FS CVCNN	|	Softmax	|	Siamese	|	Triplet	|	SR2CNN	|	*STC	|	ST	|	SC	|
|	----	|	----	|	----	|	----	|	----	|	----	|	----	|	----	|	----	|
|	10-1	|	10.00%	|	41.30%	|	50.60%	|	75.98%	|	85.04%	|	87.66%	|	73.35%	|	85.37%	|
|	10-5	|	47.80%	|	75.26%	|	77.51%	|	90.18%	|	93.01%	|	93.99%	|	88.66%	|	93.05%	|
|	10-10	|	69.40%	|	87.48%	|	83.34%	|	93.00%	|	94.32%	|	95.28%	|	92.31%	|	94.04%	|
|	10-15	|	77.20%	|	91.03%	|	89.47%	|	93.78%	|	94.81%	|	95.88%	|	93.29%	|	94.45%	|
|	10-20	|	84.30%	|	93.56%	|	91.47%	|	94.18%	|	94.82%	|	96.15%	|	94.34%	|	94.99%	|
|	20-1	|	5.00%	|	35.59%	|	38.02%	|	57.02%	|	64.52%	|	66.11%	|	51.33%	|	61.20%	|
|	20-5	|	38.80%	|	61.73%	|	57.00%	|	71.27%	|	76.05%	|	80.01%	|	71.37%	|	75.50%	|
|	20-10	|	53.60%	|	72.18%	|	66.73%	|	74.96%	|	80.94%	|	84.42%	|	77.46%	|	79.05%	|
|	20-15	|	63.25%	|	76.99%	|	69.78%	|	77.93%	|	82.94%	|	86.53%	|	81.06%	|	81.22%	|
|	20-20	|	72.50%	|	81.18%	|	72.38%	|	79.29%	|	84.63%	|	87.74%	|	83.02%	|	82.59%	|
|	30-1	|	3.33%	|	27.68%	|	28.81%	|	46.18%	|	51.28%	|	55.94%	|	41.81%	|	52.04%	|
|	30-5	|	26.90%	|	53.70%	|	47.77%	|	62.02%	|	68.91%	|	72.46%	|	61.40%	|	66.20%	|
|	30-10	|	47.40%	|	64.22%	|	58.30%	|	67.54%	|	73.33%	|	77.60%	|	68.96%	|	71.22%	|
|	30-15	|	54.57%	|	69.99%	|	62.79%	|	70.12%	|	75.58%	|	80.14%	|	73.68%	|	73.89%	|
|	30-20	|	63.30%	|	74.04%	|	65.70%	|	72.62%	|	77.77%	|	81.37%	|	76.20%	|	75.52%	|

