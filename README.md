# A-Method-for-Solving-the-FS-SEI-Problem
A Method for Solving the FS-SEI Problem.

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

### pre-training demo
* Start pre-training by running the pretrain.py file.
* After the pre-training is completed, the Visualization.py file can be run to visualize the features of the pre-trained model, and the pre-trained model can be evaluated using unsupervised clustering indicators.

### Fine-tuning with pre-trained checkpoints
* Start fine-tuning by running the fine-tuning.py file.
