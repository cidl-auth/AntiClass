# Inducing Neural Collapse via Anticlasses and One-Cold Cross-Entropy Loss (TNNLS'25)

Official PyTorch implementation of  
[**Inducing Neural Collapse via Anticlasses and One-Cold Cross-Entropy Loss (TNNLS'25)**](https://ieeexplore.ieee.org/document/11053771)  

**Authors:** Dimitrios Katsikas, Nikolaos Passalis, Anastasios Tefas  

---

> ***Abstract** While softmax cross-entropy (CE) loss is the standard objective for supervised classification, it primarily focuses on the ground-truth classes, ignoring the relationships between the nontarget, complementary classes. This leaves valuable information unexploited during optimization. In this work, we propose a novel loss function, one-cold CE (OCCE) loss, which addresses this limitation by structuring the activations of these complementary classes. Specifically, for each class, we define an anticlass, which consists of everything that is not part of the target class—this includes all complementary classes as well as out-of-distribution (OOD) samples, noise, or in general any instance that does not belong to the true class. By setting a uniform one-cold encoded distribution over the complementary classes as a target for each anticlass, we encourage the model to equally distribute activations across all nontarget classes. This approach promotes a symmetric geometric structure of classes in the final feature space, increases the degree of neural collapse (NC) during training, addresses the independence deficit problem of neural networks, and improves generalization. Our extensive evaluation shows that incorporating OCCE loss in the optimization objective consistently enhances performance across multiple settings, including classification, open-set recognition, and OOD detection.* 


<p align="center">
  <img src="assets/framework.jpg" alt="Framework">
</p>

---

## Requirements

To install the required dependencies, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Training

To train classification models, use the `train.py` script. This should reproduce the closed-set classification results of our Table 1 from the paper. To train a ResNet18v2 (PreActivationResNet) on CIFAR100, use the following commands:

#### Train with baseline cross-entropy

```bash
python train.py --model 'preactivationresnet18' --dataset 'cifar100' --w_ce=1.0 --w_occe=0.0 --gpu 0 --seed 0
```

#### Train with proposed unified ce+occe with γ = 1.0

```bash
python train.py --model 'preactivationresnet18' --dataset 'cifar100' --w_ce=1.0 --w_occe=1.0 --gpu 0 --seed 0
```

---

## Example Logs

We provide example TensorBoard logs for three training runs (different seeds) in the logs/ folder. The baseline CE model achieves 24.83% test error, while adding OCCE reduces it to 23.80%. The small ~0.1% differences from the paper are due to different PyTorch versions, but the overall behavior should match across versions.  To plot the accuracy curves from these logs, use the `plot.py` script:

```bash
python plot.py
```

---

## Links to other useful resources

The class `OCCELoss` in `losses.py` can be easily integrated in other training procedures, by adding it alongside the cross-entropy criterion where used. In our paper, we used this approach to evaluate our methods under various classification settings. The following repositories were used to evaluate against benchmarks:

- **Open Set Recognition:** We used the official implementation of [Adversarial Reciprocal Points Learning for Open Set Recognition (TPAMI'21)](https://github.com/iCGY96/ARPL).
- **Out of Distribution Detection:** We used the [OpenOOD: Benchmarking Generalized OOD Detection](https://github.com/Jingkang50/OpenOOD) library.
- **Measuring Neural Collapse:** We used the official implementation of [A Geometric Analysis of Neural Collapse with Unconstrained Features](https://github.com/tding1/Neural-Collapse?utm_source=catalyzex.com).  
- **Measuring Independence Deficit:** We used the official implementation of [Rank Diminishing in Deep Neural Networks](https://github.com/Yukun-Huang/Rank-Diminishing-in-Deep-Neural-Networks).
---

## Citation 

For more technical details and results, please check our [paper](https://ieeexplore.ieee.org/document/11053771):

```bibtex
@ARTICLE{kat2025anticlasses,
  author={Katsikas, Dimitrios and Passalis, Nikolaos and Tefas, Anastasios},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Inducing Neural Collapse via Anticlasses and One-Cold Cross-Entropy Loss}, 
  year={2025},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TNNLS.2025.3580892}}
```
