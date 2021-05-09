<h1 align="center"> Adaptive wavelet distillation 🌊</h1>
<p align="center"> Wavelets which adapt given data and a pre-trained model.
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
  <img src="https://img.shields.io/badge/pytorch-1.0%2B-blue">
  <img src="https://img.shields.io/github/checks-status/Yu-Group/adaptive-wavelets/master">
</p>  
<p align="center"> <i> Faster and more interpretable. </i></p>

![](docs/awd.jpg)

# Related work

- TRIM (ICLR 2020 workshop [pdf](https://arxiv.org/abs/2003.01926), [github](https://github.com/csinva/transformation-importance)) - using simple reparameterizations, allows for calculating disentangled importances to transformations of the input (e.g. assigning importances to different frequencies)
- ACD (ICLR 2019 [pdf](https://openreview.net/pdf?id=SkEqro0ctQ), [github](https://github.com/csinva/hierarchical-dnn-interpretations)) - extends CD to CNNs / arbitrary DNNs, and aggregates explanations into a hierarchy
- CDEP (ICML 2020 [pdf](https://arxiv.org/abs/1909.13584), [github](https://github.com/laura-rieger/deep-explanation-penalization)) - penalizes CD / ACD scores during training to make models generalize better
- DAC (arXiv 2019 [pdf](https://arxiv.org/abs/1905.07631), [github](https://github.com/csinva/disentangled-attribution-curves)) - finds disentangled interpretations for random forests
- PDR framework (PNAS 2019 [pdf](https://arxiv.org/abs/1901.04592)) - an overarching framewwork for guiding and framing interpretable machine learning