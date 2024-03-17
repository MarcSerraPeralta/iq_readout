# iq_readout

Classifiers for the IQ readout data from superconducting qubits.

# Summary of classifiers

## 2-state classifiers

1. [Gaussian-Mixture Linear Discriminant Analysis](iq_readout/two_state_classifiers/gmlda.md)
1. [Decay Linear Discriminant Analysis](iq_readout/two_state_classifiers/decaylda.md)

## 3-state classifiers

1. [Gaussian-Mixture Discriminant Analsys](iq_readout/three_state_classifiers/gmda.md)

## Notes on the fitting of the classifiers

The algorithms for setting up the classifiers from the readout calibraton data are based on fitting the PDFs to the histograms of the data. If the data does not fulfill the assumptions described in the markdown file of a given classifier, then it may not be the optimal one (in the sense of *optimal Bayes classifier* and *minimal Bayes error rate*). For example, the linear classifier from `sklearn` may lead to a higher readout fidelity than `iq_readout.two_state_classifiers.GaussMixLinearClassifier` (even though they are both linear classifiers) because the decision boundary is found by minimizing the classification error (not by fitting the PDFs). 