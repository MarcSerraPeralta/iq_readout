# Gaussian-Mixture Linear Discriminant Analysis (GMLDA)

**Characteristics**:
- 2-state classifier for 2D data
- Uses max-likelihood classification
- Decision boundary is a straing line (hence *linear*)
- Assumes that the classes follow a Gaussian mixture that share the same covariance matrix $\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)$ and means $\vec{\mu}_0, \vec{\mu}_1$, i.e.
```math
p(\vec{x}|0) = f_0(\vec{x}; \vec{\mu}_0, \vec{\mu}_1, \sigma, \theta_0) = \sin^2(\theta_0)\tilde{N}(\vec{x}; \vec{\mu}_0, \sigma) + 
\cos^2(\theta_0)\tilde{N}(\vec{x}; \vec{\mu}_1, \sigma)
```
and
```math
p(\vec{x}|1) = f_1(\vec{x}; \vec{\mu}_0, \vec{\mu}_1, \sigma, \theta_1) = \sin^2(\theta_1)\tilde{N}(\vec{x}; \vec{\mu}_0, \sigma) + 
\cos^2(\theta_1)\tilde{N}(\vec{x}; \vec{\mu}_1, \sigma)
```
with
```math
\tilde{N}(\vec{x}; \vec{\mu}, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{|\vec{x} - \vec{\mu}|^2}{2\sigma^2}\right)
```
a multivariate Gaussian with covariance matrix $\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)$. 

## Max-likelihood classifier

Given $\vec{x}$, a max-likelihood classifier outputs the class $c$ that has larger probability, i.e. $c$ such that $p(\vec{x}|c) \geq p(\vec{x}|j) \forall j \neq c$. 

## Linearity

The decision boundary for (2-state) max-likelihood classifiers is given by $p(\vec{x}|0) = p(\vec{x}|1)$. Therefore, by reordering the terms we get
```math
(\sin^2(\theta_0) - \sin^2(\theta_1))\tilde{N}(\vec{x}; \vec{\mu}_0, \sigma) = 
(\cos^2(\theta_1) - \cos^2(\theta_1))\tilde{N}(\vec{x}; \vec{\mu}_1, \sigma)
```
which can be simplified by using that $\sin^2(\theta) + \cos^2(\theta) = 1$, leading to
```math
\tilde{N}(\vec{x}; \vec{\mu}_0, \sigma) = \tilde{N}(\vec{x}; \vec{\mu}_1, \sigma).
```
By taking the logarithm and simplifying, we obtain
```math
|\vec{x} - \vec{\mu}_0| = |\vec{x} - \vec{\mu}_1|,
```
which is a linear equation for the decision boundary. Noteworthy, the decision boundary does not depend on the weights of the Gaussian mixture, it crosses $\vec{x}=(\vec{\mu}_0 - \vec{\mu}_1)/2$, and it is perpendicular to the direction $\vec{\mu}_0 - \vec{\mu}_1$. 

## Notes on the algorithm

As the classifier is linear, the data can be projected to the axis orthogonal to the decision boundary. 
The projection axis corresponds to the vector $\vec{\mu}_0 - \vec{\mu}_1$. This can be estimated from the means of the data for each class $c$,
```math 
\vec{\nu}_c = \sum_{i=1}^N \vec{x}^{(i)}_c / N, 
```
because $\vec{\mu}_0 - \vec{\mu}_1 \propto \vec{\nu}_1 - \vec{\nu}_0$. The justification is that, given $\vec{x}_c \sim p(\vec{x}|c)$, the estimator of the mean $\vec{\nu}_c = \sin^2(\theta_c) \vec{\mu}_0 + \cos^2(\theta_c) \vec{\mu}_1$, thus $\vec{\nu}_1 - \vec{\nu}_0 = (\sin^2(\theta_1) - \sin^2(\theta_0)) (\vec{\mu}_1 - \vec{\mu}_0)$. 

The algorithm uses the following tricks
1. work with projects the data (to have more samples in each bin of the histogram)
1. combine $\vec{x}_c$ from both classes to extract the means and standard deviation (to have more samples in each bin of the histogram). *Note: the parameters $\theta_c$ are extracted from each $\vec{x}_c$*
1. the threshold for the projected data can be obtained exactly from projecting $(\vec{\mu}_0 - \vec{\mu}_1)/2$. *Note: although the threshold could be used for the predictions, it is not used in this algorithm*