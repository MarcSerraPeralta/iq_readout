# Gaussian-Mixture Discriminant Analysis (GMLDA)

**Characteristics**:
- 3-state classifier for 2D data
- Uses max-likelihood classification
- Decision boundary is not a straight line 
- Assumes that the classes follow a Gaussian mixture that share the same covariance matrix $\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)$ and means $\vec{\mu}_i$, i.e.
```math
p(\vec{x}|0) = f_0(\vec{x}; \vec{\mu}_0, \vec{\mu}_1, \vec{\mu}_2, \sigma, \theta_0, \phi_0) = \sin^2(\theta_0)\cos^2(\phi_0)\tilde{N}(\vec{x}; \vec{\mu}_0, \sigma) + 
\sin^2(\theta_0)\sin^2(\phi_0)\tilde{N}(\vec{x}; \vec{\mu}_1, \sigma) + 
\cos^2(\theta_0)\tilde{N}(\vec{x}; \vec{\mu}_2, \sigma)
```
and
```math
p(\vec{x}|1) = f_1(\vec{x}; \vec{\mu}_0, \vec{\mu}_1, \vec{\mu}_2, \sigma, \theta_1, \phi_1) = \sin^2(\theta_1)\cos^2(\phi_1)\tilde{N}(\vec{x}; \vec{\mu}_0, \sigma) + 
\sin^2(\theta_1)\sin^2(\phi_1)\tilde{N}(\vec{x}; \vec{\mu}_1, \sigma) + 
\cos^2(\theta_1)\tilde{N}(\vec{x}; \vec{\mu}_2, \sigma)
```
and 
```math
p(\vec{x}|2) = f_2(\vec{x}; \vec{\mu}_0, \vec{\mu}_1, \vec{\mu}_2, \sigma, \theta_2, \phi_2) = \sin^2(\theta_1)\cos^2(\phi_1)\tilde{N}(\vec{x}; \vec{\mu}_0, \sigma) + 
\sin^2(\theta_1)\sin^2(\phi_1)\tilde{N}(\vec{x}; \vec{\mu}_1, \sigma) + 
\cos^2(\theta_1)\tilde{N}(\vec{x}; \vec{\mu}_2, \sigma)
```
with
```math
\tilde{N}(\vec{x}; \vec{\mu}, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{|\vec{x} - \vec{\mu}|^2}{2\sigma^2}\right)
```
a multivariate Gaussian with covariance matrix $\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)$. 

## Max-likelihood classifier

Given $\vec{x}$, a max-likelihood classifier outputs the class $c$ that has larger probability, i.e. $c$ such that $p(c|\vec{x}) \geq p(j|\vec{x}) \forall j \neq c$. These probabilities are calculated using Bayes' theorem and the probabilities of the qubit being in state $j$. By default, it uses $p(j)=1/3$, where $p(c|\vec{x}) \geq p(j|\vec{x}) \forall j \neq c$ is equivalent to $p(\vec{x}|c) \geq p(\vec{x}|j) \forall j \neq c$. 


## Non-linearity

As a counter example for linearity, given the three means $\mu_0 = (0,0)$, $\mu_1 = (-1,0)$ and $\mu_2 = (+1,0)$ and the decision line for 0 and 1, if $p(\vec{x}|0) = \tilde{N}(\vec{x}; \vec{\mu}_0, \sigma)$ and $p(\vec{x}|1) = 0.75\tilde{N}(\vec{x}; \vec{\mu}_1, \sigma) + 0.25\tilde{N}(\vec{x}; \vec{\mu}_2, \sigma)$, the decision lines are not given by a straight line. 

## Notes on the algorithm

The algorithm uses the following tricks
1. combine $\vec{x}_c$ from both classes to extract the means and standard deviation (to have more samples in each bin of the histogram). *Note: the parameters* $\theta_c$ *and* $\phi_c$ *are extracted from each* $\vec{x}_c$ 
