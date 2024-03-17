# Gaussian-Mixture Discriminant Analysis (GMDA)

**Characteristics**:
- 3-state classifier for 2D data
- Uses max-likelihood classification
- Decision boundary is not a straight line (hence *non-linear*)
- Assumes that the classes follow a Gaussian mixture that share the same covariance matrix $\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)$ and means $\vec{\mu}_i$, i.e.
```math
p(\vec{z}|0) = f_0(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \vec{\mu}_2, \sigma, \theta_0, \phi_0) = \sin^2(\theta_0)\cos^2(\phi_0)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + 
\sin^2(\theta_0)\sin^2(\phi_0)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma) + 
\cos^2(\theta_0)\tilde{N}(\vec{z}; \vec{\mu}_2, \sigma)
```
and
```math
p(\vec{z}|1) = f_1(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \vec{\mu}_2, \sigma, \theta_1, \phi_1) = \sin^2(\theta_1)\cos^2(\phi_1)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + 
\sin^2(\theta_1)\sin^2(\phi_1)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma) + 
\cos^2(\theta_1)\tilde{N}(\vec{z}; \vec{\mu}_2, \sigma)
```
and 
```math
p(\vec{z}|2) = f_2(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \vec{\mu}_2, \sigma, \theta_2, \phi_2) = \sin^2(\theta_1)\cos^2(\phi_1)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + 
\sin^2(\theta_1)\sin^2(\phi_1)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma) + 
\cos^2(\theta_1)\tilde{N}(\vec{z}; \vec{\mu}_2, \sigma)
```
with
```math
\tilde{N}(\vec{z}; \vec{\mu}, \sigma) = \frac{1}{2 \pi \sigma^2} \exp \left( - \frac{|\vec{z} - \vec{\mu}|^2}{2\sigma^2}\right)
```
a 2D multivariate Gaussian with covariance matrix $\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)$. 

## Max-likelihood classifier

Given $\vec{z}$, a max-likelihood classifier outputs the class $c$ that has larger probability (given $\vec{z}$), i.e. $c$ such that $p(c|\vec{z}) \geq p(j|\vec{z}) \;\forall j \neq c$. These probabilities are calculated using Bayes' theorem and the probabilities of the qubit being in state $j$, $p(j)$. By default, it uses $p(j)=1/3$, where $p(c|\vec{z}) \geq p(j|\vec{z}) \forall j \neq c$ is equivalent to $p(\vec{z}|c) \geq p(\vec{z}|j) \forall j \neq c$. 


## Non-linearity

The decision boundaries of this classifier are not straight lines. As a counter example for linearity, given the three means $\mu_0 = (0,0)$, $\mu_1 = (-1,0)$ and $\mu_2 = (+1,0)$ and the decision line for 0 and 1, if $p(\vec{z}|0) = \tilde{N}(\vec{z}; \vec{\mu}_0, \sigma)$ and $p(\vec{z}|1) = 0.75\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma) + 0.25\tilde{N}(\vec{z}; \vec{\mu}_2, \sigma)$, the decision lines are not given by a straight line. 

 
## Notes on the algorithm

The algorithm for setting up the classifier from the readout calibraton data is based on fitting the PDFs to the histograms of the data. If the data does not fulfill the assumptions described above, the classifier may not be the optimal one (in the sense of *optimal Bayes classifier* and *minimal Bayes error rate*). For example, the linear classifier from `sklearn` may lead to a higher readout fidelity (even though they are both linear classifiers) because its decision boundary is found by minimizing the classification error. 

The algorithm uses the following tricks
1. combine $\vec{z}_c$ from both classes to extract the means and standard deviation (to have more samples in each bin of the histogram). *Note: the parameters* $\theta_c$ *and* $\phi_c$ *are extracted from each* $\vec{z}_c$ 
