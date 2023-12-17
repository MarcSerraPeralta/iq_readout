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
(\sin^2(\theta_0) - \sin^2(\theta_1))\tilde{\mathcal{N}}(\vec{x}; \vec{\mu}_0, \sigma) = 
(\cos^2(\theta_1) - \cos^2(\theta_1))\tilde{\mathcal{N}}(\vec{x}; \vec{\mu}_1, \sigma)
```
which can be simplified by using that $\sin^2(\theta) + \cos^2(\theta) = 1$, leading to
```math
\tilde{\mathcal{N}}(\vec{x}; \vec{\mu}_0, \sigma) = \tilde{\mathcal{N}}(\vec{x}; \vec{\mu}_1, \sigma).
```
By taking the logarithm and simplifying, we obtain
```math
|\vec{x} - \vec{\mu}_0| = |\vec{x} - \vec{\mu}_1|,
```
which is a linear equation for the decision boundary. Noteworthy, the decision boundary does not depend on the weights of the Gaussian mixture, it crosses $\vec{x}=(\vec{\mu}_0 - \vec{\mu}_1)/2$, and it is perpendicular to the direction $\vec{\mu}_0 - \vec{\mu}_1$. 