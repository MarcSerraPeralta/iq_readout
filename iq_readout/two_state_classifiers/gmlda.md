# Gaussian-Mixture Linear Discriminant Analysis (GMLDA)

**Characteristics**:
- 2-state classifier for 2D data
- Uses max-likelihood classification
- Decision boundary is a straight line (hence *linear*)
- Assumes that the classes follow a Gaussian mixture that share the same covariance matrix $\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)$ and means $\vec{\mu}_0, \vec{\mu}_1$, i.e.
```math
p(\vec{z}|0) = f_0(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \sigma, \theta_0) = \sin^2(\theta_0)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + 
\cos^2(\theta_0)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma)
```
and
```math
p(\vec{z}|1) = f_1(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \sigma, \theta_1) = \sin^2(\theta_1)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + 
\cos^2(\theta_1)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma)
```
with
```math
\tilde{N}(\vec{z}; \vec{\mu}, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{|\vec{z} - \vec{\mu}|^2}{2\sigma^2}\right)
```
a multivariate Gaussian with covariance matrix $\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)$. 

## Max-likelihood classifier

Given $\vec{z}$, a max-likelihood classifier outputs the class $c$ that has larger probability, i.e. $c$ such that $p(c|\vec{z}) \geq p(j|\vec{z}) \forall j \neq c$. These probabilities are calculated using Bayes' theorem and the probabilities of the qubit being in state $j$. By default, it uses $p(j)=1/2$, where $p(c|\vec{z}) \geq p(j|\vec{z}) \forall j \neq c$ is equivalent to $p(\vec{z}|c) \geq p(\vec{z}|j) \forall j \neq c$. 

## Linearity

The decision boundary for (2-state) max-likelihood classifiers is given by $p(\vec{z}|0) = p(\vec{z}|1)$. Therefore, by reordering the terms we get
```math
(\sin^2(\theta_0) - \sin^2(\theta_1))\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) = 
(\cos^2(\theta_1) - \cos^2(\theta_1))\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma)
```
which can be simplified by using that $\sin^2(\theta) + \cos^2(\theta) = 1$ (and $\theta_0 \neq \theta_1$), leading to
```math
\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) = \tilde{N}(\vec{z}; \vec{\mu}_1, \sigma).
```
By taking the logarithm and simplifying, we obtain
```math
|\vec{z} - \vec{\mu}_0| = |\vec{z} - \vec{\mu}_1|,
```
which is a linear equation for the decision boundary. Noteworthy, the decision boundary does not depend on the weights of the Gaussian mixture, it crosses $\vec{z}=(\vec{\mu}_1 - \vec{\mu}_0)/2$, and it is perpendicular to the direction $\vec{\mu}_1 - \vec{\mu}_0$. 

## Notes on the algorithm

As the classifier is linear, the data can be projected to the axis orthogonal to the decision boundary. 
The projection axis corresponds to the line with direction $\vec{\mu}_1 - \vec{\mu}_0$ that crosses these two means. 
The direction is chosen this way to have the *blob* from state 0 on the left and the *blob* from state 1 on the right. 
The projection axis can be estimated from the means of the data for each class $c$, $\\{\vec{z}^{(i)}_c\\}_i$, given by
```math 
\vec{\nu}_c = \frac{1}{N}\sum_{i=1}^N \vec{z}^{(i)}_c, 
```
because $\vec{\mu}_1 - \vec{\mu}_0 \propto \vec{\nu}_1 - \vec{\nu}_0$. The justification is that, given $\vec{z}_c \sim p(\vec{z}|c)$, the estimator of the mean is $\vec{\nu}_c = \sin^2(\theta_c) \vec{\mu}_0 + \cos^2(\theta_c) \vec{\mu}_1$, thus $\vec{\nu}_1 - \vec{\nu}_0 = (\sin^2(\theta_1) - \sin^2(\theta_0)) (\vec{\mu}_1 - \vec{\mu}_0)$. 

The algorithm uses the following tricks
1. work with projected data (to have more samples in each bin of the histogram)
1. combine $\vec{z}_c$ from both classes to extract the means and standard deviation (to have more samples in each bin of the histogram). *Note: the parameters* $\theta_c$ *are extracted from each* $\vec{z}_c$ 

The algorithm can give $p(z_{\parallel}|i)$ with $z_{\parallel}$ the projection of $\vec{z}$ or $p(\vec{z}|i)$. Note that the two pdfs are related, i.e. $p(z_{\parallel}|0) / p(z_{\parallel}|1) = p(\vec{z}|0) / p(\vec{z}|1)$. The explanation uses the coordinate system of the projection axis and its perpendicular, labelled $\vec{z} = z_{\parallel} \hat{e}\_{\parallel} +z_{\perp}\hat{e}_{\perp}$, which gives
```math 
\frac{p(\vec{z}|0)}{p(\vec{z}|1)} = \exp \left( -\frac{1}{2\sigma^2}((z_{\parallel} - \vec{\mu}_{0,\parallel})^2 - (z_{\parallel} - \vec{\mu}_{1,\parallel})^2) \right)
```
because $\vec{\mu}_{0,\perp} = \vec{\mu}_{1,\perp}$ and the terms cancel each other. We then just need to use that
```math
p(z_{\parallel}|i) = \int_{-\infty}^{+\infty} p(z_{\parallel}, z_{\perp}|i) dz_{\perp} \propto \exp \left( -\frac{1}{2\sigma^2}(z_{\parallel} - \vec{\mu}_{i,\parallel})^2 \right)
```
leading to $p(z_{\parallel}|0) / p(z_{\parallel}|1) = p(\vec{z}|0) / p(\vec{z}|1)$. 
