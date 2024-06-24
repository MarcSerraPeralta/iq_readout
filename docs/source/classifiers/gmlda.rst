Gaussian-Mixture Linear Discriminant Analysis (``gmlda``)
=========================================================

Documentation for :py:mod:`iq_readout.two_state_classifiers.GaussMixLinearClassifier`

Characteristics
---------------

- 2-state classifier for 2D data
- Uses max-likelihood classification
- Decision boundary is a straight line (hence *linear*)
- Assumes that the classes follow a Gaussian mixture that share the same covariance matrix :math:`\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)` and means :math:`\vec{\mu}_0, \vec{\mu}_1`, i.e.

.. math::

   p(\vec{z}|0) = f_0(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \sigma, \theta_0) = \sin^2(\theta_0)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + \cos^2(\theta_0)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma)

and

.. math::

   p(\vec{z}|1) = f_1(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \sigma, \theta_1) = \sin^2(\theta_1)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + \cos^2(\theta_1)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma)

with

.. math::

   \tilde{N}(\vec{z}; \vec{\mu}, \sigma) = \frac{1}{2 \pi \sigma^2} \exp \left( - \frac{|\vec{z} - \vec{\mu}|^2}{2\sigma^2}\right)

a 2D multivariate Gaussian with covariance matrix :math:`\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)`. 


Max-likelihood classifier
-------------------------

Given :math:`\vec{z}`, a max-likelihood classifier outputs the class :math:`c` that has larger probability (given :math:`\vec{z}`), i.e. :math:`c` such that :math:`p(c|\vec{z}) \geq p(j|\vec{z}) \;\forall j \neq c`. These probabilities are calculated using Bayes' theorem and the probabilities of the qubit being in state :math:`j`, :math:`p(j)`. By default, it uses :math:`p(j)=1/2`, where :math:`p(c|\vec{z}) \geq p(j|\vec{z}) \forall j \neq c` is equivalent to :math:`p(\vec{z}|c) \geq p(\vec{z}|j) \forall j \neq c`. 


Linearity
---------

The decision boundary for (2-state) max-likelihood classifiers is given by the (parametrized) curve :math:`\vec{d}(t)` that fulfills :math:`p(0|\vec{d}) = p(1|\vec{d}) \;\forall t`. As the PDFs :math:`p(\vec{z}|i)` are symmetric with respect to :math:`z_{\parallel}`, the decision boundary must also be symmetric. Moreover, because the contribution of :math:`z_{\perp}` to :math:`p(\vec{z}|i)` is the same for both :math:`i=0` and :math:`i=1` (i.e. :math:`\tilde{N}\_1(z\_{\perp}; \vec{\mu}\_{1,\perp}, \sigma)` with :math:`\vec{\mu}\_{1,\perp}=\vec{\mu}\_{0,\perp}`), then the decision boundary is of the form :math:`f(z\_{\parallel}) = g(z\_{\parallel}) \\;\forall z\_{\perp}`. Therefore the decision boundary is a straight line along the direction of :math:`\hat{e}\_{\perp}` that crosses the point in the :math:`\hat{e}\_{\parallel}`-axis that fulfills :math:`f(z\_{\parallel}) = g(z\_{\parallel})`. 

The expression for :math:`\vec{d}(t)` can be found as follows. Firstly, we make use of the Bayes' theorem and substitue :math:`p(\vec{z}|i)` for this classifier to get

.. math ::

   p(0)\left[ c_0 \exp \left( - \frac{|\vec{d} - \vec{\mu}_0|^2}{2\sigma^2}\right) + (1-c_0)\exp \left( - \frac{|\vec{d} - \vec{\mu}_1|^2}{2\sigma^2}\right) \right] = (1 - p(0))\left[ b_0 \exp \left( - \frac{|\vec{d} - \vec{\mu}_0|^2}{2\sigma^2}\right) + (1-b_0)\exp \left( - \frac{|\vec{d} - \vec{\mu}_1|^2}{2\sigma^2}\right) \right],

with :math:`c_0 = \sin^2(\theta_0)` and :math:`b_0 = \sin^2(\theta_1)`. By multiplying the expression by :math:`\exp(+|\vec{d} - \vec{\mu}_0|^2 / 2\sigma^2)` and rearraging terms, we can write

.. math ::

   \exp \left( - \frac{|\vec{d} - \vec{\mu}_0|^2 - |\vec{d} - \vec{\mu}_0|^2}{2\sigma^2}\right) = \frac{(1-p(0))b_0 - p(0)c_0}{(1-c_0)p_0 - (1-p_0)(1-b_0)} \equiv P.

After taking the logarithm and using :math:`|\vec{a}|^2 - |\vec{b}|^2 = (\vec{a} - \vec{b})(\vec{a} + \vec{b})`, we obtain the linear equation

.. math ::

   \left( \vec{d} - \frac{\vec{\mu}_0 + \vec{\mu}_1}{2} \right) \cdot (\vec{\mu}_0 - \vec{\mu}_1) = \sigma^2 \log(P)

with solution

.. math ::

   \vec{d}(t) = \mu_{\perp} t + \frac{\vec{\mu}_0 + \vec{\mu}_1}{2} + \frac{\vec{\mu}_0 - \vec{\mu}_1}{|\vec{\mu}_0 - \vec{\mu}_1|} \sigma^2 \log(P),

with :math:`t \in \mathbb{R}` and :math:`\mu_{\perp}` the vector perpendicular to :math:`\vec{\mu}_0 - \vec{\mu}_1`. This curve is a line that is perpendicular with the axis defined by the two *blobs* in the IQ plane. In the particular case :math:`p(0) = 1/2`, then :math:`P=1` and thus the line is in the middle of the two *blobs*. 


Notes on the algorithm
----------------------

The algorithm for setting up the classifier from the readout calibraton data is based on fitting the PDFs to the histograms of the data. If the data does not fulfill the assumptions described above, the classifier may not be the optimal one (in the sense of *optimal Bayes classifier* and *minimal Bayes error rate*). For example, the linear classifier from `sklearn` may lead to a higher readout fidelity (even though they are both linear classifiers) because its decision boundary is found by minimizing the classification error. 

As the classifier is linear, the data can be projected to the axis orthogonal to the decision boundary. 
The projection axis corresponds to the line with direction :math:`\vec{\mu}_1 - \vec{\mu}_0` that crosses these two means. 
The direction is chosen this way to have the *blob* from state 0 on the left and the *blob* from state 1 on the right. 
The projection axis can be estimated from the means of the data for each class :math:`c`, :math:`\\{\vec{z}^{(i)}_c\\}_i`, given by

.. math ::

   \vec{\nu}_c = \frac{1}{N}\sum_{i=1}^N \vec{z}^{(i)}_c, 

because :math:`\vec{\mu}_1 - \vec{\mu}_0 \propto \vec{\nu}_1 - \vec{\nu}_0`. The justification is that, given :math:`\vec{z}_c \sim p(\vec{z}|c)`, the estimator of the mean is :math:`\vec{\nu}_c = \sin^2(\theta_c) \vec{\mu}_0 + \cos^2(\theta_c) \vec{\mu}_1`, thus :math:`\vec{\nu}_1 - \vec{\nu}_0 = (\sin^2(\theta_1) - \sin^2(\theta_0)) (\vec{\mu}_1 - \vec{\mu}_0)`. 

The algorithm uses the following tricks
1. work with projected data (to have more samples in each bin of the histogram)
1. combine :math:`\vec{z}_c` from both classes to extract the means and standard deviation (to have more samples in each bin of the histogram). *Note: the parameters* :math:`\theta_c` *are extracted from each* :math:`\vec{z}_c` 

The algorithm can give :math:`p(z_{\parallel}|i)` with :math:`z_{\parallel}` the projection of :math:`\vec{z}` or :math:`p(\vec{z}|i)`. Note that the two pdfs are related, i.e. :math:`p(z_{\parallel}|0) / p(z_{\parallel}|1) = p(\vec{z}|0) / p(\vec{z}|1)`. The explanation uses the coordinate system of the projection axis and its perpendicular, labelled :math:`\vec{z} = z_{\parallel} \hat{e}\_{\parallel} +z_{\perp}\hat{e}_{\perp}`, which gives

.. math ::

   \frac{p(\vec{z}|0)}{p(\vec{z}|1)} = \exp \left( -\frac{1}{2\sigma^2}((z_{\parallel} - \vec{\mu}_{0,\parallel})^2 - (z_{\parallel} - \vec{\mu}_{1,\parallel})^2) \right)

because :math:`\vec{\mu}\_{0,\perp} = \vec{\mu}\_{1,\perp}` and the terms cancel each other. We then just need to use that

.. math::

   p(z_{\parallel}|i) = \int_{-\infty}^{+\infty} p(z_{\parallel}, z_{\perp}|i) dz_{\perp} \propto \exp \left( -\frac{1}{2\sigma^2}(z_{\parallel} - \vec{\mu}_{i,\parallel})^2 \right)

leading to :math:`p(z_{\parallel}|0) / p(z_{\parallel}|1) = p(\vec{z}|0) / p(\vec{z}|1)`. 
