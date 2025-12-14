Decay Linear Discriminant Analysis
==================================

Documentation for :py:mod:`iq_readout.two_state_classifiers.DecayLinearClassifier`

The PDFs are derived in *Improved quantum error correction using soft information* by Pattison *et al.* (`arxiv pdf <https://arxiv.org/pdf/2107.13589.pdf>`_). In this classifier, the :math:`p(\vec{z}|0)` is a mixture of Gaussians to handle state preparation errors in the readout calibration data. 

Characteristics
---------------

- 2-state classifier for 2D data
- Uses max-likelihood classification
- Decision boundary is a straight line (hence *linear*)
- Assumes that the integration weights of the readout traces (output voltage as a function of time) are constant
- Assumes that the Gaussian distributions present in :math:`p(\vec{z}|i)` share the same covariance matrix :math:`\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)` and means :math:`\vec{\mu}_0, \vec{\mu}_1`


.. math::
   p(\vec{z}|0) = f_0(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \sigma, \theta_0) = \sin^2(\theta_0)\tilde{N}_2(\vec{z}; \vec{\mu}_0, \sigma) + \cos^2(\theta_0)\tilde{N}_2(\vec{z}; \vec{\mu}_1, \sigma)

and

.. math::
   p(\vec{z}|1) = f_1(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \sigma, \theta_1, \tilde{T}_1) = \Big[ \sin^2(\theta_1)\tilde{N}_1(z_{\parallel}; \vec{\mu}_{0, \parallel}, \sigma, \tilde{T}_1) + \cos^2(\theta_1) D(z_{\parallel}; \vec{\mu}_{0, \parallel}, \vec{\mu}_{1, \parallel}, \sigma) \Big] \tilde{N}_1(z_{\perp}; \vec{\mu}_{1,\perp}, \sigma)

with :math:`\vec{z} = z_{\perp} \hat{e}_{\perp} + z_{\parallel} \hat{e}_{\parallel}`, :math:`\hat{e}_{\parallel} = (\vec{\mu}_1 - \vec{\mu}_0) / |\vec{\mu}_1 - \vec{\mu}_0|`, :math:`\hat{e}_{\perp} \perp \hat{e}_{\parallel}`,

.. math::
   \tilde{N}_2(\vec{z}; \vec{\mu}, \sigma) = \frac{1}{2 \pi \sigma^2} \exp \left( - \frac{|\vec{z} - \vec{\mu}|^2}{2\sigma^2}\right)

a 2D Gaussian with covariance matrix :math:`\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)`, 

.. math::
   \tilde{N}_1(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{|\vec{z} - \vec{\mu}|^2}{2\sigma^2}\right)

a 1D Gaussian,

.. math::
   D(x; \mu_0, \mu_1, \sigma, \tilde{T}_1) = \exp\Big(-\frac{(x - \mu_0)^2}{2\sigma^2} + C(x; \mu_0, \mu_1, \sigma, \tilde{T}_1)^2 \Big) \sqrt{\frac{2\sigma^2 \tilde{T}_1^2}{4P(\mu_0, \mu_1, \sigma)}} \frac{\mathrm{erf}(C(x; \mu_0, \mu_1, \sigma, \tilde{T}_1) + \sqrt{P(\mu_0, \mu_1, \sigma)}) + \mathrm{erf}(C(x; \mu_0, \mu_1, \sigma, \tilde{T}_1))}{1 - \exp(-1/\tilde{T}_1)}


.. math::
   C(x; \mu_0, \mu_1, \sigma, \tilde{T}_1) = \frac{\mathrm{sign}(\mu_1 - \mu_0) \cdot (\mu_0 - x)}{\sqrt{2} \sigma} + \frac{\sigma}{\sqrt{2}|\mu_1 - \mu_0| \tilde{T}_1}


.. math::
   P(\mu_0, \mu_1, \sigma) = \frac{(\mu_1 - \mu_0)^2}{2\sigma^2}

and :math:`\tilde{T}_1 = T_1 / t_M` the normalized amplitude decay time with respect to the measurement time. 


Example
-------

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   from iq_readout.two_state_classifiers import DecayLinearClassifier
   from iq_readout.plots.shots1d import plot_two_pdfs_projected
   from iq_readout.plots.shots2d import plot_shots_2d, plot_boundaries_2d
   
   shots_0, shots_1 = np.load("data_two_state_calibration.npy")
   classifier = DecayLinearClassifier.fit(shots_0, shots_1)

   fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))
   axes[0] = plot_shots_2d(axes[0], shots_0, shots_1)
   axes[1] = plot_two_pdfs_projected(
        axes[1],
        classifier,
        shots_0,
        shots_1,
   )
   axes[0] = plot_boundaries_2d(axes[0], classifier)

   axes[0].set_title("Decision boundaries")
   axes[1].set_title("PDFs")
   plt.show()


Max-likelihood classifier
-------------------------

Given :math:`\vec{z}`, a max-likelihood classifier outputs the class :math:`c` that has larger probability (given :math:`\vec{z}`), i.e. :math:`c` such that :math:`p(c|\vec{z}) \geq p(j|\vec{z}) \;\forall j \neq c`. These probabilities are calculated using Bayes' theorem and the probabilities of the qubit being in state :math:`j`, :math:`p(j)`. By default, it uses :math:`p(j)=1/2`, where :math:`p(c|\vec{z}) \geq p(j|\vec{z}) \forall j \neq c` is equivalent to :math:`p(\vec{z}|c) \geq p(\vec{z}|j) \forall j \neq c`. 


Linearity
---------

The decision boundary for (2-state) max-likelihood classifiers is given by the (parametrized) curve :math:`\vec{d}(t)` that fulfills :math:`p(0|\vec{d}) = p(1|\vec{d}) \;\forall t`. As the PDFs :math:`p(\vec{z}|i)` are symmetric with respect to :math:`z_{\parallel}`, the decision boundary must also be symmetric. Moreover, because the contribution of :math:`z_{\perp}` to :math:`p(\vec{z}|i)` is the same for both :math:`i=0` and :math:`i=1` (i.e. :math:`\tilde{N}_1(z_{\perp}; \vec{\mu}_{1,\perp}, \sigma)` with :math:`\vec{\mu}_{1,\perp}=\vec{\mu}_{0,\perp}`), then the decision boundary is of the form :math:`f(z_{\parallel}) = g(z_{\parallel}) \\;\forall z_{\perp}`. Therefore the decision boundary is a straight line along the direction of :math:`\hat{e}_{\perp}` that crosses the point in the :math:`\hat{e}_{\parallel}`-axis that fulfills :math:`f(z_{\parallel}) = g(z_{\parallel})`. 


Notes on the algorithm
----------------------

The algorithm for setting up the classifier from the readout calibraton data is based on fitting the PDFs to the histograms of the data. If the data does not fulfill the assumptions described above, the classifier may not be the optimal one (in the sense of *optimal Bayes classifier* and *minimal Bayes error rate*). For example, the linear classifier from `sklearn` may lead to a higher readout fidelity (even though they are both linear classifiers) because its decision boundary is found by minimizing the classification error. 

As the classifier is linear, the data can be projected to the axis orthogonal to the decision boundary. 
The projection axis corresponds to the line with direction :math:`\vec{\mu}_1 - \vec{\mu}_0` that crosses these two means. 
The direction is chosen this way to have the *blob* from state 0 on the left and the *blob* from state 1 on the right. 
The projection axis can be estimated from the means of the data for each class :math:`c`, :math:`\{\vec{z}^{(i)}_c\}_i`, given by

.. math ::
   \vec{\nu}_c = \frac{1}{N}\sum_{i=1}^N \vec{z}^{(i)}_c, 

because :math:`\vec{\mu}_1 - \vec{\mu}_0 \propto \vec{\nu}_1 - \vec{\nu}_0`. The justification follows the same used in the linearity section. 

The algorithm uses the following tricks:

#. work with projects the data (to have more samples in each bin of the histogram)
#. combine :math:`\vec{z}_c` from both classes to extract the means and standard deviation (to have more samples in each bin of the histogram). *Note: the parameters* :math:`\theta_c` *are extracted from each* :math:`\vec{z}_c` 

The algorithm can give :math:`p(z_{\parallel}|i)` with :math:`z_{\parallel}` the projection of :math:`\vec{z}` or :math:`p(\vec{z}|i)`. Note that the two pdfs are related, i.e. :math:`p(z_{\parallel}|0) / p(z_{\parallel}|1) = p(\vec{z}|0) / p(\vec{z}|1)`. The explanation can be found in :ref:`classifiers/gmlda:Gaussian-Mixture Linear Discriminant Analysis`. 
