Gaussian-Mixture Discriminant Analysis
======================================

Documentation for :py:mod:`iq_readout.three_state_classifiers.GaussMixClassifier`

Characteristics
---------------

- 3-state classifier for 2D data
- Uses max-likelihood classification
- Decision boundary is not a straight line (hence *non-linear*)
- Assumes that the classes follow a Gaussian mixture that share the same covariance matrix :math:`\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)` and means :math:`\vec{\mu}_i`, i.e.

.. math::
   p(\vec{z}|0) &= f_0(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \vec{\mu}_2, \sigma, \theta_0, \phi_0) \\\\
   &= \sin^2(\theta_0)\cos^2(\phi_0)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + \sin^2(\theta_0)\sin^2(\phi_0)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma) + \cos^2(\theta_0)\tilde{N}(\vec{z}; \vec{\mu}_2, \sigma)

and

.. math::
   p(\vec{z}|1) &= f_1(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \vec{\mu}_2, \sigma, \theta_1, \phi_1) \\\\
   &= \sin^2(\theta_1)\cos^2(\phi_1)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + \sin^2(\theta_1)\sin^2(\phi_1)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma) + \cos^2(\theta_1)\tilde{N}(\vec{z}; \vec{\mu}_2, \sigma)

and 

.. math::
   p(\vec{z}|2) &= f_2(\vec{z}; \vec{\mu}_0, \vec{\mu}_1, \vec{\mu}_2, \sigma, \theta_2, \phi_2) \\\\
   &= \sin^2(\theta_1)\cos^2(\phi_1)\tilde{N}(\vec{z}; \vec{\mu}_0, \sigma) + \sin^2(\theta_1)\sin^2(\phi_1)\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma) + \cos^2(\theta_1)\tilde{N}(\vec{z}; \vec{\mu}_2, \sigma)

with

.. math::
   \tilde{N}(\vec{z}; \vec{\mu}, \sigma) = \frac{1}{2 \pi \sigma^2} \exp \left( - \frac{|\vec{z} - \vec{\mu}|^2}{2\sigma^2}\right)

a 2D multivariate Gaussian with covariance matrix :math:`\Sigma=\mathrm{diag}(\sigma^2, \sigma^2)`. 


Example
-------

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   from iq_readout.three_state_classifiers import GaussMixClassifier
   from iq_readout.plots.shots1d import plot_several_pdfs_along_line
   from iq_readout.plots.shots2d import plot_shots_2d, plot_boundaries_2d 
   
   shots_0, shots_1, shots_2 = np.load("data_three_state_calibration.npy")
   classifier = GaussMixClassifier.fit(shots_0, shots_1, shots_2)

   fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9))
   axes[0, 0] = plot_shots_2d(axes[0, 0], shots_0, shots_1, shots_2)
   axes[0, 0] = plot_boundaries_2d(axes[0, 0], classifier)

   params = classifier.params
   mu_0 = [params[0]["mu_0_x"], params[0]["mu_0_y"]]
   mu_1 = [params[1]["mu_1_x"], params[1]["mu_1_y"]]
   mu_2 = [params[2]["mu_2_x"], params[2]["mu_2_y"]]

   plot_several_pdfs_along_line(
       axes[0, 1],
       [mu_0, mu_1],
       classifier,
       shots_0,
       shots_1,
       shots_2,
   )
   plot_several_pdfs_along_line(
       axes[1, 0],
       [mu_1, mu_2],
       classifier,
       shots_0,
       shots_1,
       shots_2,
   )
   plot_several_pdfs_along_line(
       axes[1, 1],
       [mu_0, mu_2],
       classifier,
       shots_0,
       shots_1,
       shots_2,
   )

   axes[0, 0].set_title("Decision boundaries")
   axes[0, 1].set_title("PDFs projected in 0-1 axis")
   axes[1, 0].set_title("PDFs projected in 1-2 axis")
   axes[1, 1].set_title("PDFs projected in 0-2 axis")
   plt.show()


Max-likelihood classifier
-------------------------

Given :math:`\vec{z}`, a max-likelihood classifier outputs the class :math:`c` that has larger probability (given :math:`\vec{z}`), i.e. :math:`c` such that :math:`p(c|\vec{z}) \geq p(j|\vec{z}) \;\forall j \neq c`. These probabilities are calculated using Bayes' theorem and the probabilities of the qubit being in state :math:`j`, :math:`p(j)`. By default, it uses :math:`p(j)=1/3`, where :math:`p(c|\vec{z}) \geq p(j|\vec{z}) \forall j \neq c` is equivalent to :math:`p(\vec{z}|c) \geq p(\vec{z}|j) \forall j \neq c`. 


Non-linearity
-------------

The decision boundaries of this classifier are not straight lines. As a counter example for linearity, given the three means :math:`\mu_0 = (0,0)`, :math:`\mu_1 = (-1,0)` and :math:`\mu_2 = (+1,0)` and the decision line for 0 and 1, if :math:`p(\vec{z}|0) = \tilde{N}(\vec{z}; \vec{\mu}_0, \sigma)` and :math:`p(\vec{z}|1) = 0.75\tilde{N}(\vec{z}; \vec{\mu}_1, \sigma) + 0.25\tilde{N}(\vec{z}; \vec{\mu}_2, \sigma)`, the decision lines are not given by a straight line. 

 
Notes on the algorithm
----------------------

The algorithm for setting up the classifier from the readout calibraton data is based on fitting the PDFs to the histograms of the data. If the data does not fulfill the assumptions described above, the classifier may not be the optimal one (in the sense of *optimal Bayes classifier* and *minimal Bayes error rate*). For example, the linear classifier from `sklearn` may lead to a higher readout fidelity (even though they are both linear classifiers) because its decision boundary is found by minimizing the classification error. 

The algorithm uses the following tricks

#. combine :math:`\vec{z}_c` from all classes to extract the means and standard deviation (to have more samples in each bin of the histogram). *Note: the parameters* :math:`\theta_c` *and* :math:`\phi_c` *are extracted from each* :math:`\vec{z}_c` 
