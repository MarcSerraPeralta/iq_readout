Max-Fidelity Linear Discriminant Analysis
=========================================

Characteristics
---------------

- 2-state classifier for 2D data
- Uses threshold to classify (projected) 2D data
- Decision boundary is a straight line (hence *linear*)
- Does not assume any probability density function for the 2D data, thus

  - the PDFs will correspond to the histograms of the calibration data
  - the accuracy in the threshold is limited by the bin separation of the histogram of the calibration data


Example
-------

.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   from iq_readout.two_state_classifiers import MaxFidLinearClassifier
   from iq_readout.plots.shots1d import plot_two_pdfs_projected
   from iq_readout.plots.shots2d import plot_shots_2d, plot_boundaries_2d
   
   shots_0, shots_1 = np.load("data_two_state_calibration.npy")
   classifier = MaxFidLinearClassifier.fit(shots_0, shots_1)

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


Threshold classifier
--------------------

The threshold classifiers are two-state linear classifiers that use a threshold line (hence *linear*) to divide the 2D plane into two regions (hence *two-state classifier*). 
Given a threshold :math:`z_{thr}` and a projection function :math:`\vec{z} \rightarrow z_{\parallel}=\vec{z}\cdot \hat{e}_{\parallel}`, it outputs 0 if :math:`z_{\parallel} \leq z_{thr}` and 1 otherwise. 

*Note: the threshold can be updated based on the prior probabilities of the classes, but it requires some computation*


Linearity
---------

The linearity of this classifier comes from its definition, as the decision boundary is a line perpendicular to :math:`\hat{e}_{\parallel}` that crosses the point :math:`\vec{A} = z_{thr}\hat{e}_{\parallel}`. 


Maximum fidelity in a threshold classifier
------------------------------------------

The assignment infidelity :math:`\epsilon` is defined as :math:`\epsilon = (p(m=0|s=1) + p(m=1|s=0))/2`, where :math:`p(m|s)` is the probability of measuring :math:`m` given that the qubit was in state :math:`s`. 
Note that we have assumed that the probability of seeing state 0 and 1 are the same (i.e. p(s=0)=p(s=1)=1/2), but it can be generalized to any :math:`p(s)` using :math:`\epsilon = p(m=1|s=0)p(s=0) + p(m=0|s=1)p(s=1)`. 
Given the cumulative density functions :math:`CDF(z_{\parallel}|s)`, we can rewrite the assingment infidelity as 

.. math::
   \epsilon = \frac{1}{2} [CDF(z_{thr}|1) + 1 - CDF(z_{thr}|0)],

because the probability of incorrectly assigning :math:`m=0` to state :math:`s=1` is the probability that the projected data is less than the threshold, and equivalently for the other case.
Note that we have assumed that (1) the 0 blob is on the left of the 1 blob in the projected axis, and (2) the density functions "behave as expected", meaning that :math:`CDF(z_{\parallel}|0) \geq CDF(z_{\parallel}|1) \;\forall z_{\parallel}`. 
This last assumption can be broken if the PDFs exhibit more than one maximum. 

The threshold that maximizes the assingment fidelity :math:`F = 1 - \epsilon` (i.e. minimizes the assignment infidelity) is given by the point that maximizes the distance between the cumulative density functions, 

.. math::
   z^*_{thr} = \mathrm{argmax}_{z_{thr}} F = \mathrm{argmax}_{z_{thr}} CDF(z_{thr}|0) - CDF(z_{thr}|1),

where we have omited the constants and factors as we are only interested in the :math:`\mathrm{argmax}`, not the :math:`\mathrm{max}` value. 
In the general case where the priors are not equal, we have

.. math::
   z^*_{thr} = \mathrm{argmax}_{z_{thr}} F = \mathrm{argmax}_{z_{thr}} CDF(z_{thr}|0)p(s=0) - CDF(z_{thr}|1)p(s=1),



Notes on the algorithm
----------------------

As the classifier is linear, the data can be projected to the axis orthogonal to the decision boundary. 
The projection axis corresponds to the line with direction :math:`\vec{\mu}_1 - \vec{\mu}_0` that crosses these two means. 
The direction is chosen this way to have the *blob* from state 0 on the left and the *blob* from state 1 on the right. 
The projection axis can be estimated from the means of the data for each class :math:`c`, :math:`\{\vec{z}^{(i)}_c\}_i`, given by

.. math:: 
   \vec{\nu}_c = \frac{1}{N}\sum_{i=1}^N \vec{z}^{(i)}_c, 

because :math:`\vec{\mu}_1 - \vec{\mu}_0 \propto \vec{\nu}_1 - \vec{\nu}_0`. The justification is that, given :math:`\vec{z}_c \sim p(\vec{z}|c)`, the estimator of the mean is :math:`\vec{\nu}_c = \sin^2(\theta_c) \vec{\mu}_0 + \cos^2(\theta_c) \vec{\mu}_1`, thus :math:`\vec{\nu}_1 - \vec{\nu}_0 = (\sin^2(\theta_1) - \sin^2(\theta_0)) (\vec{\mu}_1 - \vec{\mu}_0)`. 

The algorithm uses the following tricks:

#. work with projected data (to have more samples in each bin of the histogram)

