---
layout: post
title: Benefits of Depth for Optimization
date: 
author:
visible: True
---


Among the most fundamental questions in the theory of deep learning is the role of depth.
Conventional wisdom, backed by multiple theoretical studies, states that adding layers to a network boosts its expressive power.
On the other hand, it is generally believed that this expressive gain comes at a price - optimization (training) of deeper networks is more difficult.
This belief is somewhat hardcoded into the "landscape characterization" approach for analyzing deep network optimization.
Papers following this approach typically study local minima and/or saddle points in the objective of a deep network, while implicitly assuming that the ideal landscape is convex (single global minimum, no other critical point).
In a [new paper](https://arxiv.org/abs/1802.06509) we take a different tack, and reach the counterintuitive conclusion that depth can actually *accelerate* optimization, even in cases where the original problem was simple and convex.


## The Effect of a Single Multiplicative Scalar

Let us begin by considering what is perhaps the simplest possible instance of depth - addition of a single multiplicative scalar.
Suppose we would like to learn a linear regression model by optimizing $\ell_p$ loss over a training set $S$:

$$\min_{\mathbf{w}}~L(\mathbf{w}):=\frac{1}{p}\sum_{(\mathbf{x},y)\in{S}}(\mathbf{x}^\top\mathbf{w}-y)^p$$

We may convert the linear model to an extremely simple "deep network" by replacing the vector $\mathbf{w}$ with a vector $\mathbf{w_1}$ times a scalar $\omega_2$:

$$\min_{\mathbf{w_1},\omega_2}~L(\mathbf{w_1},\omega_2):=\frac{1}{p}\sum_{(\mathbf{x},y)\in{S}}(\mathbf{x}^\top\mathbf{w_1}\omega_2-y)^p$$

It is not difficult to show (details in paper) that if one applies gradient descent over $\mathbf{w_1}$ and $\omega_2$, with small learning rate and near-zero initialization (as customary in deep learning), the induced dynamics on the overall model $\mathbf{w}=\mathbf{w_1}\omega_2$ can be written as follows:

$$\mathbf{w}^{(t+1)}\leftarrow\mathbf{w}^{(t)}-\rho^{(t)}\nabla{L}(\mathbf{w}^{(t)})-\sum_{\tau=1}^{t-1}\mu^{(t,\tau)}\nabla{L}(\mathbf{w}^{(\tau)})$$

where $\rho^{(t)}$ and $\mu^{(t,\tau)}$ are appropriately defined (time-dependent) coefficients.
The seemingly benign addition of a single multiplicative scalar thus turned plain gradient descent into a scheme that involves a time-varying learning rate and a certain momentum term.
This obviously does not mean that training will accelerate, but it does demonstrate the immense impact that depth, even in its simplest form, can have on optimization.
Below we discuss our general acceleration analysis.
As a teaser, here are results of an experiment evaluating this exact setting, with $p=4$:

<p style="text-align:center;">
<img src="/assets/acc_oprm/warmup_exp.png" width="40%" alt="Acceleration through addition of a single multiplicative scalar" />
</p>


## Overparameterization: Decoupling Optimization from Expressiveness

Studying the effect of depth on optimization entails an inherent difficulty - deeper networks may seem to converge faster due to their superior expressiveness.
In other words, if optimization of a deep network progresses more rapidly than that of a shallow one, it may not be obvious whether this is a result of a true acceleration phenomenon, or simply a byproduct of the fact that the shallow model cannot reach the same loss as the deep one.
We resolve this conundrum by focusing on models whose representational capacity is oblivious to depth - *linear neural networks*, the subject of many recent studies.
With linear networks, adding layers does not alter expressiveness; it manifests itself only in the replacement of a matrix parameter by a product of matrices - an *overparameterization*.
Accordingly, if this leads to accelerated convergence, one can be certain that it is not an outcome of any phenomenon other than favorable properties of depth for optimization.


## Implicit Dynamics of Depth


## Beyond Regularization


## Acceleration on $\ell_p$ Regression


## Non-Linear Experiment


## Concluding Thoughts
