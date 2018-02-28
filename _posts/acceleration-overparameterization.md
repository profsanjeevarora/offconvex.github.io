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

Suppose we are interested in learning a linear model parameterized by a matrix $W$, through minimization of some training loss $L(W)$.
Instead of working directly with $W$, we replace it by a depth $N$ linear neural network, i.e. we overparameterize it as $W=W_{N}W_{N-1}\cdots{W_1}$, with $W_j$ being weight matrices of individual layers.
In the paper we show that if one applies gradient descent over $W_{1}\ldots{W}_N$, with small learning rate $\eta$, and with the condition:

$$W_{j+1}^\top W_{j+1} = W_j W_j^\top$$

satisfied at optimization commencement (note that this approximately holds with standard near-zero initialization), the dynamics induced on the overall input-output mapping $W$ can be written as follows:

$$W^{(t+1)}\leftarrow{W}^{(t)}-\eta\sum_{j=1}^{N}\left[W^{(t)}(W^{(t)})^\top\right]^\frac{j-1}{N}\nabla{L}(W^{(t)})\left[(W^{(t)})^\top{W}^{(t)}\right]^\frac{N-j}{N}$$

We validate empirically that this analytically derived update rule (over classic linear model) indeed complies with deep network optimization, and take a series of steps to theoretically interpret it.
We find that the transformation applied to the gradient $\nabla{L}(W)$ (multiplication from the left by $[WW^\top]^\frac{j-1}{N}$, and from the right by $[W^\top{W}]^\frac{N-j}{N}$, followed by summation over $j$) is a particular preconditioning scheme, that promotes movement along directions already taken by optimization.
More concretely, the preconditioning can be seen as a combination of two elements:
* an adaptive learning rate that increases step sizes away from initialization; and
* a "momentum-like" operation that stretches the gradient along the azimuth taken so far.

An important point to make is that the update rule above, referred to hereafter as the *end-to-end update rule*, does not depend on widths of hidden layers in the linear neural network, only on its depth ($N$).
This implies that from an optimization perspective, overparameterizing using wide or narrow networks has the same effect - it is only the number of layers that matters.
Therefore, acceleration by depth need not be computationally demanding - a fact we clearly observe in our experiments (see below).


## Beyond Regularization


## Acceleration on $\ell_p$ Regression


## Non-Linear Experiment


## Concluding Thoughts
