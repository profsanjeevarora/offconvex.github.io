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
In a [new paper](https://arxiv.org/abs/1802.06509) we take a different tack, and reach the counterintuitive conclusion that sometimes increasing depth can *accelerate* optimization. 

## The Effect of a Single Multiplicative Scalar

Let's begin by considering what is perhaps the simplest possible learning problem: linear regression model with $\ell_p$ loss. Below, $S$ is the training set and $y$ is the label for $x$.

$$\min_{\mathbf{w}}~L(\mathbf{w}):=\frac{1}{p}\sum_{(\mathbf{x},y)\in{S}}(\mathbf{x}^\top\mathbf{w}-y)^p$$
 
When $p>2$, we discovered an acceleration effect via simple overparametrization:  convert this linear model to an extremely simple "depth 2 net" by replacing the vector $\mathbf{w}$ with a vector $\mathbf{w_1}$ times a scalar $\omega_2$. Clearly this doesn't change the set of feasible solutions, but it makes the objective nonconvex:

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

<p style="text-align:center;">
<img src="/assets/acc_oprm/update_rule.png" width="60%" alt="End-to-end update rule" />
</p>


## Beyond Regularization

The end-to-end update rule defines an optimization scheme whose steps are a function of the gradient $\nabla{L}(W)$ and the parameter $W$.
As opposed to many acceleration methods (e.g. [momentum](https://distill.pub/2017/momentum/) or [Adam](https://arxiv.org/abs/1412.6980)) that explicitly maintain auxiliary variables, this scheme is memoryless, and by definition born from gradient descent over something (overparameterized objective).
It is therefore natural to ask if we can represent the end-to-end update rule as gradient descent over some regularization of the loss $L(W)$, i.e. over some function of $W$.
We prove, somewhat surprisingly, the answer is almost always negative - as long as the loss $L(W)$ does not have a critical point at $W=0$, the end-to-end update rule, i.e. the effect of overparameterization, cannot be attained via *any* regularizer.


## Acceleration on $\ell_p$ Regression

So far we treated the effect of depth (in the form of overparameterization) on optimization by presenting an equivalent preconditioning scheme and discussing some of its properties.
We have not, however, provided any evidence in support of acceleration (faster convergence) resulting from this scheme.
Apparently, whether or not acceleration takes place depends on the particular objective $L(W)$.
We focus in the paper on the setting of linear regression with $\ell_p$ loss, and show, via both theoretical arguments and experiments, that a speedup occurs when $p>2$.
Following is a sample result from an experiment with $\ell_2$ loss:

<p style="text-align:center;">
<img src="/assets/acc_oprm/L2_exp.png" width="40%" alt="L2 regression experiment" />
</p>

As can be seen, adding layers here slightly slowed down optimization (in line with previous observations by [Saxe et al.](https://arxiv.org/abs/1312.6120)).
In contrast, with $\ell_4$ loss depth led to significant acceleration, as demonstrated by the plot below:

<p style="text-align:center;">
<img src="/assets/acc_oprm/L4_exp.png" width="40%" alt="L4 regression experiment" />
</p>

These results correspond to a task in which the output is a scalar.
To emphasize the fact that the effect of depth on optimization need not be computationally expensive (end-to-end update rule does not depend on hidden layer widths - see above), we used hidden layers each comprising a single scalar.
A speedup by orders of magnitude was thus obtained with essentially no overhead in terms of computation or storage.
Moreover, we compared this speedup to different optimization algorithms, and found it superior to two well-known acceleration methods - [AdaGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) and [AdaDelta](https://arxiv.org/abs/1212.5701).

To the best of our knowledge, our results provide first empirical evidence for the fact that depth, even without any gain in expressiveness, and despite introducing
non-convexity to a formerly convex problem, can lead to favorable optimization, sometimes more so than carefully designed algorithms tailored for convex problems.
This obviously warrants further investigation, both theoretical and empirical.
We provide a few more experiments and discussions in the paper.


## Non-Linear Experiment

As a final sanity test, we evaluated the effect of overparameterization on optimization in a non-idealized (yet simple) deep learning setting - the [convolutional network tutorial for MNIST built into TensorFlow](https://github.com/tensorflow/models/tree/master/tutorials/image/mnist).
We introduced overparameterization by simply placing two matrices in succession instead of the matrix in each dense layer.
With an addition of roughly 15% in number of parameters, optimization accelerated by orders of magnitude:

<p style="text-align:center;">
<img src="/assets/acc_oprm/cnn_exp.png" width="40%" alt="TensorFlow MNIST CNN experiment" />
</p>

We note that similar experiments on other convolutional networks also gave rise to a speedup, but not nearly as prominent as the above.
Empirical characterization of conditions under which overparameterization accelerates optimization in non-linear settings is potentially an interesting direction for future research.


## Concluding Thoughts
