---
layout: post
title:  "Face Recognition: Eigenfaces"
date:   2018-11-25 21:55:55 +0200
math: true
mathjax: true
tags: [face recognition, eigenfaces, pca]
---

The main idea behind eigenfaces is that we want to learn a low-dimensional space - known as the eigenface subspace - on which we assume the faces intrinsically lie. From there, we can then compare faces within this low-dimensional space in order to perform facial recognition. It's a relatively simple approach to facial recognition, but indeed one of the most famous and effective ones of the early approaches. It still works well in simple, controlled scenarios.

Assume we have a set of $$ m $$ images $$ \{I_i\}^{m}_{i=1} $$, where $$ I_i \in \mathcal{G}^{r \times c} $$; $$ \mathcal{G} = \{0, 1, \dots, 255\} $$; and $$ r \times c $$ is the spatial dimension of the image. The first step to the algorithm is to resize all the images in the set to the same size. Typically, the images are converted to grayscale, since it is assumed that colour is not an important factor. This is clearly debatable, however, for the purposes of this post we will assume that the images are grayscale images.

Each image is then converted to a vector, by appending each row into one long vector. Given an image from the set, we convert it to a vector $$ \Gamma_i \in \mathcal{G}^{rc} $$.

We then calculate the mean face $$ \Psi $$:

$$ \Psi = \frac{1}{m} \sum_{i=1}^{m} \Gamma_i $$

We then zero-centre the image vectors $$ \Gamma_i $$ by subtracting the mean from each. This results in a set of vectors $$ \Phi_i $$:

$$ \Phi_i = \Gamma_i - \Psi $$

We then perform PCA on the matrix $$ A $$, where $$ A $$ is given by:

$$ A = [\Phi_1 \Phi_2 \cdots \Phi_m] \in \mathbb{R}^{rc \times m} $$

More concretely, we compute the covariance matrix $$ C \in \mathbb{R}^{rc \times rc} $$:

$$ C = \frac{1}{m} \sum_{i=1}^m \Phi_i \Phi_i^T = A A^T $$

We would then typically compute the eigen decomposition of this matrix. However, in the interest of speed, the eigen decomposition is instead computed for $$ A^T A \in \mathbb{R}^{m \times m} $$. This is mathematically justified since the $$ m $$ eigenvalues of $$ A^T A $$ (along with their associated eigenvectors) correspond to the $$ m $$ largest eigenvalues of $$ A A^T $$ (along with their associated eigenvectors).

We then retain the first $$ k $$ principal components: the $$ k $$ eigenvectors with largest associated absolute eigenvalues. This corresponds to a matrix $$ V \in \mathbb{R}^{m \times k} $$, where the columns of the matrix are these chosen eigenvectors. We then compute the so-called projection matrix $$ U \in \mathbb{R}^{rc \times k} $$:

$$ U = A V $$

Lastly, we can finally find the eigenface subspace $$ \Omega \in \mathbb{R}^{k \times m} $$:

$$ \Omega = U^T A $$

Now, for the actual facial recognition part! Consider a resized grayscale test image $$ I \in \mathcal{G}^{r \times c} $$. We reshape this into a vector:

$$ \Gamma \in \mathcal{G}^{rc \times 1} $$

We then mean-normalise:

$$ \Phi = \Gamma - \Psi $$

Finally, we project the test face onto the eigenface subspace (i.e. the linear manifold learned by PCA):

$$ \hat{\Phi} = U^T \Phi $$

Given this projected face, we can find which face it is closest to in the eigenface subspace, and classify it as that that person's face:

$$ \text{prediction} = \argmin_{i} ||\Omega_i - \hat{\Phi}||_2 $$

where $$ \Omega_i $$ is the $$ i^{\text{th}} $$ face in the eigenface subspace. It is clear that this is using Euclidean distance, as this is the metric used in the classical eigenface algorithm. We can, however, instead opt for $$ L_1 $$ distance or any other distance metric.