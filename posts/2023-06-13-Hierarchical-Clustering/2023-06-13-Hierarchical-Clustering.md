---
toc: true
categories: 
- exposition
- machine-learning
date: '2023-06-13'
title: Cluster Distance in Hierarchical Clustering
description: An In-Depth Discussion of Different Linkage Methods and the Lance–Williams Algorithm
---
# Motivation

Hierarchical clustering is one of the most commonly used algorithms to discover clusters from data. It works by iteratively merging smaller clusters that are closest to each other into bigger ones. A key ingredient to hierarchical clustering is a metric to quantify the distance between two clusters (which is different from the measure of distance between to individual data points). A number of such cluster distance metrics (a.k.a. linkage methods) are available, such as single linkage, complete linkage, average linkage, centroid distance, and Ward's method (see [this wiki page for many more](https://en.wikipedia.org/wiki/Hierarchical_clustering#Cluster_Linkage)).

During the process of hierarchical clustering, as clusters are being merged, there is the need to compute the distance between the a newly merged cluster and all the other clusters (or data points), in order to find the next two clusters to merge. Of course, one could simply maintain the raw data and re-compute all pair-wise distances among all clusters every time after a merge happens. This is, however, computationally quite wasteful because, at the very least, pair-wise distances among clusters that are not newly merged do not need to be computed again. 

With computational efficiency in mind, it turns out that for a large collection of linkage methods, you do not even need to maintain the raw data. Calculate the pair-wise distance among all individual data points once to get the distance matrix, and that's all you need moving forward. In fact, the ```hclust``` function in ```R```, for example, explicitly takes distance matrix (not raw data) as input. 

How does this work? In particular, how can we update the distance matrix once two (smaller) clusters are merged, to properly reflect the cluster distance between the newly formed cluster and the rest of data? The goal of this blog is to answer this question, through which we will get to know the [Lance–Williams Algorithm](https://en.wikipedia.org/wiki/Ward%27s_method#Lance%E2%80%93Williams_algorithms), and also clarify some (often omitted) technical details about the centroid and Ward's methods. 

# Notation

Consider three clusters, $I$, $J$, and $K$, where clusters $I$ and $J$ are to be merged in the current iteration of hierarchical clustering, and we want to be able to calculate the distance between the newly formed (bigger) cluster $IJ$ and cluster $K$. Note that, for the purpose of this discussion, we don't really need to differentiate between clusters and individual data points (i.e., can safely treat a single data points as a cluster of size 1). Denote $n_I$, $n_J$, and $n_K$ as the sizes of the three clusters, respectively. Let $i \in I$, $j \in J$, and $k \in K$ index individual data within the three clusters, and $d()$ denote a chosen distance measure between individual data points (Euclidean, Manhattan, Matching, etc.). With (slight) abuse of notation, I will also use $d()$ to represent distance between clusters. For example $d(I,J)$ will denote the distance between clusters $I$ and $J$, bearing in mind that the way $d(I,J)$ is computed will depend on the specific choice of linkage method.

# Lance-Williams Algorithm

The Lance-Williams algorithm says that, for a large collection of linkage methods, the distance $d(IJ, K)$ can be expressed as a recursive expression of $d(I, K)$, $d(J, K)$, and $d(I,J)$, as such:

$$
d(IJ,K) = \alpha_i d(I,K) + \alpha_j d(J, K) + \beta d(I,J) + \gamma |d(I,K) - d(J,K)|
$$

where  specific to the choice of linkage method (see [this paper](https://doi.org/10.2307/2344237) for a list of parameter values for common linkage methods). This is pretty impressive (and useful), because it allows us to easily compute the distance between a merged cluster with other clusters using (already computed and available) inter-cluster distances. Cool, but why does it work?

## Illustration for Single / Complete / Average Linkage

The $\alpha,\beta,\gamma$ parameters values for single, complete, and average linkage methods are:

| Linkage Method   | $\alpha_1$      | $\alpha_2$      | $\beta$ | $\gamma$ |
| ---------------- | --------------- | --------------- | ------- | -------- |
| Single Linkage   | $1/2$           | $1/2$           | $0$     | $-1/2$   |
| Complete Linkage | $1/2$           | $1/2$           | $0$     | $1/2$    |
| Average Linkage  | $n_I/(n_I+n_J)$ | $n_J/(n_I+n_J)$ | $0$     | $0$      |

Let's start with the single linkage method to illustrate why this is true. Recall that single linkage uses the nearest neighbors between the two clusters as the cluster distance, i.e., $d(I,J) = \min_{i \in I, j \in J} d(i,j)$. The RHS of the Lance-Williams equation is

$$
1/2 d(I,K) + 1/2 d(J,K) - 1/2 |d(I,K) - d(J,K)|
$$

If $d(I,K) > d(J,K)$, the above simplifies to $d(J,K)$, and if $d(I,K) < d(J,K)$, it simplifies to $d(I,K)$. In other words, the above expression is equivalent to $\min\{d(I,K), d(J,K)\}$, which, by definition of single linkage, is exactly $d(IJ, k)$. The same derivation will show you that the parameter values for complete linkage are also correct.

Now, for average linkage, we only need to notice that, by definition, $d(I,K) = \frac{\sum d(i,k)}{n_I \cdot n_K}$ and $d(J,K) = \frac{\sum d(j,k)}{n_J \cdot n_K}$. Simply plugging in the parameters will show you that it works as intended.

## Illustration for Centroid and Ward's Method

The more (nuanced) cases are methods like centroid and Ward's, which relies on the concept of "centroid" (i.e., geometric mean of a cluster). Their parameter values are as follows (now far from obvious):

| Linkage Method    | $\alpha_1$                | $\alpha_2$                | $\beta$                | $\gamma$ |
| ----------------- | ------------------------- | ------------------------- | ---------------------- | -------- |
| Centroid Distance | $n_I/(n_I+n_J)$           | $n_J/(n_I+n_J)$           | $-n_I n_J/(n_I+n_J)^2$ | $0$      |
| Ward's Method     | $(n_I+n_K)/(n_I+n_J+n_K)$ | $(n_J+n_K)/(n_I+n_J+n_K)$ | $-n_K/(n_I+n_J+n_K)$   | $0$      |

Here I will derive this result for the centroid distance (and similar derivation can be done for the Ward's method). First, let $C_I$, $C_J$, and $C_K$ denote the centroids of the three initial clusters -- they are single points just like individual data. Note that, by definition of centroid, $C_{IJ} = \frac{n_I c_I + n_J c_J }{n_I+n_J}$. 

For now, let's assume a Squared Euclidean distance metric (the reason is not clear at all at this point, but stay with me for now), meaning that for any two data points with coordinates $x=(x_1, \ldots, x_M)$ and $y=(y_1, \ldots, y_M)$, we have $d(x,y) = \sum_{m=1}^M (x_m - y_m)^2$ where $m$ indexes each one of the $M$ features. A nice thing about this metric is that the squared difference on each feature is fully additive. So, we don't need to carry around the summation over all features -- we can just need to work (symbolically) with $(x-y)^2$. 

Next, plug in the parameter values for the RHS of Lance-William, we get:

$$
n_I/(n_I+n_J) (C_I - C_K)^2 + n_J/(n_I+n_J) (C_J - C_K)^2 -n_I n_J/(n_I+n_J)^2 (C_I - C_J)^2
$$

Open up all the squares and re-arrange the terms, we will eventually see that it indeed equals $d(IJ,K)$, which is 

$$
\left(\frac{n_I c_I + n_J c_J }{n_I+n_J} - C_K \right)^2
$$

However, the important thing to notice is that the above derivation **only works** when the underlying distance metric is Squared Euclidean. This is why, when using centroid or Ward's method for cluster distance, one should always pick Euclidean distance as the metric to measure distance between data points (then the software implementations will square those distances when performing Lance-Williams, see [this](https://github.com/scipy/scipy/blob/v1.10.1/scipy/cluster/_hierarchy_distance_update.pxi) as and example of how Scipy does it). In fact, the notation of "centroid" only really make sense in a Euclidean space. Technically, you can still adopt centroid / Ward's methods with non-Euclidean distance measures, but the price you pay is that the Lance-Williams recursive relationship (which makes computation much easier) would no longer hold, and you need to re-compute cluster distances from raw data after every merge.
