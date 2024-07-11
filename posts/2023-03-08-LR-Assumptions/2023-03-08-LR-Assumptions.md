---
categories:
- causal-inference
- exposition
- machine-learning
date: '2023-03-08'
description: Two interesting questions about linear regression
title: The Assumptions We Take for Granted
toc: true

---

> Acknowledgement: content in this post come from discussions with [Xuan Bi](https://sites.google.com/site/xuanbigts). Question 1 below is adapted from an interview question brought to my attention by Yuxuan Yang (MSBA Class of 2023).

In this blog post, I will discuss two interesting questions related to linear regression, one focusing on the coefficient estimation aspect (e.g., for statistical inference) and the other focusing on the prediction aspect (e.g., as a numeric prediction model). A central theme underlying the answers to both questions is an assumption behind linear regression that is often taken for granted.

# Question 1: Estimating Regression with Replicating Data

You have a sample of $N$ data points $(x_i,y_i)_{i=1}^N$, with which you want to estimate the following linear regression model

$$
Y=\beta x + \varepsilon
$$

The intercept term is omitted for simplicity (e.g., imagine the data has been centered). For this question, suppose that none of the "usual suspects" that might threaten the estimation exist, i.e., there is no endogeneity whatsoever. Denote the coefficient estimate as $\widehat{\beta}^{(N)}$. Now imagine you make a copy of every single data point, thereby creating a sample of $2N$ data points. Then estimate the same regression on this twice-as-large sample and denote the coefficient estimate as $\widehat{\beta_1}^{(2N)}$. **The question is:** what is the relationship between $\widehat{\beta_1}^{(N)}$ and $\widehat{\beta}^{(2N)}$, and between $SE(\widehat{\beta}^{(N)})$ and $SE(\widehat{\beta}^{(2N)})$?

To answer this question, one could (mechanically) follow the least-square formula for coefficient estimate and its standard error and discover that

$$
\widehat{\beta}^{(2N)} = \widehat{\beta}^{(N)}, ~~~ SE(\widehat{\beta}^{(2N)})=\frac{1}{\sqrt{2}}SE(\widehat{\beta}^{(N)})
$$

In other words, estimating a regression on twice the data still gives the same coefficients but smaller standard errors. The same conclusion can be easily verified via a simulation.

However, if you think a bit deeper, this answer is really weird. Just by making a copy of every single data point in a sample, we are able to "magically" reduce the standard error of coefficient estimates. Put differently, without bringing in *any* new information, we can somehow make our estimates more precise. This cannot be possible - taking this logic to its extreme, we can simply make multiple copies of every data point and achieve arbitrarily precise estimates from essentially finite data.

What has gone wrong? The answer is hidden in an assumption of linear regression that is often taken for granted, namely the **i.i.d assumption**. The i.i.d assumption states that the data points in the sample to estimate regression (1) are assumed to be *independently and identically distributed*. This assumption is clearly violated in the $2N$ sample, because a duplicating pair of data points are not independent by definition. Meanwhile, when mechanically running the linear regression on the $2N$ sample (or mechanically applying the least-square formula with the twice-as-large sample), we have *pretended* that data points in the $2N$ sample are actually i.i.d. The (spurious) standard error reduction comes from the fact that *if* data in the $2N$ sample are truly i.i.d., then you are indeed working with additional information, which is the fact that each unique data point magically has an exact repeated observation drawn independently from the same underlying population. 

# Question 2: Generalization Error of Linear Regression

Next, let's consider a numeric prediction problem and treat the linear regression as a predictive model trained on the labeled sample $(x_i,y_i)$. Any standard statistical learning textbook would tell us that the generalization error (i.e., expected out-of-sample $L_2$ loss) of such a model can be decomposed into the sum of bias and variance:

$$
\mathbb{E}(\widehat{Y}-Y)^2=(\mathbb{E}(\widehat{\beta}) - \beta)^2 + Var(\widehat{\beta}) + \sigma^2
$$

where $\sigma^2=Var(\varepsilon)$. Because the regression is unbiased (free from any endogeneity issue), the generalization error simplifies to $Var(\widehat{\beta}) + \sigma^2$. Suppose we train two linear regression models, one on the $N$ sample and another on the (duplicated) $2N$ sample. Based on this generalization error expression, the model trained on $2N$ sample would have smaller $Var(\widehat{\beta})$ and therefore smaller generalization error. In other words, we are able to improve the (expected out-of-sample) predictive performance of a linear regression model simply by duplicating its training data. However, this conclusion is also preposterous - in fact, because the coefficient estimate stays the same (as we discussed in Question 1), the regression line doesn't change at all across the two samples, and therefore would give the exact same predictions for any data point.

By now it should be clear what has gone wrong. The variance reduction on $2N$ sample is wishful thinking. It comes from implicitly assuming the i.i.d condition when in fact it is clearly violated by the way we construct the $2N$ sample. In conclusion, simply by making copies of data points in a sample has no meaningful effect on either the estimation of regression coefficients or the regression model's predictive ability. 

