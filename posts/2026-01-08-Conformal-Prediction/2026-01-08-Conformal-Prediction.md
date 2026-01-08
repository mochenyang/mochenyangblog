---
categories:
- causal-inference
- machine-learning
- exposition
date: '2026-01-08'
description: Learning Notes
title: Conformal Prediction
toc: true

---

> This is my learning notes about conformal prediction, based on [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/pdf/2107.07511) by Anastasios N. Angelopoulos and Stephen Bates.

# The Question
Imagine a black-box machine learning model (and for concreteness, consider a multi-class classification model). It produces prediction probabilities over possible classes (which can be turned into class predictions if needed). These probabilities _may be_ an indicator of the model's uncertainty, but there is no inherent guarantee that the prediction probabilities are any good, in the sense that they reflect any sort of statistical reality. After all, the model is not explicitly trained to produce valid probability estimates, and its complexity (e.g., in the case of deep neural nets) precludes precise characterization of its statistical properties. Given such a model, is it possible to obtain uncertainty quantification on the model's predictions with statistical guarantees?

This may sound hopeless at first -- we cannot take the model's prediction probabilities at face value, nor can we open up the black box and analyze the model itself (like with a simple linear regression). Because of these challenges, it seems very impressive that we _can_ do something, with the machinery called **conformal prediction**.

# How Does Conformal Prediction Work
For a new (unlabeled) data instance, the goal of conformal prediction is to generate a _prediction set_ (that is, a set of "plausible" classes that the data may belong to) that, in expectation, is guaranteed to contain the true class with a user-specified confidence level (i.e., guaranteed "coverage" in expectation).

Conformal prediction requires three ingredients:

1. A user-specified **confidence level** $\alpha$ (e.g., $\alpha = 0.05$ means user wants coverage of 95\%).
2. A **calibration dataset** $\{(X_1, y_1), \ldots, (X_n, y_n)\}$ with known true labels (i.e., a random partition of labeled data).
3. A **conformal score function** $s(X,y)$ that indicates the _disagreemnt_ between $X$ and $y$. In other words, it returns a larger value when the model's is more confidently wrong about its prediction. For classification tasks, one natural choice is $s(X,y) = 1 - PredProb(y)$, where $PredProb(y)$ stands for the predicted probability assigned to true label.

The procedure to obtain conformal prediction for a new data instance with input $X_{new}$ is pretty simple:

1. Compute the conformal scores for each calibration data: $s(X_1, y_1), \ldots, s(X_n, y_n)$. This would form an empirical distribution of conformal scores.
2. Take the $1-\alpha$ quantile of this distribution, denote it as $\widehat{q}$.[^1]
3. For the new instance $X_{new}$, iterate through all possible classes, and include a candidate class $y_{candidate}$ if and only if the corresponding conformal score $s(X_{new}, y_{candidate})$ is not greater than $\widehat{q}$. More formally, the conformal prediction set is $C(X_{new}) = \{y_{candidate}: s(X_{new}, y_{candidate}) \leq \widehat{q} \}$.

That's it. This guarantees that, in expectation, the prediction set contains the true class for $X_{new}$ with a probability of at least $1-\alpha$ and at most $1-\alpha + \frac{1}{n+1}$ (that is, the coverage is almost exactly $1-\alpha$).

# Why Does Conformal Prediction Work
The amazing thing here is that the coverage property does not depend on the inner working of the model or data distribution. And the obvious question is how this is possible -- it feels too good to be true to get distribution-free and model-agnostic uncertainty quantification "almost for free". Here, I provide some rough intuitions and simulation validations.

## Intuition
The "secret sauce" lies in the calibration dataset. By applying the conformal score function on the calibration set, we essentially obtain the empirical distribution of conformal scores. And because calibration set is a random sample taken from the population, we expect the empirical distribution to approximate what the actual conformal score distribution would look like. Next, because the new data instance is yet another random sample from the same population, we expect its conformal score to follow the same conformal score distribution. Therefore, using nothing but the empirical quantile from the conformal scores, we can locate all "plausibly correct" classes for the new data instance, and include them in the prediction set. Based on this intuition, it's obviously important to have a sufficiently large calibration set, which sharpens the coverage bound (specifically bringing down the upper bound).

Moreover, the quality of the conformal score function is also crucial. Essentially, the conformal score function acts as a "heuristic uncertainty score", and the conformal prediction process "refines" those scores to formulate statistically valid prediction sets. The more informative the conformal scores are, the more informative the prediction sets would be (manifested as having tighter prediction sets). The following simulations confirm this.

## Simulation
I first simulate the prediction probabilities that would be generated by a black-box machine learning model for a 10-class classification task. The true class is assigned a higher probability than other classes.

```R
# simulate a black-box model's predicted probs
# among all classes, randomly pick one and assign a high score, then randomly assign low score for all other classes
simulate_predicted_probs = function(n_instances, n_classes) {
  probs <- matrix(runif(n_instances * n_classes), nrow = n_instances, ncol = n_classes)
  for (i in 1:n_instances) {
    high_class = sample(1:n_classes, 1)
    probs[i, high_class] = 0.5
  }
  probs <- probs / rowSums(probs)  # normalize to sum to 1
  return(probs)
}
```

Next, the conformal prediction procedure can be implemented with just a few lines of code.
```R
# implement conformal prediction, returns prediction set
conformal_prediction = function(calibration_scores, new_scores, alpha) {
  all_classes = c(1:n_class)
  probs_adj = ceiling((N_calibration + 1) * (1 - alpha)) / N_calibration
  if (probs_adj >= 1) {
    return(all_classes)
  } else {
    q_hat = quantile(calibration_scores, probs = probs_adj)
    return(all_classes[new_scores <= q_hat])
  }
}
```

To demonstrate the impact of conformal score quality on the "strength" of conformal prediction, I simulate a "good" score function and a "bad" score function. The good score function returns 1 minus the prediction probability of true class (i.e., an informative signal), whereas the bad score function returns 1 minus the prediction probability of a random class.

```R
# a "good" conform score function, 1 - prob(true class)
# pred_probs should be a vector of length n_class and true_label should be an integer index
good_conformal_score = function(pred_probs, true_label) {
  return(1 - pred_probs[true_label])
}

# a "bad" conform score function, 1 - prob(random class)
bad_conformal_score = function(pred_probs) {
  random_class = sample(1:n_class, 1)
  return(1 - pred_probs[random_class])
}
```

Finally, the following code set up the global variables and run the simulation. I report two metrics -- (1) the coverage rate (averaged over all new data instances) and (2) the average size of prediction sets (as a measure of "sharpness").[^2]

```R
# assuming there are 10 classes in total
# assuming there are 1000 calibration data instances and 1000 unlabeled instances
n_class = 10
N_calibration = 1000
N_new = 1000
alpha = 0.05
set.seed(123456)

# simulate pred probs for calibration set and new data
pred_probs_calibration = simulate_predicted_probs(N_calibration, n_class)
pred_probs_new = simulate_predicted_probs(N_new, n_class)
# simulate true labels as the one that received the highest predicted prob
true_labels_calibration = apply(pred_probs_calibration, 1, which.max)
true_labels_new = apply(pred_probs_new, 1, which.max)

# compute the calibration scores under both good and bad score functions
calibration_scores_good = c()
calibration_scores_bad = c()
for (i in 1:N_calibration) {
  calibration_scores_good[i] = good_conformal_score(pred_probs_calibration[i, ], true_labels_calibration[i])
  calibration_scores_bad[i] = bad_conformal_score(pred_probs_calibration[i, ])
}

# for each new data instance, compute the prediction sets, and record the set size and coverage respectively
set_sizes_good = c()
set_sizes_bad = c()
coverage_good = c()
coverage_bad = c()
for (i in 1:N_new) {
  new_scores = 1 - pred_probs_new[i, ]
  pred_set_good = conformal_prediction(calibration_scores_good, new_scores, alpha)
  pred_set_bad = conformal_prediction(calibration_scores_bad, new_scores, alpha)
  set_sizes_good[i] = length(pred_set_good)
  set_sizes_bad[i] = length(pred_set_bad)
  coverage_good[i] = ifelse(true_labels_new[i] %in% pred_set_good, 1, 0)
  coverage_bad[i] = ifelse(true_labels_new[i] %in% pred_set_bad, 1, 0)
}

# print(paste0("Good score function: Average prediction set size = ", mean(set_sizes_good), 
#              ", Coverage = ", mean(coverage_good)))
# print(paste0("Bad score function: Average prediction set size = ", mean(set_sizes_bad), 
#              ", Coverage = ", mean(coverage_bad)))

# output should be 
# "Good score function: Average prediction set size = 2.252, Coverage = 0.958"
# "Bad score function: Average prediction set size = 9.596, Coverage = 1"
```

It's clear that both a good and a bad score function guarantee proper (lower-bound) coverage, but the prediction sets from the good score function are substantially more useful because they are small. In contrast, the prediction sets under the bad score function almost always just include all classes (which is useless).

[^1]: Technically, one takes the $\frac{\lceil (n+1)(1-\alpha) \rceil}{n}$ quantile. The small adjustment is explained in the original article.

[^2]: To be fully rigorous, the averaging should be taken over both new data instances and (repeated sampling of) calibration instances. This is simplified.