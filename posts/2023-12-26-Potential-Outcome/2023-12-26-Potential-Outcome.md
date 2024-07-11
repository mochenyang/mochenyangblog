---
toc: true
categories:
- exposition
- causal-inference
date: '2023-12-26'
title: Common Identification Strategies and Program Evaluation
description: Connections between linear regression, IV, DID and ATE, LATE, ATT estimations
---
# Basic Setup
The potential outcome framework is arguably one of the main theorecial building blocks of causal inference. In program evaluation (i.e., a term often used in econometrics to refer to the evaluation / estimation of the "effect" of certain treatment or program), people frequently talk about quantities such as **ATE** (average treatement effect), **ATT** (average treatment effect on the treated), **LATE** (local average treatment effect), and also connect them with identification strategies, respectively linear regression, DID regression, and IV regression. The purpose of this blog post is to clarify these connections by providing an explanation for each.

For basic setup, let's index individuals (often the unit of analysis in a given estimation task) by $i$, and use $Y_i(d)$, where $d \in \{0,1\}$, to denote the potential outcomes if $i$ was untreated or treated, respectively. A key idea in the potential outcome framework is that each individual, regardless of which group they are actually assigned to, fundamentally has a potential outcome for each condition. In reality, of course, only one of the potential outcomes can be observed (otherwise treatment effect estimation would have been trivial, simply $Y_i(1) - Y_i(0)$). Because of this partial observability, it makes sense to also keep track of treatment assignment with a variable $D_i \in \{0,1\}$. The goal of program evaluation is to estimate the effect of treatment (more formal definitions to come) using these partially observed outcomes.

# Linear Regression and ATE
Starting with the simplest and cleanest scenario, where treatment is randomly assigned to individuals and each individual fully comply with the assigned treatment (think about a clinical trial where test pills or placebo pills are directly fed to volunteers'). Using our notations, this simply means that $\forall i$, the potential outcome $Y_i(D_i)$ is observed.

A standard approach of estimating treatment effect is via a linear regression (equivalent to a $t$-test if there is no other covariates):

$$
Y_i(D_i) = \beta_0 + \beta_{LR} D_i + \varepsilon_i
$$

The interpretation of $\beta_{LR}$ is straightforward: $\beta_{LR} = \mathbb{E}(Y_i(D_i) \vert D_i = 1) - \mathbb{E}(Y_i(D_i) \vert D_i = 0) = \mathbb{E}(Y_i(1) - Y_i(0)).$ This quantity is **ATE** by definition, which can be readily estimated by a linear regression in a randomized experiment without non-compliance.

# IV Regression and LATE
Of course, not all settings are that clean. A common issue that arises in program evaluation is non-compliance. Think about another clinical trial where test pills or placebo pills are given to volunteers to take home. Compliers would take their intended pills and realize $Y_i(D_i)$, whereas non-compliers may, for example, simply forget to take their pills (in which case $Y_i(0)$ would be realized regardless of assigned conditions). Worse yet, individuals may decide to take the pills or not based on their perceived benefits, which can break the intended randomization.

Given the assigned treatment $D_i$, the _actual_ treatment status of individual $i$ depends on that particular individual's own choice (e.g., whether $i$ decides to swallow the pill or not), which subsequently determines the realized outcome. This extra layer of uncertainty that "assigned treatment may not equal to received treatment" is why program evaluation with potential non-compliance is a mind-twister. To keep track of things clearly, let's use $W_i(D_i) \in \{0,1\}$ to denote the actual treatment status of $i$, and $Y_i(W_i(D_i))$ to denote the realized outcome. Although the notation is a bit cumbersome, it has the advantage of clarity.

Although $D_i$ may be randomly assigned, $W_i(D_i)$ is not, and therefore regressing $Y_i(W_i(D_i))$ on $W_i(D_i)$ is no longer a reliable way to estimate treatment effect. However, $D_i$ naturally serves as a valid instrumental variable for $W_i(D_i)$, and we can tease out a form of treatment effect estimate via 2SLS. Standard 2SLS argument gives the so-called "wald estimator":

$$
\beta_{IV} = \frac{\mathbb{E}(Y_i(W_i(D_i)) | D_i = 1) - \mathbb{E}(Y_i(W_i(D_i)) | D_i = 0)}{\mathbb{E}(W_i(D_i) | D_i = 1) - \mathbb{E}(W_i(D_i) | D_i = 0)}
$$

But what does this mean? To get an intuitive understanding, the following table helps.

|              | $W_i(D_i)$ | $Y_i(W_i(D_i))$ |
|--------------|----------|---------------|
| Complier     |    $D_i$      |        $Y_i(D_i)$       |
| Never-Taker  |      $0$    |       $Y_i(0)$        |
| Defier       |     $1 - D_i$     |       $Y_i(1-D_i)$        |
| Always-Taker |    $1$      |      $Y_i(1)$         |

Let's start with the denominator: $\mathbb{E}(W_i(D_i) \vert D_i = 1)$ is simply the proportion of individuals who actually received treatment among those who were assigned treatment. Based on the above table, it is the compliers plus the always-takers. Similarly, $\mathbb{E}(W_i(D_i) \vert D_i = 0)$ is the proportion of individuals who would receive treatment even if they were assigned to control. It is the defiers plus the always-takers. Under the common assumption that there is no defier, the denominator reflects the proportion of compliers, i.e., individuals who received treatment _only because_ they were assigned to the treatment group.

By the same logic (and with the help of the table), the numerator reflects the expected outcome change associated with compliers as $D_i$ changes from 0 to 1. Therefore, the division of the two then becomes the treatment effect conditional on compliers, i.e., $\mathbb{E}(Y_i(1) - Y_i(0) \vert i \in \text{Complier})$. This quantity is **LATE**, as it measures the treatment effect locally, for the complier group. Of course, this is not a rigorous proof, but you can find one in many econometrics textbooks / lecture materials, such as [this one](https://econ.lse.ac.uk/staff/spischke/ec533/The%20LATE%20theorem.pdf).

# DID Regression and ATT
What if there is no non-compliance, but the treatment is not randomly assigned? In the absence of a randomized experiment, we generally cannot hope to estimate treatment effect with a linear regression of outcome on (non-random) treatment. Sometimes, however, we find ourselves in a quasi-experimental setting where the treatment manifest as a "shock" in time (e.g., introduction of some new features on a platform), and affects some individuals while others remain untreated. This two-group two-period setting is suitable for a DID regression.

In a typical (panel) DID setup, there is a time indicator $T \in \{0,1\}$ that marks "before" vs. "after" the shock, and a treatment indicator $D_i \in \{0,1\}$ defined the same as before. $Y_{i,T}(D_i)$ therefore reflects the potential outcomes of individual in period $T$ with treatment status $D_i$. By convention, $Y_{i,0}(D_i)$ are often called pre-treatment outcomes and $Y_{i,1}(D_i)$ post-treatment outcomes.

The standard DID regression takes the following form (a.k.a a two-way fixed-effect regression):

$$
Y_{i,T}(D_i) = \beta_0 + \beta_1 D_i + \beta_2 T + \beta_{DID} D_i \times T + \varepsilon_i
$$

where $\beta_1 D_i$ and $\beta_2 T$ respectively account for individual-specific and period-specific unobserved factors that may have affected treatment assignment, and $\beta_{DID}$ is the coefficient of interest. 

Again, what does $\beta_{DID}$ measure here? As the intuition of "diff-in-diff" goes, it might seem that

$$
\beta_{DID} = [\mathbb{E}(Y_{i,1}(1)) - \mathbb{E}(Y_{i,0}(1))] - [\mathbb{E}(Y_{i,1}(0)) - \mathbb{E}(Y_{i,0}(0))]
$$

However, this is not entirely accurate. Note that the term in the first $[.]$ can only be estimated among individuals in the treated group (who are affected by the treatment shock), and the term in the second $[.]$ can only be estimated among individuals in the control group (who are not affected by the treatment shock).  So, more precisely:

$$
\beta_{DID} = [\mathbb{E}(Y_{i,1}(1) - Y_{i,0}(1) | i \in \text{Treated})] - [\mathbb{E}(Y_{i,1}(0) - Y_{i,0}(0) | i \in \text{Control})]
$$

But this is not very satisfactory. For each particular $i$, it either contributes to the estimation of the first term or the second term, but not both. As far as "treatment estimation" goes, we ideally want to understand the effect on the $i$, imagine if it was treated vs. not treated. This is where the _parallel trend assumption_ comes in, which asserts that treated and control individuals are "similar" in the absence of treatment. Mathematically, it means

$$
\mathbb{E}(Y_{i,1}(0) - Y_{i,0}(0) | i \in \text{Control}) = \mathbb{E}(Y_{i,1}(0) - Y_{i,0}(0) | i \in \text{Treated})
$$

This assumption says that, suppose the shock never happened (i.e., in the absence of treatment), then the cross-period change in outcome should (in expectation) be the same regardless of whether an individual was assigned to the treatment group or the control group. In other words, the shock is the only reason for any outcome divergence between treated and control individuals. In practice, this assumption is often tested by comparing the observed pre-treatemnt outcomes between treated and control individuals.

With this assumption, we can re-write $\beta_{DID}$ as

$$
\begin{align*}
\beta_{DID} & = [\mathbb{E}(Y_{i,1}(1) - Y_{i,0}(1) | i \in \text{Treated})] - [\mathbb{E}(Y_{i,1}(0) - Y_{i,0}(0) | i \in \text{Treated})] \\
& = [\mathbb{E}(Y_{i,1}(1) - Y_{i,1}(0) | i \in \text{Treated})] - [\mathbb{E}(Y_{i,0}(1) - Y_{i,0}(0) | i \in \text{Treated})]
\end{align*}
$$

The second expectation term equals 0 because, at time period $T = 0$, the treatment hasn't taken place yet. So, in the end, we have

$$
\beta_{DID} = \mathbb{E}(Y_{i,1}(1) - Y_{i,1}(0) | i \in \text{Treated})
$$

which is referred to at **ATT** and reflects the average treatment effect for those that received the treatment.