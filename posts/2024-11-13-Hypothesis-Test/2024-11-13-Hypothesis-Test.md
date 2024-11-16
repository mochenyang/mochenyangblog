---
categories:
- causal-inference
- exposition
date: '2024-11-13'
description: A casual tour through hypothesis testing, p-value, and R. A. Fisher
title: Why Are Null Hypotheses Often about Equality?
toc: true

---

> **Disclaimer**: the topic discussed in this blog post would most likely seem completely obvious / trivial to statisticians or econometricians. However, it may not be so to many people (myself included) who apply statistics as a tool and have not studied its underlying mathematical principles in a systematic manner.

# The Question
Have you ever wondered: **why, in so many statistical tests, the null hypotheses take the form of some sort of equality?** This is especially evident for two-sided tests. For example:

- In a two-sample $t$ test comparing two means $\mu_1$ and $\mu_2$, the null hypothesis is $\mu_1 = \mu_2$;
- In a one-sample $t$ test of whether a mean $\mu_1$ is significantly different 0, the null hypothesis is $\mu_1 = 0$;
- In an $F$ test of regression coefficients, the null hypothesis is $\beta_0 = \beta_1 = \dots = \beta_k = 0$;
- In a $\chi^2$ test of frequency, the null hypothesis is the observed frequency matches expected frequency;
- In a Pearson's correlation test, the null hypothesis is zero correlation (i.e., $\rho = 0$);
- ... (many others)

More generally, the question is, when we test for a certain hypothesis (e.g., "whether the means of two samples are equal to each other or not"), why is the null hypothesis typically chosen as representing the case of equality rather than case of inequality? 

Regarding this question, I have heard at least three answers from various sources:

1. This is just the way it is. Smart people who designed the test decided that's the null hypothesis. 
2. The null hypothesis should be chosen as the hypothesis you _want_ to reject.
3. No it doesn't have to be equality (i.e., it can be inequality without causing any problems). It's just a habit.

Answer \#1 is clearly unsatisfactory. Answer \#2, believe it or not, is actually what I was taught back in undergrad, and I didn't question it for a long time. However, come to think of it, why would the formulation of null hypothesis have anything to do with what I "want" or "not want"? Finally, answer \#3, as we will see by the end of this blog post, is simply not right.

# The $p$-Value Detour
From a practical perspective, the tool (or rather, the quantity) to actually carry out hypothesis testing is the $p$-value. Therefore, I argue that in order to truly answer the aforementioned question, one needs to really understand what $p$-value represents.[^1]

There have been a lot of criticisms about the arbitrariness of significance level (against which an empirical $p$-value is compared to arrive at a conclusion). The problem of $p$-hacking in scientific research has also given $p$-value a "bad name" and an "evil vibe". However, these problems are more related to misuses / mis-interpretation of $p$-value; as a statistical quantity, $p$-value itself is well-defined (and I would argue represents a quite clever idea).

In particular, $p$-value is the **probability of getting a test statistic that is at least as extreme as what you actually observe, if the null hypothesis is true**. Take a one-sample $t$ test as an illustration here. The test statistic is the $t$ statistic (namely, sample mean devided by sample standard error). Let's say the (empirical) test statistic we compute on the actual sample we have is $\widehat{TS}$, then in a two-sided test with null hypothesis of zero mean, the $p$-value is simply

$$
\Pr(|TS| \geq |\widehat{TS}| ~ \vert \mu = 0)
$$

More generally, if we give up a little bit of mathematical rigor (in exchange for expositional conciseness), the $p$-value can be written as something like

$$
\Pr(TS \text{ being equal or more extreme than } \widehat{TS} ~ \vert H_0 \text{ is true})
$$

Now, to compute such a probability, one would need to characterize the distribution of the test statistic under $H_0$ (specifically the CDF of that distribution). Importantly, this is a **conditional distribution**, i.e., the distribution of test statistic conditional on $H_0$ being true.

And right here is a key reason for why the null hypothesis is so often an equality. If $H_0$ is complicated (or rather, ambiguous), it becomes very difficulty to derive or even estimate such a conditional distribution. However, under an exact form of $H_0$ (such as an equality), one can usually derive a tractable distribution for the test statistic.

**Remark**: all of the illustrative examples from the previous sections are based on two-tail tests (where $H_0$ is exactly an equality). What about **one-tail tests**? For example, the null hypothesis for a one-tail $t$ test is often written as $H_0: ~ \mu \leq 0$ (and the corresponding alternative hypothesis would be $H_1: ~ \mu > 0$). However, this is more of a notational choice. When calculating the $p$-value, one actually uses the **boundary value** under the null hypothesis (which in this case is again 0) to derive the test statistic distribution.[^2] 

# More Generally
Once we understand the reason behind the prevalence of null hypotheses as equality, we can also make sense of the fact that, more generally, it's not "equality" itself that is important. For example, think about the following statistical tests:

- Durbin–Wu–Hausman specification test (e.g., comparing a fixed effect estimator vs. a random effect estimator): the null hypothesis is that both estimators are consistent (and one is more efficient);
- Shapiro-Wilk Test normality test: the null hypothesis is that data is normally distributed;
- Variance Inflation Factor (VIF) test: the null hypothesis is that there is no multi-collinearity;
- Cook’s Distance / leverage test: the null hypothesis is that there is no influential (high leverage) observation.

In these examples, the null hypotheses are not exactly in the form of equality (although they might still be at a mathematical level). However, they are all formulated in a way that allow for precise characterization of the test statistic distributions. In other words, what's important is that the null hypothesis is chosen to be _exact / clearly defined / tractable_ version of the hypothesis one is seeking to test (and equality fits these criteria very well).

# R. A. Fisher
We've got [Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher) to thank for all of the above, who formalized the idea of $p$-value in hypothesis testing. In fact, he originally proposed using 0.05 as a significance cutoff in his [Statistical Methods for Research Workers](https://en.wikipedia.org/wiki/Statistical_Methods_for_Research_Workers).

So, in closing, it seems only fitting to quote Fisher. From his 1935 book [The Design of Experiments](https://en.wikipedia.org/wiki/The_Design_of_Experiments), in the context of the famous ["lady tasting tea" experiment](https://en.wikipedia.org/wiki/Lady_tasting_tea), Fisher succinctly laid out everything I've been babbling for this entire blog.

> It might be argued that if an experiment can disprove the hypothesis that the subject possesses no sensory discrimination between two different sorts of object, it must therefore be able to prove the opposite hypothesis, that she can make some such discrimina tion. But this last hypothesis, however reasonable or true it may be, is ineligible as a null hypothesis to be tested by experiment, because it is inexact. If it were asserted that the subject would never be wrong in her judgments we should again have an exact hypothesis, and it is easy to see that this hypothesis could be disproved by a single failure, but could never be proved by any finite amount of experimentation. **It is evident that the null hypothesis must be exact, that is free from vagueness and ambiguity, because it must supply the basis of the
"problem of distribution," of which the test of significance is the solution.**

[^1]: This is part of motivation for writing this blog post. Students learning applied statistics routinely find $p$-values hard to understand.

[^2]: This is also the reason why the one-tail $p$-value is exactly half of the two-tail $p$-value, if the test statistic distribution is symmetric.