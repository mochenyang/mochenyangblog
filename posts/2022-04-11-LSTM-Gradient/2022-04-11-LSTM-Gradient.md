---
categories:
- machine-learning
- exposition
date: '2022-04-11'
description: Why LSTM addresses vanishing gradients but not exploding ones?
title: The Thing about LSTM and Exploding Gradients
toc: true

---

An important challenge in training recurrent neural networks (RNNs) is the vanishing / exploding gradient problem. Here is an (over-simplified) illustration of the problem. Suppose you have an RNN with $T$ time steps, with an initial input $x$ (i.e., $h_0 = x$) and weight $w$. Assuming a linear activation function (again, for simplicity). The hidden states at time $t$ will be:
$$
h_t = w h_{t-1} = w^t h_0 = w^t x
$$
Therefore, the derivative / gradient with respect to parameter $w$ is $\frac{d h_t}{dw}=t w^{t-1} x$. The longer the time steps $t$, the higher the exponent in the $w^{t-1}$. As a result, for long sequences, the gradient vanishes even if $w$ is slightly smaller than 1, and it explodes even if $w$ is slightly greater than 1. This makes training RNN unstable.

At the root of this problem is the self-multiplication of weights across many time steps. The parameter sharing technique that enables RNNs to handle variable-length sequences is also the culprit of the vanishing / exploding gradient problem.

The LSTM architecture offers robustness against the vanishing gradient problem. To understand how, let's first layout the key pieces of a LSTM cell;

- **forget gate**: $forget_t = sigmoid(X_t, h_{t-1}, \Theta_{forget})$;
- **input gate**: $input_t = sigmoid(X_t, h_{t-1}, \Theta_{input})$;
- **output gate**: $output_t = sigmoid(X_t, h_{t-1}, \Theta_{output})$;
- **Update internal cell state**: $C_t = forget_t \cdot C_{t-1} + input_t \cdot tahn(X_t, h_{t-1}, \Theta)$;
- **Produce output**: $h_t = output_t \cdot tahn(C_t)$.

where $\Theta_{(.)}$ are all parameters that the network learns from data. The three "gates" can be conceptually thought of as "weights", and the real "magic" of LSTM lies in the internal cell state. Notice that $C_t$ is "auto-regressive", in the sense that it depends on $C_{t-1}$ through the time-varying forget gate weights. Having the forget gate weights close to 1 would allow $C_t$ to "memorize" information from previous states. This is what mitigates the vanishing gradient problem.

However, the LSTM architecture does not address the exploding gradient problem. This is because the self-multiplication problem still exists through other variables, such as $output_i$. If we remove the internal cell state for a moment, the output $h_t = output_i$ would be exactly the same as what you get in a regular RNN architecture, where self-multiplication of $\Theta_{output}$ again is a problem.

For more technical / mathematical discussions of this issue, I recommend the following [this StackExchange Q&A](https://stats.stackexchange.com/questions/320919/why-can-rnns-with-lstm-units-also-suffer-from-exploding-gradients) and [this blog post](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html).

