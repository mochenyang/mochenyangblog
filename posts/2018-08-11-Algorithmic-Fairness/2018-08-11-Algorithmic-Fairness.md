---
categories:
- machine-learning
- research
date: '2018-08-11'
title: Bias, Discrimination, and Algorithmic Fairness
toc: true

---

This is a (outdated) document on recent literature concerning discrimination and fairness issues in decisions driven by machine learning algorithms. A more comprehensive paper on this issue can be found here: [Integrating Behavioral, Economic, and Technical Insights to Address Algorithmic Bias: Challenges and Opportunities for IS Research](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3446944), which is published in *ACM Transactions on Management Information Systems (TMIS)*.

## Understanding Fairness

Various notions of fairness have been discussed in different domains. Different fairness definitions are not necessarily compatible with each other, in the sense that it may not be possible to simultaneously satisfy multiple notions of fairness in a single machine learning model.

### Definition of Fairness

A common notion of fairness distinguishes **direct discrimination** and **indirect discrimination**. Direct discrimination happens when a person is treated less favorably than another person in comparable situation on protected ground (Romei and Ruggieri 2013; Zliobaite 2015). Here, _comparable situation_ means the two persons are otherwise similarly except on a _protected_ attribute, such as gender, race, etc. In contrast, indirect discrimination happens when an "apparently neutral practice put persons of a protected ground at a particular disadvantage compared with other persons" (Zliobaite 2015).

The two main types of discrimination are often referred to by other terms under different contexts. Direct discrimination is also known as _systematic discrimination_ or _disparate treatment_, and indirect discrimination is also known as _structural discrimination_ or _disparate outcome_. The disparate treatment/outcome terminology is often used in legal settings (e.g., Barocas and Selbst 2016).

More operational definitions of fairness are available for specific machine learning tasks. Consider a binary classification task. The outcome/label represent an important (binary) decision (```pos``` and ```neg```). One of the features is protected (e.g., gender, race), and it separates the population into several non-overlapping **groups** (e.g., ```GroupA``` and ```GroupB```). The classifier estimates the probability that a given instance belongs to ```pos``` based on its features. Two notions of fairness are often discussed (e.g., Kleinberg et al. 2016): **calibration within group** and **balance**. 

Calibration within group means that for both groups, among persons who are assigned probability _p_ of being ```pos```, there should be _p_ fraction of them that actually belong to ```pos``` in expectation. In other words, a probability score should mean what it literally means (in a frequentist sense) regardless of group. A violation of calibration means decision-maker has incentive to interpret the classifier's result differently for different groups, leading to disparate treatment. 

Balance is class-specific. Balance for ```pos``` class means that average probability assigned to people in ```GroupA``` who are actually ```pos``` should be equal to the average probability assigned to people in ```GroupB``` who are actually ```pos```. Balance for ```neg``` can be analogously defined. Balance intuitively means the classifier is not disproportionally inaccurate towards people from one group than the other. In other words, condition on the actual label of a person, the chance of misclassification is independent of the group membership. A violation of balance means that, among people who have the same outcome/label, those in one group are treated less favorably (assigned different probabilities) than those in the other. Balance can be formulated equivalently in terms of error rates, under the term of **equalized odds** (Pleiss et al. 2017) or **disparate mistreatment** (Zafar et al. 2016). In statistical terms, balance for a class is a type of conditional independence. It means that _condition on the true outcome_, the predicted probability of an instance belong to that class is independent of its group membership.

Some other fairness notions are available. _Statistical Parity_ requires members from the two groups should receive the same probability of being ```pos```. One should not confuse statistical parity with balance, as the former does not concern about the actual outcomes - it simply requires average predicted probability of ```pos``` to be equal for two groups. 

Fairness notions are slightly different (but conceptually related) for numeric prediction or regression tasks. Calders et al. (2013) discuss two definitions. First, **equal means** requires the average predictions for people in the two groups should be equal. This is conceptually similar to _statistical parity_ in classification. Second, **balanced residuals** requires the average residuals (errors) for people in the two groups should be equal. This is conceptually similar to _balance_ in classification.


### Relationship among Different Fairness Definitions

First, **not all fairness notions are equally important in a given context**. For example, when base rate (i.e., the actual proportion of ```pos``` in a population) differs in the two groups, statistical parity may not be feasible (Kleinberg et al., 2016; Pleiss et al., 2017). Another case against the requirement of statistical parity is discussed in Zliobaite et al. (2011) and Kamiran et al. (2013). Specifically, statistical disparity in the data (measured as the difference between ```pos``` probabilities received by members of the two groups) is _not all_ discrimination. Part of the difference may be explainable by other attributes that reflect legitimate/natural/inherent differences between the two groups. They argue that statistical disparity only after _conditioning on_ these attributes should be treated as actual discrimination (a.k.a conditional discrimination). In addition, statistical parity ensures fairness at the group level rather than individual level. Dwork et al. (2011) argue for a even stronger notion of individual fairness, where pairs of similar individuals are treated similarly.

Second, **not all fairness notions are compatible with each other**. Kleinberg et al. (2016) show that the three notions of fairness in binary classification, i.e., calibration within groups, balance for ```pos``` class, and balance for ```neg``` class cannot be achieved simultaneously, unless under one of two trivial cases: (1) perfect prediction, or (2) equal base rates in two groups. Such impossibility holds even approximately (i.e., approximate calibration and approximate balance cannot all be achieved unless under approximately trivial cases). Pleiss et al. (2017) extends their work and shows that, when base rates differ, calibration is compatible only with a substantially relaxed notion of balance, i.e., weighted sum of false positive and false negative rates is equal between the two groups, with at most one particular set of weights. These incompatibility findings indicates trade-offs among different fairness notions. Speicher et al. (2018) discuss the relationship between group-level fairness and individual-level fairness. They define a fairness index over a given set of predictions, which can be decomposed to the sum of between-group fairness and within-group fairness. They theoretically show that increasing between-group fairness (e.g., increase statistical parity) can come at a cost of decreasing within-group fairness. 

### Relationship between Fairness and Predictive Performance

In general, a discrimination-aware prediction problem is formulated as a constrained optimization task, which aims to achieve highest accuracy possible, without violating fairness constraints. There is evidence suggesting trade-offs between fairness and predictive performance. Calders et al, (2009) considered the problem of building a binary classifier where the label is correlated with the protected attribute, and proved a trade-off between accuracy and level of dependency between predictions and the protected attribute. Corbett-Davies et al. (2017) demonstrates that maximizing predictive accuracy with a single threshold (that applies to both groups) typically violates fairness constraints. Conversely, fairness-preserving models with group-specific thresholds typically come at the cost of overall accuracy. In essence, the trade-off is again due to different base rates in the two groups. Hardt et al. (2016) proposed algorithms to determine group-specific thresholds that maximize predictive performance under balance constraints, and similarly demonstrated the trade-off between predictive performance and fairness. Speicher et al. (2018) showed that a classifier achieve optimal fairness (based on their definition of a fairness index) can have arbitrarily bad accuracy performance.

Importantly, such trade-off does not mean that one needs to build inferior predictive models in order to achieve fairness goals. In particular, in Hardt et al. (2016), the classifier is still built to be as accurate as possible, and fairness goals are achieved by adjusting classification thresholds. In the same vein, Kleinberg et al. (2018a) proved that "an equity planner" with fairness goals should still build the same classifier as one would without fairness concerns, and adjust decision thresholds. Moreover, such a classifier should take into account the protected attribute (i.e., group identifier) in order to produce correct predicted probabilities. 

Another interesting dynamic is that discrimination-aware classifiers may not always be fair on new, unseen data (similar to the over-fitting problem). Cotter et al. (2018) discuss this issue, using ideas from hyper-parameter tuning.


## Addressing Algorithmic Bias

### Measurement and Detection

Both Zliobaite (2015) and Romei et al. (2013) surveyed relevant measures of fairness or discrimination. One may compare the number or proportion of instances in each group classified as certain class. Sometimes, the measure of discrimination is mandated by law. For instance, the _four-fifths rule_ (Romei et al. 2013) in hiring context requires the job selection rate for the protected group is at least 80% that of the other group. It is a measure of disparate impact. 

If fairness or discrimination is measured as the number or proportion of instances in each group classified to a certain class, then one can use standard statistical tests (e.g., two sample t-test) to check if there is systematic/statistically significant differences between groups. Zliobaite (2015) review a large number of such measures, and Pedreschi et al. (2012) discuss relationships among different measures.

In the particular context of machine learning, previous definitions of fairness offer straightforward measures of discrimination. For instance, the degree of balance of a binary classifier for the positive class can be measured as the difference between average probability assigned to people with positive class in the two groups. In addition, Pedreschi et al. (2009) developed several metrics to quantify the degree of discrimination in association rules (or IF-THEN decision rules in general). Two similar papers are Ruggieri et al. (2010ab), which also associate these discrimination metrics with legal concepts, such as affirmative action. See also Kamishima et al. (2012) for more discussions on measuring different types of discrimination in IF-THEN rules. Speicher et al. (2018) define a _fairness index_ that can quantify the degree of fairness for any two prediction algorithms. Their definition is rooted in the inequality index literature in economics. Yang and Stoyanovich (2016) develop measures for rank-based prediction outputs to quantify/detect statistical disparity.

Specialized methods have been proposed to detect the existence and magnitude of discrimination in data. Adebayo and Kagal (2016) use the orthogonal projection method to create multiple versions of the original dataset, each one removes an attribute and makes the remaining attributes orthogonal to the removed attribute. Then, the model is deployed on each generated dataset, and the decrease in predictive performance measures the dependency between prediction and the removed attribute. Zhang and Neil (2016) treat this as an anomaly detection task, and develop subset scan algorithms to find subgroups that suffer from significant disparate mistreatment.

Discrimination has been detected in several real-world datasets and cases. For example, Kamiran et al. (2012) identified discrimination in criminal records where people from minority ethnic groups were assigned higher risk scores. Caliskan et al. (2017) detect and document a variety of implicit biases in natural language, as picked up by trained word embeddings. Chouldechova (2017) showed the existence of disparate impact using data from the COMPAS risk tool.

### Prevention/Mitigation

Techniques to prevent/mitigate discrimination in machine learning can be put into three categories (Zliobaite 2015; Romei et al. 2013): (1) **data pre-processing**, (2) **algorithm modification**, and (3) **model post-processing**.

**Data pre-processing** tries to manipulate training data to get rid of discrimination embedded in the data. 

- Calders et al, (2009) propose two methods of cleaning the training data: (1) flipping some labels, and (2) assign unique weight to each instance, with the objective of removing dependency between outcome labels and the protected attribute. The first approach of flipping training labels is also discussed in Kamiran and Calders (2009), and Kamiran and Calders (2012).
- Hajian et al. (2011) discuss a data transformation method to remove discrimination learned in IF-THEN decision rules. The high-level idea is to manipulate the confidence scores of certain rules.
- Zemel et al. (2013) propose to learn a set of intermediate representation of the original data (as a multinomial distribution) that achieves statistical parity, minimizes representation error, and maximizes predictive accuracy.
- Mancuhan and Clifton (2014) build non-discriminatory Bayesian networks. Their algorithm depends on deleting the protected attribute from the network, as well as pre-processing the data to remove discriminatory instances.
- Bolukbasi et al. (2016) discuss de-biasing technique to remove stereotypes in word embeddings learned from natural language.
- Celis et al. (2016) study the problem of not only removing bias in the training data, but also maintain its diversity, i.e., ensure the de-biased training data is still representative of the feature space.
- Lum and Johndrow (2016) propose to de-bias the data by transform the entire feature space to be orthogonal to the protected attribute.
- Bower et al. (2018) use a regression-based method to transform the (numeric) label so that the transformed label is independent of the protected attribute conditioning on other attributes. This addresses conditional discrimination.

**Algorithm modification** directly modifies machine learning algorithms to take into account fairness constraints. A general principle is that simply removing the protected attribute from training data is not enough to get rid of discrimination, because other correlated attributes can still bias the predictions. This problem is known as **redlining**. 

- Calders and Verwer (2010) propose to modify naive Bayes model in three different ways: (i) change the conditional probability of a class given the protected attribute; (ii) train two separate naive Bayes classifiers, one for each group, using data only in each group; and (iii) try to estimate a "latent class" free from discrimination.
- Kamiran et al. (2010) develop a discrimination-aware decision tree model, where the criteria to select best split takes into account not only homogeneity in labels but also heterogeneity in the protected attribute in the resulting leaves. 
- Kamishima et al. (2011) use regularization technique to mitigate discrimination in logistic regressions. The regularization term increases as the degree of statistical disparity becomes larger, and the model parameters are estimated under constraint of such regularization.
- Dwork et al. (2011) formulate a linear program to optimize a loss function _subject to_ individual-level fairness constraints. They define a distance score for pairs of individuals, and the outcome difference between a pair of individuals is bounded by their distance. A follow up work, Kim et al. (2018), relaxes the knowledge requirement on the distance metric.
- Feldman et al. (2014) specifically designed a method to remove disparate impact defined by the four-fifths rule, by formulating the machine learning problem as a constraint optimization task. 
- Fish et al. (2014) adapt AdaBoost algorithm to optimize simultaneously for accuracy and fairness measures.
- Bechavod and Ligett (2017) address the disparate mistreatment notion of fairness by formulating the machine learning problem as a optimization over not only accuracy but also minimizing differences between false positive/negative rates across groups.
- Berk et al. (2017) apply regularization method to regression models. The key contribution of their paper is to propose new regularization terms that account for both individual and group fairness.
- Grgic-Hlaca et al. (2017) propose to build ensemble of classifiers to achieve fairness goals. Interestingly, they show that an ensemble of unfair classifiers can achieve fairness, and the ensemble approach mitigates the trade-off between fairness and predictive performance.
- Dwork et al. (2017) develop a decoupling technique to train separate models using data only from each group, and then combine them in a way that still achieves between-group fairness.
- Agarwal et al. (2018) reduces the fairness problem in classification (in particular under the notions of statistical parity and equalized odds) to a cost-aware classification problem.

**Model post-processing** changes how the predictions are made from a model in order to achieve fairness goals. 

- Kamiran et al. (2010) propose to re-label the instances in the leaf nodes of a decision tree, with the objective to minimize accuracy loss and reduce discrimination. The predictions on unseen data are made not based on majority rule with the re-labeled leaf nodes.


## Other Thoughts

1. The issue of algorithmic bias is closely related to the interpretability of algorithmic predictions. By making a prediction model more interpretable, there may be a better chance of detecting bias in the first place. Meanwhile, model interpretability affects users' trust toward its predictions (Ribeiro et al. 2016). Regulations have also been put forth that create "right to explanation" and restrict predictive models for individual decision-making purposes (Goodman and Flaxman 2016).

2. The design of discrimination-aware predictive algorithms is only part of the design of a discrimination-aware decision-making tool, the latter of which needs to take into account various other technical and behavioral factors. Such a gap is discussed in Veale et al. (2018)

## References

- Adebayo, J., & Kagal, L. (2016). Iterative Orthogonal Feature Projection for Diagnosing Bias in Black-Box Models, 37. Retrieved from http://arxiv.org/abs/1611.04967
- Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A Reductions Approach to Fair Classification.
- Barocas, S., & Selbst, A. (2016). Big Data's Disparate Impact. _California Law Review_, 104(1), 671–729. 
- Bechavod, Y., & Ligett, K. (2017). Penalizing Unfairness in Binary Classification. Retrieved from http://arxiv.org/abs/1707.00044
- Berk, R., Heidari, H., Jabbari, S., Joseph, M., Kearns, M., Morgenstern, J., … Roth, A. (2017). A Convex Framework for Fair Regression, 1–5. Retrieved from http://arxiv.org/abs/1706.02409
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Debiasing Word Embedding, (Nips), 1–9.
- Bower, A., Niss, L., Sun, Y., & Vargo, A. (2018). Debiasing representations by removing unwanted variation due to protected attributes. Retrieved from https://www.fatml.org/media/documents/debiasing_representations.pdf
- Calders, T., & Verwer, S. (2010). Three naive Bayes approaches for discrimination-free classification. _Data Mining and Knowledge Discovery_, 21(2), 277–292.
- Calders, T., Kamiran, F., & Pechenizkiy, M. (2009). Building classifiers with independency constraints. ICDM Workshops 2009 - IEEE International Conference on Data Mining, (December), 13–18.
- Calders, T., Karim, A., Kamiran, F., Ali, W., & Zhang, X. (2013). Controlling attribute effect in linear regression. Proceedings - IEEE International Conference on Data Mining, _ICDM_, (1), 71–80.
- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. _Science_, 356(6334), 183–186.
- Celis, L. E., Deshpande, A., Kathuria, T., & Vishnoi, N. K. (2016). How to be Fair and Diverse? Retrieved from http://arxiv.org/abs/1610.07183
- Chouldechova, A. (2017). Fair Prediction with Disparate Impact : A Study of Bias in Recidivism Prediction Instruments. Big Data, 5(2), 153–163.
- Corbett-Davies, S., Pierson, E., Feller, A., Goel, S., & Huq, A. (2017). Algorithmic decision making and the cost of fairness. [arXiv](https://arxiv.org/pdf/1701.08230.pdf)
- Cotter, A., Gupta, M., Jiang, H., Srebro, N., Sridharan, K., & Wang, S. (2018). Training Fairness-Constrained Classifiers to Generalize.
- Custers, B. (2013). Discrimination and Privacy in the information society. Discrimination and Privacy in the Information Society (Vol. 3).
- Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2011). Fairness Through Awareness. [arXiv](https://arxiv.org/pdf/1104.3913.pdf)
- Dwork, C., Immorlica, N., Kalai, A. T., & Leiserson, M. (2017). Decoupled classifiers for fair and efficient machine learning.
- Feldman, M., Friedler, S., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2014). Certifying and removing disparate impact. [arXiv](https://arxiv.org/pdf/1412.3756.pdf) 
- Fish, B., Kun, J., & Lelkes, A. (2014). Fair Boosting : a Case Study.
- Goodman, B., & Flaxman, S. (2016). European Union regulations on algorithmic decision-making and a "right to explanation," 1–9.
- Grgic-Hlaca, N., Zafar, M. B., Gummadi, K. P., & Weller, A. (2017). On Fairness, Diversity and Randomness in Algorithmic Decision Making. 
- Hajian, S., Domingo-Ferrer, J., & Martinez-Balleste, A. (2011). Discrimination prevention in data mining for intrusion and crime detection. 2011 IEEE Symposium on Computational Intelligence in Cyber Security, 47–54. 
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning, (Nips).
- Kamiran, F., & Calders, T. (2009). Classifying without discriminating. 2009 2nd International Conference on Computer, Control and Communication, IC4 2009.
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. Knowledge and Information Systems (Vol. 33).
- Kamiran, F., Calders, T., & Pechenizkiy, M. (2010). Discrimination aware decision tree learning. Proceedings - IEEE International Conference on Data Mining, ICDM, 869–874. 
- Kamiran, F., Karim, A., Verwer, S., & Goudriaan, H. (2012). Classifying socially sensitive data without discrimination: An analysis of a crime suspect dataset. In Proceedings - 12th IEEE International Conference on Data Mining Workshops, ICDMW 2012 (pp. 370–377). 
- Kamiran, F., Žliobaite, I., & Calders, T. (2013). Quantifying explainable discrimination and removing illegal discrimination in automated decision making. Knowledge and Information Systems (Vol. 35).
- Kamishima, T., Akaho, S., & Sakuma, J. (2011). Fairness-aware learning through regularization approach. Proceedings - IEEE International Conference on Data Mining, ICDM, 643–650.
- Kamishima, T., Akaho, S., Asoh, H., & Sakuma, J. (2012). Considerations on fairness-aware data mining. Proceedings - 12th IEEE International Conference on Data Mining Workshops, ICDMW 2012, 378–385. 
- Kim, M. P., Reingold, O., & Rothblum, G. N. (2018). Fairness Through Computationally-Bounded Awareness. https://doi.org/10.1145/2090236.2090255
- Kleinberg, J., & Raghavan, M. (2018b). Selection Problems in the Presence of Implicit Bias. [arXiv](https://arxiv.org/pdf/1801.03533.pdf)
- Kleinberg, J., Lakkaraju, H., Leskovec, J., Ludwig, J., & Mullainathan, S. (2017). Human decisions and machine predictions. _The quarterly journal of economics_, 133(1), 237-293.
- Kleinberg, J., Ludwig, J., Mullainathan, S., & Rambachan, A. (2018a). Algorithmic Fairness. _AEA Papers and Proceedings_, 108, 22–27.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). Inherent Trade-Offs in the Fair Determination of Risk Scores. [arXiv](https://arxiv.org/pdf/1609.05807.pdf)
- Lum, K., & Johndrow, J. (2016). A statistical framework for fair predictive algorithms, 1–6. Retrieved from http://arxiv.org/abs/1610.08077
- Mancuhan, K., & Clifton, C. (2014). Combating discrimination using Bayesian networks. Artificial Intelligence and Law, 22(2), 211–238. 
- Pedreschi, D., Ruggieri, S., & Turini, F. (2009). Measuring Discrimination in Socially-Sensitive Decision Records. Proceedings of the 2009 SIAM International Conference on Data Mining, 581–592.
- Pedreschi, D., Ruggieri, S., & Turini, F. (2012). A study of top-k measures for discrimination discovery. Proceedings of the 27th Annual ACM Symposium on Applied Computing. 
- Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). On Fairness and Calibration. [arXiv](http://arxiv.org/abs/1709.02012)
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD 2016.
- Romei, A., & Ruggieri, S. (2013). A multidisciplinary survey on discrimination analysis. _Knowledge Engineering Review_, 29(5), 582–638.
- Ruggieri, S., Pedreschi, D., & Turini, F. (2010a). Integrating induction and deduction for finding evidence of discrimination. Artificial Intelligence and Law, 18(1), 1–43.
- Ruggieri, S., Pedreschi, D., & Turini, F. (2010b). Data mining for discrimination discovery. ACM Transactions on Knowledge Discovery from Data, 4(2), 1–40.
- Speicher, T., Heidari, H., Grgic-Hlaca, N., Gummadi, K. P., Singla, A., Weller, A., & Zafar, M. B. (2018, July). A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual &Group Unfairness via Inequality Indices. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2239-2248). ACM.
- Veale, M., Van Kleek, M., & Binns, R. (2018). Fairness and Accountability Design Needs for Algorithmic Support in High-Stakes Public Sector Decision-Making. CHI Proceeding, 1–14.
- Yang, K., & Stoyanovich, J. (2016). Measuring Fairness in Ranked Outputs. 
- Zafar, M. B., Valera, I., Rodriguez, M. G., & Gummadi, K. P. (2016). Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment.
- Zemel, R. S., Wu, Y., Swersky, K., Pitassi, T., & Dwork, C. (2013). Learning Fair Representations. Proceedings of the 30th International Conference on Machine Learning, 28, 325–333. 
- Zhang, Z., & Neill, D. B. (2016). Identifying Significant Predictive Bias in Classifiers, (June), 1–5. Retrieved from http://arxiv.org/abs/1611.08292 
- Zliobaite, I. (2015). A survey on measuring indirect discrimination in machine learning. [arXiv](https://arxiv.org/pdf/1511.00148.pdf)
- Zliobaite, I. (2015). On the relation between accuracy and fairness in binary classification. [arXiv](https://arxiv.org/pdf/1505.05723.pdf)
- Zliobaite, I., Kamiran, F., & Calders, T. (2011). Handling conditional discrimination. Proceedings - IEEE International Conference on Data Mining, ICDM, (1), 992–1001.

## Other Resources

[Emergence of Intelligent Machines](https://www.youtube.com/channel/UC6f1sYAoj6_sxF-lhroS6SQ): a series of talks on algorithmic fairness, biases, interpretability, etc.
