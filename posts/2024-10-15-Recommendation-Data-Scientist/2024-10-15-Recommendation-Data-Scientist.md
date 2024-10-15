---
categories:
- exposition
date: '2024-10-15'
title: Effective Communications as Data Scientists
description: Principles and Guidelines
toc: true

---

> **Acknowledgment**: This blog post is in collaboration with [Lee Thomas](https://www.linkedin.com/in/leecthomas/) (Senior Director, Professional Communication, Carlson School of Management). Its intended audience is students in the MSBA program. 

> "_Your strategy is only as good as your ability to articulate it._" -- Bob Iger, CEO of Walt Disney

One of the most important skills of a successful data scientist is the ability to communicate one's data analytics solutions to a managerial / executive audience. Being able to communicate effectively and convincingly can lead to acceptance of one's solutions, whereas ineffective communication fails to result in productive decisions. 

At the same time, communication is also among the most under-recognized / under-appreciated skills, especially for MSBA students with limited professional experiences. Far too often, students subscribe to a "technical myth" where hard-core techical skills (e.g., coding, building fancy deep learning / AI models) are excessively valued over skills that are perceived to be "softer" (including presentation, team / project management, and communication). The myth is quickly busted when students need to communicate their data analytics effort in front of a managerial audience, only to find that they cannot seem to get the core messages across the room. 

The purpose of this blog post is to discuss a few broadly applicable principles and guidelines of effective communications as data scientists. We do not claim that simply following these principles and guidelines would automatically create the prefect executive summaries or flawless presentations -- there is no "magic formula" for effective communications. However, we hope to at least navigate you away from common pitfalls and mistakes that we have frequently observed in the past.

It is worth noting that Lee has created an "ABCD" framework, highlighting four crucial and productive strategies for effective communication: (**A**)dapt to the Audience, (**B**)etter than Before, (**C**)onvincing Content, and (**D**)esign & Delivery. This blog post should be treated as a supplement to the "ABCD" framework.

# The Four Horsemen of Communications

Over a number of years teaching and coaching MSBA students, we have distilled four most common mistakes people make when communicating their data analytics work. Colloquially, we refer to them as the "four horsemen of communications". Loosely speaking, they each correspond to one element of the "ABCD" framework, but focuses on what _not_ to do.

## The First Horseman: Being Too Technical

Being too technical in a written report (e.g., an executive summary) or a verbal presentation means you are using technical concepts or terms without providing sufficient explanations for what they mean. As a student, you may well be familiar with various data science terminologies, but your audience may not be. Effective communications start with "meeting your audience where they are", and that requires refraining from throwing out technical terms as if everybody should already understand them.

It is important to point out that "not being too technical" is different from "let's dumb everything down and not use technical terms at all". The complexity of real-world analytics problems usually necessitate some levels of technicality. The key is to provide enough contexts and intuitions to help your audience understand your work.

Think about the following two (contrasting) examples of reporting model selection results in the context of predicting sales leads:

> **Example 1.1**: We perform model selection based on AUC, and the best predictive model achieves an AUC score of 0.9.

vs.

> **Example 1.2**: We select the best predictive model based on a metric called Area-under-the-Curve (AUC), which reflects a model's ability to correctly prioritizing a more promising sales lead ahead of a less promising one. The metric takes value between 0 and 1, with higher value indicating better performance. Our best model achieves an AUC score of 0.9.

It should be clear that example 2 is better and more understandable, even though both contain the same information from a technical perspective. It is worth noting that example 2 does not shy away from mentioning technical terms such as AUC, but provides sufficient explanation under the specific context of the task to facilitate understanding.


## The Second Horseman: Recommendations in a Vacuum

How can you convince your audience that your solution to a problem is "good"? We get this type of questions every year. The reality is, it is often not meaningful to talk about the "good" or "bad" of a solution in a vacuum. Instead, a good solution is typically one that _improves_ upon the pre-existing solution. For example, a "good" clustering solution is one that reveals some patterns in the data that we previously don't know about, and a "good" predictive model may be one that achieves higher performance (on a particular metric of choice) than our currently deployed model. 

This is exactly what the "better than before" element in the "ABCD" framework is about. However, very often, we observe students making recommendations that fail to highlight this aspect.

Take the example previous model selection example:

> **Example 2.1**: We perform model selection based on AUC, and the best predictive model achieves an AUC score of 0.9.

Is 0.9 AUC good or bad? It's hard to say. Although 0.9 seems to be a large number (being close to 1), there really is no way to tell how "impressive" this model is, unless you provide some perspective to compare with. The following re-write improves on this aspect:

> **Example 2.2**: We perform model selection based on AUC, and the best predictive model achieves an AUC score of 0.9. The previously deployed model achieves 0.8 AUC score on the same testing data, which means that our new model improves performance by a significant margin.

Furthermore, if there are opportunities to "translate" model performance improvement to economic / business values, then doing so creates an even stronger message (e.g., the improvement of AUC from 0.8 to 0.9 roughly corresponds to an annual additional revenue of \$xyz).

This applies to exploratory analytics as well (even though model evaluations tend to be subjective). Below is an example:

> **Example 2.3**: We conduct a clustering analysis on the customer base, and discover three unique types of customer groups, which we term as high-spenders, low-spenders, and coupons-lovers. 

vs. 

> **Example 2.4**: We conduct a clustering analysis on the customer base, and discover three unique types of customer groups, which we term as high-spenders, low-spenders, and coupons-lovers. Notably, the "coupon-lovers" cluster features customers who tend to have high spending on discounted product categories using coupons, but have low spending on other product categories. This group of customers was previously included as a part of either high-spenders or low-spenders. We believe acknowledging them separately can help create meaningful marketing promotion strategies down the line.

## The Third Horseman: Jumping to Conclusions

The third horseman, jumping to conclusions, has been plaguing students' communication efforts for as long as we can remember. This problem can manifest in two different (but related) symptoms, which we discuss separately below.

### Symptom 1: Making Recommendations that are "Too Big"
Take a look at the following example:

> **Example 3.1**: We conduct a clustering analysis on the customer base, and discover a high-spender group and a low-spender group. Therefore, our recommendation for growing the business is to encourage the low-spenders to spend more (e.g., via promotional campaigns). 

This is an example of making an out-sized recommendation based only on a very preliminary analysis. Just because some customers don't spend a lot of money does not mean that they will automatically spend more if you ask. Why are they low-spenders in the first place? Is it because they truly have purchasing power but are waiting for better deals (in which case a promotional campaign might work), or is it because they don't need to buy anything else (in which case promotional campaign is counter-productive)? Simply put, the prescriptive question of "what to do" for low-spenders _cannot_ be faithfully answered based on an exploratory clustering analysis alone, period. 

Business decision-making often involves going through incremental and inter-connected smaller steps. "Big" recommendations jump to the last step without making sure that intermediate questions have been properly considered. 

This type of mistake likely stems from an overly narrow perception of what a recommendation can be. Especially when the analyses are exploratory in nature, it is perfectly reasonable to frame your recommendations as _questions_, rather than _actions_. Putting this to work, we can re-write the previous example as:

> **Example 3.2**: We conduct a clustering analysis on the customer base, and discover a high-spender group and a low-spender group. Given the overall objective of growing the business. We believe it is necessary to dig deeper into the low-spender group and understand why they were not spending more money. This could involve examining their transaction data (if available) or conduct small scale interviews or focus groups. Once we have a better understanding of their behaviors, we will be able to make more informed decisions.

Here, the recommendation is less about "what to do" and more about "which question to figure out next". In a complicated business decision, narrowing down which questions to answer is just as valuable (if not more so) than answering the questions.

### Symptom 2: Making Recommendations without Sufficient Evidence
A related symptom is making recommendations based not on specific evidence from your analyses, but based on what you "feel" is appropriate. Think about the following example:

> **Example 3.3**: We have analyzed customer purchases of the focal product following broadcasting of TV ads, and our findings suggest that these TV ads are largely ineffective. Therefore, we recommend running Facebook ads to increase awareness.

Here, unless you were secretly also analyzing online advertisement data related to the focal product and had reason to believe that Facebook ads would solve the awareness problem, then the recommendation is essentially wishful thinking. If you believe that advertising channel (i.e., TV vs. social media) is important, then your recommendation should focus on pointing out this possibility, rather than assuming it will work without evidence. For instance, you can say:

> **Example 3.4**: We have analyzed customer purchases of the focal product following broadcasting of TV ads, and our findings suggest that these TV ads are largely ineffective. We recommend two directions for further exploration. First, we should conduct deeper analyses using TV viewership data to understand whether the lukewarm TV ads performance was because of suboptimal placement (e.g., the ads were aired during low-traffic hours) or because our target customers simply don't watch these TV ads. Second, if the TV channel was the problem, we recommend exploring online advertising channels (such as Facebook ads) as a potential alternative, provided that we can run small-scale pilots to gauge their effectiveness.


## The Four Horseman: Lack of Attention to Aesthetics

The last horseman may appear "trivial" at first, but it bears outsized influences over the outcomes of your communication efforts. Humans are visual animals, and yes, beauty sells. Small aesthetics-related quirks in your executive summaries or presentation slides, which in retrospect often seem so obvious, can leave a bad impression on your audience. Below is a checklist of these "little big things":

- Copy-paste raw coding outputs without careful re-formatting is almost always a bad idea;
- List abbreviated variable names that no one except you can understand;
- Present a visual (e.g., a figure or plot) without title or sensible names of x/y-axis;
- Write a report with the title "Report", or similarly, have a slide with the title "Some Results" -- please provide more meaningful titles;
- Typos and grammar errors;
- ...