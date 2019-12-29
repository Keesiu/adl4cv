# Cost-Efficient Deep Active Learning Using Monte-Carlo Dropout


Deep Learning enabled many real world applications on human-level in the fields of Computer Vision and Natural Language Processing. Recent advancements were not only caused by superior neural network architectures and increased computing power, but also the increased amount of available data. Especially in Supervised Learning, a large and well labeled dataset is a decisive factor for the performance of the Neural Network. But most data in practical scenarios is actually not labeled. Even though the amount of labeled data required can be reduced via the paradigm of Transfer Learning, for each specific use case, often one still needs to label a fair amount of additional data manually. Furthermore, this manual labeling can become quite expensive, especially when domain expertise is needed, for example when labeling medical images.
<br>

The efficiency of the data labeling process can be increased through Active Learning (AL). In a general Machine Learning context, Active Learning classically pursues the idea of iteratively querying a user by showing data points that the user labels interactively by hand. The goal of Active Learning is to reach a sufficient model performance with as few manually labeled data points as possible. A core challenge in this process is to come up with good samples to query. Commonly, the idea is to use the samples where the model is very insecure about. The existing literature provides and compares many strategies to measure uncertainty for predictions. However, in a Deep Learning context, uncertainty is usually not provided per se. The softmax output of the final classification layer is often interpreted as a probability distribution, but only because it depicts one technically rather than from a probability-theoretical point-of-view. However, \cite{gal2016} argue that this is not suited as a measurement of uncertainty. In particular, for new data points which quite differ from everything seen before, the softmax may give over-confident predictions.
<br>

gal2016 propose another way to measure uncertainty, namely by leveraging the dropout mechanism, so-called Monte-Carlo Dropout (MC Dropout). Usually, dropout is used as a regularization strategy to improve model performance. \cite{gal2016} shows, that applying the dropout during test time actually leads to multiple predictions in a Monte-Carlo manner, which allows Bayesian approximations of the prediction uncertainty.
<br>

As shown by wang2016, the data efficiency can be further improved by utilizing their proposed CEAL (Cost-Efficient Active Learning) framework. Additionally to the manually labeled samples, the most certain ones get automatically pseudo-labeled by the algorithm. All labeled and pseudo-labeled data is then used for the next Active Learning cycle.
<br>

Our goal is to implement Deep Active Learning using Monte-Carlo Dropout gal2016 with CEAL wang2016 and benchmark it in the context of image classification.
<br>

For the present project our milestones are:

1. Using Transfer Learning on the pre-trained ResNet-18 on the full Caltech-256, particularly with Dropout.
2. Implement the Active Learning pipeline utilizing Monte Carlo Dropout and the Transfer Learning principle.
3. Integrating the CEAL approach.
4. Benchmarking different combinations in respect to labeling efficiency and performance: 

    - Full-trained model without AL (upper baseline)
    - AL with CEAL + MC Dropout
    - AL with CEAL approach
    - AL with MC Dropout
    - AL with softmax-based policy
    - AL with random policy (bottom baseline)
    
<br>
With this project, we aim for following contributions. Firstly, to the best of our knowledge, we are the first ones to combine Monte-Dropout with CEAL in a image classification context. Secondly, we benchmark those concepts against each other.# adl4cv
