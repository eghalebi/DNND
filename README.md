This repository is the official implementation of 
# A Nonparametric Bayesian Model for Sparse Dynamic Multigraphs

<!--[My Paper Title].(https://arxiv.org/abs/2030.12345). -->

Dynamic Nonparametric Network Distribution (DNND) describes a generative distribution over sequences of interactions, captured as a time-evolving mixture of dynamic behavioral patterns, that is able to capture both sparse and dense behavior.

- [Method](#Method)
- [Code](#Code)
- [Datasets](#Datasets)
- [References](#References)

# Method
Most real-world networks are dynamic, with the underlying distribution evolving over time. We show that the basic edge-exchangeable framework can be adapted to yield dynamically evolving multigraphs with provable sparsity, replacing the underlying distribution on the space of vertices with time-dependent processes. This construction allows the distribution to evolve over time, in a manner that encourages new edges to contain recently visited vertices. We incorporate this basic dynamic multigraph into a dynamic, hierarchical that retains this sparsity while capturing complex, time-evolving interaction structure. 
<!--The Dynamic Nonparametric Network Distribution (\alg) uses a temporally evolving clustering structure and a  hierarchical Bayesian nonparametric framework to capture both global changes in cluster popularity and shifting dynamics within clusters. A judicious choice of base measure for the cluster-specific distributions means that our distribution can generate either sparse or dense multigraphs, with the degree of sparsity controlled by a single parameter.  The increased flexibility allowed by our model leads to improved performance over both its exchangeable counterpart and a range of state-of-the-art dynamic network models. -->

 Please refer to our paper for detailed explanations and more results. 

# Code

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running

To run the model(s) in the paper, run this command:

```run
python main.py 0 8 1 .5 1.5 1 .5 <path_to_output> 1000 10 0 0.15 exponential 20 exponential 20 1 1 0 1 1 100
```

## Evaluation

To evaluate output results, use util.py to run f1_score, portfolia (for AP@k and hits@k).


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |



# Datasets
The datasets contain:
 - [Social Evolution network](http://realitycommons.media.mit.edu/socialevolution.html)
 - [CollegeMsg Network](http://snap.stanford.edu/data/CollegeMsg.html)
 - [Email-Eu-core temporal network](http://snap.stanford.edu/data/email-Eu-core-temporal.html)

# References


