This repository is the official implementation of 
# A Nonparametric Bayesian Model for Sparse Dynamic Multigraphs

Dynamic Nonparametric Network Distribution (DNND) describes a generative distribution over sequences of interactions, captured as a time-evolving mixture of dynamic behavioral patterns, that is able to capture both sparse and dense behavior.

- [Method](#Method)
- [Code](#Code)
- [Datasets](#Datasets)
- [References](#References)

# Method
Most real-world networks are dynamic, with the underlying distribution evolving over time. We show that the basic edge-exchangeable framework can be adapted to yield dynamically evolving multigraphs with provable sparsity, replacing the underlying distribution on the space of vertices with time-dependent processes. This construction allows the distribution to evolve over time, in a manner that encourages new edges to contain recently visited vertices. We incorporate this basic dynamic multigraph into a dynamic, hierarchical that retains this sparsity while capturing complex, time-evolving interaction structure. 

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

Please refer to the paper for results.

# Datasets
The datasets contain:
 - [Social Evolution network](http://realitycommons.media.mit.edu/socialevolution.html)
 - [CollegeMsg Network](http://snap.stanford.edu/data/CollegeMsg.html)
 - [Email-Eu-core temporal network](http://snap.stanford.edu/data/email-Eu-core-temporal.html)

# References


