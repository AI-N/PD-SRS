# PD-SRS
# Code 
This is the code for a submitted paper on personalized diversified session-based Recommendation using GNNs.
We have implemented our methods in Pytorch.

Here are three datasets we used in our work:

1- Reddit https://www.kaggle.com/colemaclean/subreddit-interactions

2- Xing http://2016.recsyschallenge.com/

These two datasets are available from A-PGNN study: https://github.com/CRIPAC-DIG/A-PGNN

3- Diginetica: http://cikm2016.cs.iupui.edu/cikm-cup or https://competitions.codalab.org/competitions/11161

# Usage
First, run the file preprocess.py to preprocess the data.
For example:

```python preprocess.py --dataset=Xing```

Then, run the file main.py to train the model.
For example: 

```python main.py --dataset=Xing```

You can change the parameters according to the usage.

# Requirements
Python 3

PyTorch 0.4.0

# Some other related GitHub links:
NISER ("paper title: NISER: Normalized Item and Session Representations with Graph Neural Networks"): https://github.com/johnny12150/NISER

SR-GNN ("paper title: Session-Based Recommendation with Graph Neural Networks"): https://github.com/CRIPAC-DIG/SR-GNN

A-PGNN ("paper title: Personalized graph neural networks with attention mechanism for session-aware recommendation"): https://github.com/CRIPAC-DIG/A-PGNN 

STAMP ("paper title: Stamp: short-term attention/memory priority model for session-based recommendation"): https://github.com/uestcnlp/STAMP

NARM ("paper title: Neural attentive session-based recommendation"): https://github.com/lijingsdu/sessionRec_NARM
