# TRGIR-Rec
The code to reproduce the experimental results for "A Text-based Deep Reinforcement Learning Framework using Self-supervised Graph Representation for Interactive Recommendation".

## Datasets
The data pre-processing codes is also included. You could download Amazon data from *[here](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles)*, and run the dataProcessing.py first, then run the dataPrepare.py.

## Runtime Environment
The code has been tested under Windows 10(version 1909) and Ubuntu 16.04 with TensorFlow 1.15.0 and Python 3.7.6.

Support independent training with CPU and joint training with CPU and GPU when CUDA is available.

## Resource
You can download and add these resource to this project under the folder `./resource`.

The pre-trained word vectors is available on *[GloVe.6B](http://nlp.stanford.edu/data/glove.6B.zip)*, which was trained on Wikipedia2014 and Gigaword 5.

The Long Stopword List can be obtained *[here](https://www.ranks.nl/stopwords)*.

## Model Training
Take `Digital_Music` for example. After getting the source data, you should run data process first:

```
# Digital_Music in Self-supervised Graph Representation

python dataProcessing.py -d Digital_Music -dm sg
python dataPrepare.py -d Digital_Music -dm sg
```

To train our DQN model on `Digital_Music`: 

```
python TRGIR-DQN.py 
```

or train our DDPG model on `Digital_Music`: 

```
python TRGIR-DDPG.py 
```

You can modify the source codes to run other datasets. For other embedding methods, you should change the input by modify 'method' from 'mf' to 'sa' and 'sg''.
