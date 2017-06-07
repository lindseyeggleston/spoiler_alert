# Spoiler Alert!

### Project Overview
I will use a recurrent neural network (RNN) and long short term memory (LSTM) model to generate new believable dialogue between characters as well as plot from the Song of Ice and Fire book series.

language modeling   
lstm allows for more contextual memory


### Approach

Stage 1: Problem Framing
- Identify problem
- Outline project approach
- Define realistic MVPs
- Product: Clear and concise project objective in readme file and comprehension project timeline

Stage 2: Data Preprocessing
- Extract character specific vocabulary from their respective chapters
- Extract character interations and script
- tokenization and lemmatization
- tf-idf with nltk and gensim
- Product: vectorized sequences of

Stage 3: RNN
- Define parameters
- Construct basic 3-layer rnn
- Experiment with dropout
- RMSProps or Adam adaptive learning rate methods
- Train on passages and script to determine best input
- Product: a trained rnn model

Stage 4: Test Model
- Hold out one book from training to predict on
- Vectorize new data (scenarios for character dialogue)
- Product:

Phase 5: Web App and Visualization
- Graph theory to visualize character interactions
- Product:



### References:

- [RNN overview](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy
- [RNNs in depth](https://arxiv.org/pdf/1506.00019.pdf) by Lipton, Berkowitz, and Elkan
- [RNN for Shakespeare](https://github.com/martin-gorner/tensorflow-rnn-shakespeare/blob/master/rnn_train.py): example code
- [RNN w/ LSTM](https://pythonprogramming.net/rnn-tensorflow-python-machine-learning-tutorial/): more example code
- [TensorFlow](https://www.tensorflow.org/tutorials/recurrent)
