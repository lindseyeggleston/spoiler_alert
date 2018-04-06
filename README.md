![alt_text](https://github.com/lindseyeggleston/spoiler_alert/blob/master/flask_app/static/images/banner.png)

### Project Overview
Spoiler Alert in a natural language text generator that produces new text for the Songs of Ice and Fire book series. It employs a recurrent neural network (RNN) with long short-term memory (LSTM) architecture in each hidden layer. The RNN trained on the novels written by George R. R. Martin to learn vocabulary specific to the books as well as mimic the author's style of writing. Initially, the model used a character-by-character prediction on the full corpora. To increase the length of predictions and to improve sentence structure, a later version of the model incorporated word-by-word prediction.

### Approach

Stage 1: Problem Framing
- Identify problem
- Outline project approach
- Define realistic MVPs
- Product: Clear and concise project objective in readme file and comprehension project timeline

Stage 2: Data Preprocessing
- Clean text and remove word contractions
- Extract character specific vocabulary from their respective chapters
- Refine vocabulary to 5000 most frequently used words
- Tokenization with nltk (substitute 'UNKNOWN_TOKEN' for token not in vocab)
- Product: 7 clean datasets - 6 character specific texts and 1 for entire corpora

Stage 3: Build RNN
- Construct framework 3-layer rnn model with lstm
- Define hyper parameters
- Incorporate Adam adaptive learning rate methods
- Train on character datasets on AWS g2 instance
- Save trained models as h5 file in s3 bucket
- Product: 7 trained rnn model

Stage 4: Test Model and Predict
- Hold out one book from training to predict on
- Vectorize new data (scenarios for character dialogue)
- Experiment with dropout
- Generate text for each character
- Product: new text for overall story and 6 individual storylines


### Run the code
If your text file is anywhere as large as mine is, I recommend using a GPU.  

To run train the model:  
`$ python train_rnn.py filepath save_as`  
-or-  
`$ python train_lm.py filepath save_as`  
Filepath is the path to the training txt file or folder and save_as is the name under which the trained model, vocabulary, and unknown tokens dictionary will be saved. This process will clean the data before it runs through the model.

To generate new text:  
`$ python model_path vocab_path unknown_token_path`  
The parameters are the filepaths to the saved trained model, the vocabulary dictionary, and the unknown tokens dictionary.


### References:

- [RNN overview](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy
- [RNNs in depth](https://arxiv.org/pdf/1506.00019.pdf) by Lipton, Berkowitz, and Elkan
- [Language model RNN with python](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
- [RNN for Shakespeare](https://github.com/martin-gorner/tensorflow-rnn-shakespeare): example code
- [RNN w/ LSTM](https://pythonprogramming.net/rnn-tensorflow-python-machine-learning-tutorial/): more example code
- [TensorFlow](https://www.tensorflow.org/tutorials/recurrent)
- [Keras documentation](https://keras.io/)
