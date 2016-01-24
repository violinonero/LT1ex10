'''
Created on Jan 23, 2016

@author: mittul
'''
import sys
import time
import datetime
import argparse

import numpy as np
np.random.seed(1337) # seed the random number generator to produce the same initialization of the neural network

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam

'''
Read corpus from standard input

inputs: 
vocab: a dictionary of words and their associated indices
sent_end_marker: string containing the sentence end marker
unk: string containing the OOV symbol
f: string specifying the file name of a corpus to read from

outputs:
data_train: list of bigrams in the corpus
'''
def read_corpora(vocab,sent_end_marker,unk,f=None):
    train=[]
    corpus_size=0
    context_size=0
    max_context_size = 1 
    bigram_list = []
    
    sent_end_idx=-1
    if sent_end_marker in vocab:
        sent_end_idx=vocab[sent_end_marker]
    else:
        print >> sys.stderr, "WARNING: corpus has no sentence end markers !"
    
    if unk in vocab:
        unk_idx=vocab[unk]
    else:
        unk_idx=len(vocab)
        vocab[unk]=unk_idx
        print >> sys.stderr, "WARNING: OOVs detected, replacing them with default unk symbol !"
    
    in_stream = sys.stdin
    if f is not None:
        in_stream = open(f,"r")
    
    for line in in_stream:
        line=line.rstrip('\n')
        corpus_size += 1
        
        #convert oovs to unk
        idx=unk_idx
        if line in vocab: 
            idx=vocab[line]
            
        bigram_list.append(idx)
        if context_size < max_context_size:
            context_size += 1
        elif (idx == sent_end_idx):
            context_size = 0
            train.append(list(bigram_list))
            del bigram_list[:]
        else:
            train.append(list(bigram_list))
            bigram_list.pop(0)
            
    data_train= np.array(train)
    data_train=data_train.astype(np.int32)
    return data_train
 
'''
Compute the perplexity

inputs:
model: object of the model to use to compute the perplexity
test_data: numpy array of arrays of out of vocabulary words
chunk: int specifying the size of data to calculate score on iteratively 

outputs:
Prints the perplexity on the total number of words on the input data
'''
def ppl(model, test_data, chunk):
    n_test = test_data.shape[0]
    n_test_chunks = int(np.ceil(float(n_test) / float(chunk)))
    total_score = 0
    for chunk_idx in xrange(n_test_chunks):
        test_chunk_x = test_data[chunk_idx*chunk:(chunk_idx+1)*chunk,:-1]
        test_chunk_y = test_data[chunk_idx*chunk:(chunk_idx+1)*chunk,-1]
        predictions = np.log(model.predict(test_chunk_x))[np.arange(test_chunk_x.shape[0]),test_chunk_y]
        total_score += np.sum(predictions)

    print "Words: %d, Perplexity %f" % (n_test,np.exp(-1*total_score/n_test))


'''
Build the feedforward neural network

inputs:
input_dim: int specifying the dimension of the input layer
n_hidden: int specifying the dimension of the hidden layer
activatin_fn: string specifying the activation function for the hidden layer
output_fn: string specifying the output function for the output layer

outputs:
model: the keras model object with the neural network structure
'''
def build(input_dim, embedding_layer_dim, n_hidden, activation_fn, output_fn): #Build the neural network
    #BEGIN
    #ADD CODE HERE
    #END 
    return model

'''
Fit the model on the data

inputs:
train_data: numpy array of arrays with bigram data
encodings: numpy array of arrays of word encodings
model: keras model object ot be trained
lr: float specifying the learning rate for the ADAM algorithm
epsilon: float specifying the epsilon for the ADAM algorithm
nb_epochs: int specifying the number of epochs
chunk: int specifying the chunk of data to handle at a time
batch_size: int specifying the batch size of the data to evalulate the algorithm on 

outputs:
None
'''
def fit(train_data,encodings,model,lr,epsilon,nb_epochs,chunk,batch_size):
    n_train = train_data.shape[0]
    n_train_chunks = int(np.ceil(float(n_train) / float(chunk)))
    adam = Adam(lr=lr, epsilon=epsilon)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    e = 0
    while e < nb_epochs:
        e +=1
        for chunk_idx in xrange(n_train_chunks):
            train_chunk_x = train_data[chunk_idx*chunk:(chunk_idx+1)*chunk,:-1]
            train_chunk_y = train_data[chunk_idx*chunk:(chunk_idx+1)*chunk,-1]
            y_train = encodings[train_chunk_y,:]
            model.fit(train_chunk_x,y_train,batch_size=batch_size,nb_epoch=1,verbose=1)
'''
Main function
'''
if __name__ == '__main__':
    #Parse arguments
    parser = argparse.ArgumentParser(description='Run a recurrent layer based neural network LM and print perplexity on a list of files',usage='rnnlm -vocab voc_file -model path_to_model_file -weights path_to_weights_file [-train|-ppl] < data (stdin)')
    parser.add_argument('-sep', help="specify the sentence end marker, do specify the -vocab flag as well (default: </s>)")
    parser.add_argument('-vocab', help="vocab file")
    parser.add_argument('-ppl', help="space delimited list of files to computer perplexity on. eg: -ppl dev.txt test.txt")
    parser.add_argument('-hidden', type=int, help="number of nodes in the first hidden layer (default: 200)")
    parser.add_argument('-act', help="activation function")
    parser.add_argument('-oact', help="output activation function")
    parser.add_argument('-embed', type=int, help="number of nodes in the embedding layer (default: 200)")
    parser.add_argument('-lr', type=float, help="specify learning rate (default: 0.01)")
    parser.add_argument('-eps', type=float, help="specify epsilon for adagrad (default: 1e-6)")    
    parser.add_argument('-epochs', type=int, help="specify number of epochs of training to run (default: 5)") 
    parser.add_argument('-bs', type=int, help="specify the batch size for training (default: 500)")
    parser.add_argument('-ch', type=int, help="specify the chunk size for training (default: 10000)")
    parser.add_argument('-unk', type=int, help="specify your own unk symbol to handle oovs, make sure it is already in the vocab file (default: <unk>)")
    args = parser.parse_args()
        
    t0 = time.time()
    
    #read vocab from file  
    sent_end_idx=1
    vocab={}
    idx=-1
    
    unk = "<unk>"
    if args.unk:
        unk = args.unk
    
    sent_end_marker="</s>"
    if args.sep:
        sent_end_marker=args.sep

    if args.vocab:
        with open(args.vocab, 'r') as searchfile:
            for line in searchfile:
                idx +=1
                line=line.rstrip("\r\n")
                vocab[line]=idx
    else:
        print >> sys.stderr, "Must specify the vocab file using the -vocab flag"

    #read corpora from stdin 
    print  >> sys.stderr, "Reading the data"
    x = read_corpora(vocab,sent_end_marker=sent_end_marker,unk=unk)
    voc_size= len(vocab)
 
    #read model parameters
    n_embed=200
    if args.embed:
        n_embed = args.embed
        
    n_hidden=200
    if args.hidden:
        n_hidden = args.hidden
    
    act = 'sigmoid'
    if args.act:
        act = args.act
    
    oact = 'softmax'
    if args.oact:
        oact = args.oact
    
    lr=0.001
    if args.lr:
        lr = args.lr
    
    epsilon=1e-08
    if args.eps:
        epsilon = args.eps
    
    nb_epochs = 5
    if args.epochs:
        nb_epochs = args.epochs
    
    batch_size = 500
    if args.bs:
        batch_size = args.bs
    
    chunk=10000
    if args.ch:
        chunk = args.ch
    
    filelist = []
    if args.ppl:
        filelist = [item for item in args.ppl.split(' ')]
    else:
        print >> sys.stderr, "Must specify files to calculate perplexity on using the -ppl flags"
        sys.exit(1)
    
    E=np.eye(voc_size)
    print >> sys.stderr, "Using one-hot encodings"
    print  >> sys.stderr, "Building the model"
    model=build(input_dim=voc_size,embedding_layer_dim=n_embed,n_hidden=n_hidden,activation_fn=act,output_fn=oact)
    print  >> sys.stderr, "Fitting the model"
    fit(train_data=x,encodings=E,model=model,lr=lr,epsilon=epsilon,nb_epochs=nb_epochs,batch_size=batch_size,chunk=chunk)
    
    for f in filelist:
        print >> sys.stderr, "Evaluating the model on file %s" % f
        x = read_corpora(vocab, sent_end_marker, unk, f=f)
        ppl(model,x,chunk=chunk)
    
    print >> sys.stderr, "Elapsed time: %f" % (time.time() - t0)
