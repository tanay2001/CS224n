#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), embed_size, padding_idx=pad_token_idx)
        self.conv = CNN(embed_size, 21 , embed_size,5)
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(0.3)
        ## End A4 code

        ### YOUR CODE HERE for part 1j


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        output =[]
        ## A4 code
        # note input as 4 dims so we need to pass 1 batch at a time
        for batch in input:
            out = self.embeddings(batch)
            xreshape = torch.transpose(out, -1,-2)

            xconv = self.conv(xreshape)
            xhigh = self.highway(xconv)
            xout = self.dropout(xhigh)
            output.append(xout)
        #convert list to tensor 
        output = torch.stack(output)
        return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j


        ### END YOUR CODE

