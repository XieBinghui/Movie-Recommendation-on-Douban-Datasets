# coding: utf-8

from SkipGram import SkipGram
from Graphtools import Graphtools
from time import sleep
import numpy as np
import pickle


class DeepWalk(object):
    
    def __init__(self, G, window_size=5, embedding_size=64, walks_per_vertex=30, walk_length=16):
        self.window_size=window_size
        self.embedding_size=embedding_size
        self.walks_per_vertex=walks_per_vertex
        self.walk_length=walk_length
        self.whole_size=len(G.nodes)
        
        self.G=G
        self.Theta=np.random.rand(self.whole_size, embedding_size)
        self.skipgram=SkipGram(window_size, embedding_size, self.whole_size, 0.01, \
            Theta=self.Theta)

    def __call__(self, unit_iter=5, beta=0.9, showLoss=False):
        '''
        Run DeepWalk algorithm.
        '''
        valid_length=2*self.window_size+1
        
        for i in range(self.walks_per_vertex):
            nodes=self.G.nodes().numpy()
            np.random.shuffle(nodes)
            for v in nodes:
                print('-----------', v, '------------')
                walk=self.RandomWalk(v, In_Out='Both')
                
                # Test
                # print(walk)

                if len(walk)<valid_length:
                    print('Fail to walk long enough, skip once.')
                    continue
                self.skipgram.walk_train(walk, unit_iter=unit_iter, showLoss=showLoss)
            # One of the effective ways to facilitate convergence is to use
            #  diminishing stepsize
            self.skipgram.step_size=self.skipgram.step_size*beta

        print('Deepwalk finished.')

    def RandomWalk(self, vi_index, In_Out='Both'):
        '''
        Return a walk on graph G from vi, 
        controlled by parameters window_size, walks_per_vertex and walk_length.
        ***** DO NOT distinguish in_edges from out_edges by default *****
        In_Out='In': In edges only; 'Out': Out edges only.
        '''
        length=0
        current=vi_index
        walk=np.empty(self.walk_length, dtype=int)
        if In_Out=='Both':
            while length<self.walk_length:
                walk[length]=current
                pool_out=self.G.out_edges(current)[1].numpy()
                pool_in=self.G.in_edges(current)[0].numpy()
                pool=np.concatenate((pool_in, pool_out))
                if len(pool)==0:
                    return walk
            
                current=np.random.choice(pool)
                length+=1
        elif In_Out=='In':
            while length<self.walk_length:
                walk[length]=current
                pool=self.G.in_edges(current)[0].numpy()
                if len(pool)==0:
                    return walk

                current=np.random.choice(pool)
                length+=1
        elif In_Out=='Out':
            while length<self.walk_length:
                walk[length]=current
                pool=self.G.out_edges(current)[1].numpy()
                if len(pool)==0:
                    return walk

                current=np.random.choice(pool)
                length+=1
        return walk

    def predict(self, vi_index=0):
        '''
        Return descending indexes rank according to probabilities for vertex vi.
        '''
        return self.skipgram.testprob(vi_index).argsort()[::-1]


    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            print('Model saved to file.')


if __name__=='__main__':
    gtools=Graphtools()
    G=gtools.fromText('p2p-Gnutella08.txt')
    # with open('deepwalk.model', 'rb') as f:
    #     deepwalk=pickle.load(f)
    deepwalk=DeepWalk(G, 1, embedding_size=64, walks_per_vertex=2, walk_length=8)
    deepwalk(unit_iter=5, showLoss=False)
    deepwalk.save()
    print('Done.')
