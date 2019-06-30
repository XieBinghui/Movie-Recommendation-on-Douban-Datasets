# coding: utf-8
import numpy as np
from BinaryTree import Tree

class SkipGram(object):
    '''
    An implementation of Skip-Gram model, using Hierarchical Softmax.
    '''

    def __init__(self, window_size, embedding_size, whole_size, step_size=0.01, Theta=None):
        self.window_size=window_size
        self.embedding_size=embedding_size
        self.whole_size=whole_size
        self.step_size=step_size
        # Theta
        if Theta is not None:
            self.Theta=Theta
        else:
            self.Theta=np.random.rand(self.whole_size, self.embedding_size)

        self.tree=Tree(self.whole_size, self.embedding_size)
        self.tree.growTree()

    def __call__(self, target_index):
        '''
        Evaluate the output from SkipGram model.
        target_index: The index will be used to generate a vector,
                    and then the vector is used to run over the model.
        '''
        pass
    
    def get_Nodes_and_Phis(self, ui_index):
        '''
        Get Phis and corresponding nodes with respect to vi.
        OUTPUT:
            node_list: nodes on the route from the root to vi(vi excluded).
            phi_list: an (l*embedding_size) ndarray, with each row being the Phi 
                     of the corresponding node in node_list.
        '''
        node_list=self.tree.getNodeList(ui_index)
        Phi=np.empty((len(node_list), len(node_list[0].Phi)), dtype=type(node_list[0].Phi))
        for i in range(len(node_list)):
            Phi[i]=node_list[i].Phi
        return node_list, Phi

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def lossfunc(self, ui_index, vi_index):
        '''
        ui_index: one of the indexes of vi's context
        vi_index: vi's index
        '''
        _, Phi=self.get_Nodes_and_Phis(ui_index)
        seq=self.tree.index2Bin(ui_index)
        d=np.fromiter(seq, dtype=float)
        loss=-np.sum([d[i]*np.log(self.sigmoid(np.vdot(Phi[i], self.Theta[vi_index])))\
                +(1-d[i])*np.log(1-self.sigmoid(np.vdot(Phi[i], self.Theta[vi_index])))\
                    for i in range(len(d))])
        # loss=np.sum([np.log(1+np.exp(-np.vdot(Phi[i], self.Theta[vi_index]))) for i in range(len(d))])
        return loss

    def testloss(self, sampling_size=100):
        '''
        Using sampling to estimate the loss of the whole model.
        '''
        group=[None, None]
        group[0]=np.random.choice(range(self.whole_size), sampling_size)
        tmp=set(range(self.whole_size))
        tmp=tmp.difference(set(group[0]))
        group[1]=np.random.choice(list(tmp), sampling_size)
        return sum([self.lossfunc(i, j) for i, j in zip(group[0], group[1])])/sampling_size

    def testRMSE(self, a, b):
        tmp=self.Theta[a]-self.Theta[b]
        return np.sqrt(np.vdot(tmp, tmp))

    def testprob(self, vi_index):
        prob=np.empty(self.whole_size)
        for ui_index in range(self.whole_size):
            _, Phi=self.get_Nodes_and_Phis(ui_index)
            seq=self.tree.index2Bin(ui_index)
            d=np.fromiter(seq, dtype=float)
            prob[ui_index]=np.exp(np.sum([np.log(self.sigmoid(-np.vdot(Phi[i], self.Theta[vi_index]))) \
                for i in range(len(d))]))
        return prob

    def fracfunc(self, ui_index, vi_index, Phi):
        '''
        Main part used to compute gradients for Phi and Theta.
        Gradient for Phi[i]: -fracvec[i]*Theta[vi_index]
        Gradient for Theta[vi_index]: -fracvec.T.dot(Phi)
        '''
        seq=self.tree.index2Bin(ui_index)
        d=np.fromiter(seq, dtype=float)
        fracvec=np.empty_like(d)
        for i in range(len(d)):
            fracvec[i]=d[i]-self.sigmoid(np.vdot(Phi[i], self.Theta[vi_index]))
        return fracvec.reshape((len(fracvec), 1))

    def update(self, ui_index, vi_index):
        '''
        Update ui and vi once.
        '''
        nodes_list, Phi=self.get_Nodes_and_Phis(ui_index)
        fracvec=self.fracfunc(ui_index, vi_index, Phi)
        seq=self.tree.index2Bin(vi_index)
        d=np.fromiter(seq, dtype=float)
        # UPDATE Phi in tree nodes
        for i in range(len(nodes_list)):
            neg_gradi=fracvec[i]*self.Theta[vi_index]
            nodes_list[i].Phi=nodes_list[i].Phi + self.step_size*neg_gradi
        # UPDATE self.Theta
        neg_grad=fracvec.T.dot(Phi)
        self.Theta[vi_index]=self.Theta[vi_index] + self.step_size*neg_grad

    def unit_train(self, unit, n_iter=100, showLoss=False):
        '''
        A unit is a subsequence of walk with size 2*window_size+1.
        Show the total loss before and after a train.
        '''
        w=self.window_size
        if showLoss:
            for i in range(w):
                print(i, 'Before', self.lossfunc(unit[i], unit[w]))
                for j in range(n_iter):
                    self.update(unit[i], unit[w])
                print(i, 'After:', self.lossfunc(unit[i], unit[w]))
            for i in range(w+1, len(unit)):
                print(i, 'Before', self.lossfunc(unit[i], unit[w]))
                for j in range(n_iter):
                    self.update(unit[i], unit[w])
                print(i, 'After:', self.lossfunc(unit[i], unit[w]))
        else:
            for i in range(w):
                for j in range(n_iter):
                    self.update(unit[i], unit[w])
            for i in range(w+1, len(unit)):
                for j in range(n_iter):
                    self.update(unit[i], unit[w])

    def walk_train(self, walk, unit_iter=100, showLoss=False):
        '''
        Train the model using a walk. 
        The main procedule contains flopping a window with size window_size and
            updating Phi and Theta on each window.
        walk: an numpy array with each of its elements being an index.
        '''
        w=self.window_size
        unit_size=2*w+1
        for i in range(len(walk)-unit_size+1):
            # print('walk', '[', i, ':', i+unit_size, ']')
            self.unit_train(walk[i: i+unit_size], unit_iter, showLoss=showLoss)
    


if __name__=='__main__':
    window_size=2
    skipgram=SkipGram(window_size, 30, 10, 0.001)
    print(skipgram.get_Nodes_and_Phis(2))
    print(skipgram.tree.index2Bin(2))
    _, Phi=skipgram.get_Nodes_and_Phis(2)
    print(skipgram.fracfunc(1, 2, Phi))
    skipgram.walk_train([1,2,3,4,5, 6, 7, 8], 1000, False)
