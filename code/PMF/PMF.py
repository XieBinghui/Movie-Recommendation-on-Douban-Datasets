# coding: utf-8

'''
    This is an naive implementation for Probablistic Matrix Factorization
    , a classic algorithm in the arena of collaborative filtering by 
    Ruslan Salakhutdinov and Andriy Mnih.
'''

import numpy as np
from dataAPI import dataAPI
from matplotlib import pyplot as plt

class PMF(object):
    '''
    Naive implementation for the initial version of PMF
    '''
    def __init__(self, stepsize, D_dimension, batchNum, epochNum, R, lambda_u, lambda_v, UVsize):
        self.stepsize=stepsize
        self.dim=D_dimension
        self.batchNum=batchNum
        self.epochNum=epochNum
        
        self.lambda_u=lambda_u
        self.lambda_v=lambda_v
        # initialize U and V, for efficiency, U and V are transposed,
        # as their storage in memory is by row 
        self.U=0.1*np.random.randn(UVsize[0], D_dimension)
        self.V=0.1*np.random.randn(UVsize[1], D_dimension)
        self.UVsize=UVsize
        self.R=R
        self.I=np.sign(R)

    def setUV(self, U, V):
        self.U=U
        self.V=V

    def objectiveFunc(self, U, V, ratings):
        '''
        Objective function, also equation(4)*2 in the paper.
        Note that we leave out the 1/2 in the error equation here for simplicity.
        '''
        sec1=sum([(Rij-np.vdot(U[i-1], V[j-1]))**2 for i, j, Rij in ratings])
        sec2=self.lambda_u*sum([np.vdot(U[i-1], U[i-1]) for i in range(self.UVsize[0])])
        sec3=self.lambda_v*sum([np.vdot(V[j-1], V[j-1]) for j in range(self.UVsize[1])])
        return sec1+sec2+sec3

    def gradient(self):
        '''
        Return the gradient matrices gradU and gradV
        '''
        gradU=np.empty((self.UVsize[0], self.dim))
        gradV=np.empty((self.UVsize[1], self.dim))

        gradU=2*(np.dot(self.I*(np.dot(self.U, self.V.T)-self.R), self.V)+\
                self.lambda_u*self.U)
        
        gradV=2*(np.dot((self.I*(np.dot(self.U, self.V.T)-self.R)).T, self.U)+\
                self.lambda_v*self.V)

        return (gradU, gradV)

    def RMSE(self, ratings):
        '''
        Compute Root Mean Square Error between ratings and results
        from training.
        ratings: numpy array with size being (n, 3), where each row
                is a triple of User-Index+1, Movie-Index+1, TrueRating.
        '''
        tmp=0
        for i, j, Rij in ratings:
            tmp+=(np.vdot(self.U[i-1], self.V[j-1])-Rij)**2
        return np.sqrt(tmp/len(ratings))
            
    def update(self, trainingSet, beta=0.6):
        t=self.stepsize
        failcount=0

        gradU, gradV=self.gradient()
        trLoss=self.objectiveFunc(self.U, self.V, trainingSet)
        trLoss_old=trLoss
        # Armijo condition
        while trLoss>=trLoss_old:
            print('fail once:', t)
            failcount+=1

            t=t*beta
            trLoss=self.objectiveFunc(self.U-t*gradU, self.V-t*gradV, trainingSet) 
        self.U=self.U-t*gradU
        self.V=self.V-t*gradV
        
        if failcount>=3:
            self.stepsize=t

    def fit(self, trainingSet, testingSet, beta=0.6):
        epoch=0
        rmse=np.empty(self.epochNum)
        while epoch<self.epochNum:
            rmse[epoch]=self.RMSE(testingSet)
            self.update(trainingSet, beta)
            print('Epoch:', epoch, ' | ', 'RMSE_to_testingSet:', rmse[epoch])
            epoch+=1
        
        return rmse

    def saveData(self, rmse=None, saveUV=True, saveRMSE=True):
        if saveUV:
            np.save('U_PMF.npy', self.U)
            np.save('V_PMF.npy', self.V)
            print('U, V saved')
        if saveRMSE:
            np.save('RMSE_PMF.npy', rmse)
            print('rmse saved')

    def DCG(self, true_Ri, model_Ri, k=5):
        '''
        true_Ri: true rates from user information
        model_Ri: Rates from PMF
        k: DCG will be computed by the top k rates
        '''
        order=np.argsort(model_Ri)[::-1]
        y_true=np.take(true_Ri, order[:k])
        gain=2**y_true-1
        discounts=np.log2(np.arange(len(y_true))+2)
        return np.sum(gain/discounts)

    def NDCG(self, true_R, model_R, k=5):
        scores=np.zeros((len(true_R), 1))
        count=0
        for i in range(len(true_R)):
            IDCGi=self.DCG(true_R[i], true_R[i], k)
            DCGi=self.DCG(true_R[i], model_R[i], k)
            # print(DCGi, IDCGi)
            if IDCGi and DCGi:
                scores[i]=DCGi/IDCGi
                count+=1
        return np.sum(scores)/count




if __name__=='__main__':
    api=dataAPI()
    ratings, UVsize=api.fetchRatings(fromDB=True)
    R=api.generateRemark(UVsize, ratings, fromFile=False)
    # api.splitSets(0.2)
    
    trainingSet=api.readTrainingSet()
    testingSet=api.readTestingSet()

    pmf=PMF(0.001, 32, 1, 500, R, 0.1, 0.1, UVsize)
    U=np.load('U_PMF.npy')
    V=np.load('V_PMF.npy')
    pmf.setUV(U, V)
    # rmse=pmf.fit(trainingSet, testingSet, 0.6)
    # pmf.saveData(rmse=rmse)
    model_R=np.dot(U, V.T)+5
    print(model_R)
    print(np.argsort(model_R[2])[::-1][:10]+1)
    print(pmf.RMSE(trainingSet))
    print(pmf.NDCG(R, model_R))



