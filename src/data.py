#!/usr/bin/env python 
'''
Created on Jun 24, 2014

@author: rah
'''
import sys

def _main(NumberOfFeatures=8, numberOfSamples=10000, numberOfBlocks=2, clustercoefficient = .9, alpha = .1):
   simulateData(NumberOfFeatures, numberOfSamples, numberOfBlocks, clustercoefficient, alpha)
    
def simulateData(NumberOfFeatures=8, numberOfSamples=10000, numberOfBlocks=2, clustercoefficient = .9, alpha = .1):
    import numpy as np
    #import matplotlib.pyplot as plt
    linalg = np.linalg
    mean = [1 for _ in range(NumberOfFeatures)]
    cov = np.array([[alpha]*NumberOfFeatures for x in xrange(NumberOfFeatures)])
    blockSize =  NumberOfFeatures/numberOfBlocks
    counter = 0
    for i in range(NumberOfFeatures):
        if i%blockSize==0:
            counter = 0
        for j in range(i,i+blockSize-counter):
            if j>=NumberOfFeatures:
                break
            #f.write i,j
            cov[i,j] = clustercoefficient 
            #cov[j,i] = cov[i,j]
        counter = counter + 1 
    #print cov
    #print linalg.det(cov)
    cov = np.dot(cov, cov.T)
    #print cov
    #print linalg.det(cov)
     
    data = np.random.multivariate_normal(mean, cov, numberOfSamples)
    #L = linalg.cholesky(cov)
   # print data.T
    # print(L.shape)
    # (2, 2)
    #uncorrelated = np.random.standard_normal((NumberOfFeatures,numberOfSamples))
    #data2 = np.dot(L,uncorrelated) + np.array(mean).reshape(NumberOfFeatures,1)
    # print(data2.shape)
    # (2, 1000)
    #plt.scatter(data[:,0], data[:,1], c='yellow')
    #plt.show()
    #f = open('input.csv', 'w')
    #f.write(str(data.T))
    #f.close()
    data = data.T
                
    f = open('syntheticData.txt', 'w')
    f.write(' ')
    for i in range(len(data[1])):
        f.write(str(i)+' ')
        f.write(' ')
    f.write('\n')
    for i in range(len(data)):
        f.write(str(i))
        f.write(' ')
        for j in range(len(data[i])):
            f.write(str(data[i,j]))
            f.write(' ')
        f.write('\n')
    f.close() 
    return data   
   # with open('syntheticData.txt', 'r') as fin:
        #print fin.read()
def writeData(data =None, name =None, rowheader= True, colheader = False, ):
    f = open(name+'_syntheticData.txt', 'w')
    # row numbers as header
    if colheader == True:
        f.write('\t')
        for i in range(len(data[0])):
            f.write(str(i))
            if i< len(data[1])-1:
                f.write('\t')
        f.write('\n')
        
    for i in range(len(data)):
        if rowheader == True:
            f.write(str(i))
            f.write('\t')
        for j in range(len(data[i])):
            f.write(str(data[i,j]))
            if j< len(data[i])-1:
                f.write('\t')
        f.write('\n')
    f.close() 

if __name__ == '__main__':

    _main( )