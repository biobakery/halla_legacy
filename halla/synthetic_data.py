
from random import randrange
import scipy
from scipy.stats import invgamma, norm, uniform, logistic, gamma, lognorm, beta, pareto, pearsonr  
import numpy 
from numpy import array 
from numpy.random import normal, multinomial, dirichlet 
from numpy.linalg import svd
import sklearn
import csv 
import sys 
import itertools
import math 
import time
import argparse
import os
import shutil
try:
    from functools import reduce
except:
    pass
from . import logger, stats


import sys
import numpy as np
import math

def nearest_spd(M):
    # Find the nearest SPD matrix to M
    # http://www.sciencedirect.com/science/article/pii/0024379588902236
    U, s, V = np.linalg.svd(M)
    H = V * s * V.T
    Ms = (M + H) / 2
    return (Ms + Ms.T) / 2

def hard_cov_dataset_generate(nSamples, C, nX, nY):
    # Find a nearby SPD matrix to C, maintaining the zero/nonzero structure of C
    Z = abs(C) < 10*sys.float_info.epsilon
    Sc = math.sqrt(np.sum(pow(C,2)))
    for i in range(0, 500):
        E, V = np.linalg.eig(C)
        if not any(E < 100 * sys.float_info.epsilon):
            break
        
        C = nearest_spd(C)
        C[Z] = 0.0
        C = C * Sc / math.sqrt(np.sum(pow(C,2)))
    
    # Ensure there is no lingering floating point silliness
    E, V = np.linalg.eig(C)
    while any(E < 100 * sys.float_info.epsilon):
        C = C * 0.999 + 0.001 * np.eye(nX+nY)
        E, V = np.linalg.eig(C)
    
    # Generate a dataset
    xy = np.random.multivariate_normal(np.zeros(nX+nY), C, nSamples)
    
    # Split it into pieces to get the two paired datasets
    x = xy[:,:nX].T
    y = xy[:,nX:].T
    
    return (x, y, C[:nX, nX:])

def circular_continuous(nSamples, nX, nY, noiseVar, autoCov, lengthScaleX, lengthScaleY, crossCov, crossLengthScale, fractionZero):
    # Make masks for the parts of the matrix representing X and Y
    isX = np.concatenate([np.ones(nX), np.zeros(nY)])
    isY = np.concatenate([np.zeros(nX), np.ones(nY)])
    pairX = np.multiply.outer(isX, isX)
    pairY = np.multiply.outer(isY, isY)
    
    # Within-dataset "time" dimension
    t = np.concatenate([np.linspace(0, 2*math.pi, num=nX, endpoint=False),
                        np.linspace(0, 2*math.pi, num=nY, endpoint=False)])
    dt = np.subtract.outer(t, t)
    periodicCov = (pairX * autoCov * np.exp(-np.abs(np.sin(dt)) / lengthScaleX) +
                   pairY * autoCov * np.exp(-np.abs(np.sin(dt)) / lengthScaleY) +
                   np.maximum(0.0, (1 - pairX - pairY) * crossCov * np.exp(-np.abs(np.sin(dt)) / crossLengthScale) - fractionZero * crossCov))

    # Put it all together
    C = noiseVar * np.eye(nX+nY) + periodicCov
    
    return hard_cov_dataset_generate(nSamples, C, nX, nY)

def circular_block(nSamples, nX, nY, nBlocks, noiseVar, blockIntraCov, offByOneIntraCov, blockInterCov, offByOneInterCov, holeCov, holeProb):
    # Make masks for the parts of the matrix representing X and Y
    isX = np.concatenate([np.ones(nX), np.zeros(nY)])
    isY = np.concatenate([np.zeros(nX), np.ones(nY)])
    pairX = np.multiply.outer(isX, isX)
    pairY = np.multiply.outer(isY, isY)
    
    # Generate some random clusters
    clusterID = list(np.sort([np.random.randint(1, nBlocks) for i in range(0,nX)])) +\
        list(np.sort([np.random.randint(1, nBlocks) for i in range(0,nY)]))
    #print clusterID
    # Get the differences between clusters
    dClust = np.minimum(
        np.abs(np.subtract.outer(clusterID, clusterID)),
        np.abs(np.subtract.outer(clusterID, np.array(clusterID)-nBlocks)))
    
    # Generate intra-dataset and inter-dataset covariance matrices
    intraC = (blockIntraCov*(dClust==0) + offByOneIntraCov*(dClust==1)) * (pairX+pairY)
    interC = (blockInterCov*(dClust==0) + offByOneInterCov*(dClust==1)) * (1-pairX-pairY)
    
    # Put it all together
    C = noiseVar * np.eye(nX+nY) + intraC + interC

    # introduce holes in C
    holes = np.Infinity * np.ones((nX+nY,nX+nY))
    holePos = np.random.uniform(low=0, high=1, size=(nX+nY,nX+nY)) < holeProb
    holePos[np.tril_indices(nX+nY, 1)] = False
    holes[holePos] = holeCov
    holes = np.minimum(holes, holes.T)
    C = np.minimum(C, holes)
    #print C
    return hard_cov_dataset_generate(nSamples, C, nX, nY)

def rope_unrelated(nSamples, nX, nY, noiseVar, ropeMinCov, ropeMaxCov, fractionTruePositive, crossDatasetCov):
    # Make masks for the parts of the matrix representing X and Y
    isX = np.concatenate([np.ones(nX), np.zeros(nY)])
    isY = np.concatenate([np.zeros(nX), np.ones(nY)])
    pairX = np.multiply.outer(isX, isX)
    pairY = np.multiply.outer(isY, isY)
    
    # Build the within-dataset ropes
    cI = np.concatenate([np.linspace(ropeMinCov, ropeMaxCov, num=nX),
                         np.linspace(ropeMinCov, ropeMaxCov, num=nY)])
    ropeCov = (pairX + pairY) * np.minimum.outer(cI, cI)
    
    # Cross-dataset correlations
    xyCov = crossDatasetCov * (np.random.rand(nX, nY) < fractionTruePositive)
    crossCov = np.vstack([np.hstack([np.zeros((nX, nX)), xyCov]), np.hstack([xyCov.T, np.zeros((nY, nY))])])

    # Put it all together
    C = noiseVar * np.eye(nX+nY) + ropeCov + crossCov
    return hard_cov_dataset_generate(nSamples, C, nX, nY)

#circular_continuous(5, 7, 11, 0.1, 0.5, 0.3, 0.3, 0.3, 0.25, 0.01)
#rope_unrelated(5, 3, 4, 0.1, 0.4, 0.7, 0.5, 0.3)

#A strong relationship exists when the covariance (`crossCov` in `circular_continuous` or `crossDatasetCov` in `rope_unrelated`)
# is near the variance of the variables themselves, which is `noiseVar` + (`autoCov` or `ropeMaxCov`)
#now in the first one, `fractionZero` should be 0-1
#it won't be exactly that many zeros, but bigger numbers will make more zeros
#the other one, you tweak `fractionTruePositive`




def parse_arguments(args):
    """ 
    Parse the arguments from the user
    """
    parser = argparse.ArgumentParser(
        description= "HAllA synthetic data generator to produce paired data sets with association among their features.\n",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-v","--verbose", 
        help="additional output is printed\n", 
        action="store_true",
        default=False)
    parser.add_argument(
        "-f","--features",
        help="number of features in the input file D*N, Rows: D features and columns: N samples \n",
        default = 500,
        type = int,
        required=False)
    parser.add_argument(
        "-n","--samples",
        help="number of samples in the input file D*N, Rows: D features and columns: N samples \n",
        default = 50,
        type = int,
        required=False)
    parser.add_argument(
        "-c","--clusters",
        help="number of blocks, default = min(number_features/2.0, math.log(number_features,2)*1.5) \n",
        default = 0,
        type = int,
        required=False)
    parser.add_argument(
        "-a","--association",
        help="association type [sine, parabola, log, line, L, step, happy_face, default =parabola] \n",
        default = 'parabola',
        required=False)
    parser.add_argument(
        "-d","--distribution",
        help="Distribution [normal, uniform, default =uniform] \n",
        default = 'uniform',
        required=False)
    parser.add_argument(
        "-b","--noise-between",
        dest="noise_between",
        help="noise between associated blocks[0 is no noise, 1 is complete noise, default =0.25]\n",
        default = None,
        type = float,
        required=False)
    parser.add_argument(
        "-w","--noise-within",
        dest = "noise_within",
        help="noise within blocks[0 is no noise, 1 is complete noise, default =0.25]\n",
        default = None,
        type = float,
        required=False)
    parser.add_argument(
        "-o","--output",
        help="the output directory\n",
        required=True)
    parser.add_argument(
        "-s","--structure",
        help="structure [balanced, imbalanced, default =balanced] \n",
        default = 'balanced',
        required=False)
    parser.add_argument(
        "-l","--cluster-percentage",
        dest = "cluster_percentage",
        help="structure [balanced, imbalanced, default =balanced] \n",
        default = 1,
        type = float,
        required=False)

    return parser.parse_args()

def call_data_generator(args):
    number_features = args.features #+ Iter *50 
    number_samples = args.samples #+ Iter * 10
    if args.clusters == 0:
        number_blocks =  int(round(math.sqrt(number_features)* 1.5 +.5 ) ) 
    else:
        number_blocks = args.clusters
    cluster_percentage_l= args.cluster_percentage 
    association_type = args.association
    print ('Synthetic Data Generation ...')
    if association_type == "sine":
        if args.noise_within == None:
            args.noise_within = 0.1 
        if args.noise_between == None:
            args.noise_between = .1
    elif association_type == "log":
        if args.noise_within == None:
            args.noise_within = 0.55
        if args.noise_between == None:
            between_noise = .25  
    elif association_type == "parabola":
        if args.noise_within == None:
            args.noise_within = 0.25 
        if args.noise_between == None:
            args.noise_between = .15
    elif association_type == "line":
        if args.noise_within == None:
            args.noise_within = 0.55
        if args.noise_between == None:
            args.noise_between = 0.5
    elif association_type == "step":
        if args.noise_within == None:
            args.noise_within = 0.15
        if args.noise_between == None:
            args.noise_between = .25
    elif association_type == "L":
        if args.noise_within == None:
            args.noise_within = 0.05
        if args.noise_between == None:
            args.noise_between = 0.05
    elif association_type == "mixed":
        if args.noise_within == None:
            args.noise_within = 0.01
        if args.noise_between == None:
            args.noise_between = 0.01
    elif association_type == "happyface":
        if args.noise_within == None:
            args.noise_within = 0.05
        if args.noise_between == None:
            args.noise_between = 0.0
    elif association_type == "random":
        X = np.random.uniform(low=-1,high=1 ,size=(number_features, number_samples ))
        Y = np.random.uniform(low=-1,high=1,size=(number_features, number_samples))
        A = np.zeros((len(X),len(Y)))
     
    if args.structure == "imbalanced":
        if association_type == "structured_random":
            X,_,_ = halla.synthetic_data.imbalanced_synthetic_dataset_uniform( D = number_features, N = number_samples,\
                                                                               B = number_blocks, within_noise= args.noise_within , between_noise = args.noise_between, cluster_percentage = cluster_percentage_l, association_type ='parabola' ) 
            _,Y,_ = halla.synthetic_data.imbalanced_synthetic_dataset_uniform( D = number_features, N = number_samples\
                                                    , B = number_blocks, within_noise =args.noise_within  , between_noise = args.noise_between ,\
                                                   cluster_percentage = cluster_percentage_l, association_type ='parabola' ) 
            A = np.zeros((len(X),len(Y)))
        
        if args.distribution == "norm":
            X,Y,A = imbalanced_synthetic_dataset_norm( D = number_features, N = number_samples, \
                                                                               cluster_percentage = cluster_percentage_l , \
                                                                               B = number_blocks, within_noise = args.noise_within, \
                                                                               between_noise = args.noise_between ,association_type =association_type ) 
        elif args.distribution == "uniform":
            X,Y,A = imbalanced_synthetic_dataset_uniform( D = number_features, N = number_samples\
                                                    , B = number_blocks, within_noise = args.noise_within , between_noise = args.noise_between ,\
                                                   cluster_percentage = cluster_percentage_l, association_type =association_type )
    else:#if args.structure == "balanced":
        if association_type == "structured_random":
            X,_,_ = balanced_synthetic_dataset_uniform( D = number_features, N = number_samples,\
                                                                               B = number_blocks, within_noise= args.noise_within , between_noise = args.noise_between, cluster_percentage = cluster_percentage_l, association_type ='parabola' ) 
            _,Y,_ = balanced_synthetic_dataset_uniform( D = number_features, N = number_samples\
                                                    , B = number_blocks, within_noise =args.noise_within  , between_noise = args.noise_between ,\
                                                   cluster_percentage = cluster_percentage_l, association_type ='parabola' ) 
            A = np.zeros((len(X),len(Y)))
        
        if args.distribution == "norm":
            X,Y,A = balanced_synthetic_dataset_norm( D = number_features, N = number_samples, \
                                                                               cluster_percentage = cluster_percentage_l , \
                                                                               B = number_blocks, within_noise = args.noise_within, \
                                                                               between_noise = args.noise_between ,association_type =association_type ) 
        elif args.distribution == "uniform":
            X,Y,A = balanced_synthetic_dataset_uniform( D = number_features, N = number_samples\
                                                    , B = number_blocks, within_noise = args.noise_within , between_noise = args.noise_between ,\
                                                   cluster_percentage = cluster_percentage_l, association_type =association_type )  
    create_dir(args.output)
    logger.write_table(X, args.output+"/X_"+\
                             association_type+"_"+str(number_features)+"_"+\
                             str(number_samples)+".txt", prefix = "X", corner = "#")
    logger.write_table(Y, args.output+"/Y_"+\
                             association_type+"_"+str(number_features)+"_"+\
                             str(number_samples)+".txt", prefix = "Y", corner = "#")
    rowheader = ["X"+str(i) for i in range(len(X))]
    colheader = ["Y"+str(i) for i in range(len(Y))]
    logger.write_table(A, args.output+"/A_"+ \
                             association_type+"_"+str(number_features)+"_"+\
                             str(number_samples)+".txt", rowheader=rowheader, colheader=colheader,\
                             prefix = "", corner = "#")

def happyface(x):
    # Head
    arch = math.sqrt(1.0 - x*x)
    components = [arch, -arch]
    
    # Eyes
    eyeWidth = 0.2
    eyeHeight = eyeWidth * 0.6
    eyeX = 1.0 / 3.0
    eyeY = 0.25
    eyeCoord = (abs(x) - eyeX) / eyeWidth
    if eyeCoord > -1.0 and eyeCoord < 1.0:
        eyeArch = math.sqrt(1.0 - eyeCoord*eyeCoord)
        components.append(eyeY + eyeHeight * eyeArch)
        #components.append(eyeY - eyeHeight * eyeArch)
    
    # Mouth
    mouthRads = math.pi * 0.7
    mouthLowerRadius = 0.85
    mouthUpperRadius = 3
    mouthYoffset = 0.15
    mouthW2 = mouthLowerRadius * math.sin(mouthRads / 2.0)
    if abs(x) < mouthW2:
        # Lower lip
        mouthLowerCoord = x / mouthLowerRadius
        mouthLowerArch = math.sqrt(1.0 - mouthLowerCoord*mouthLowerCoord)
        components.append(-mouthLowerRadius*mouthLowerArch + mouthYoffset)
        
        # Upper lip
        mouthUpperH = math.sqrt(mouthUpperRadius*mouthUpperRadius - mouthW2*mouthW2) - mouthLowerRadius * math.cos(mouthRads / 2.0)
        mouthUpperCoord = x / mouthUpperRadius
        mouthUpperArch = math.sqrt(1.0 - mouthUpperCoord*mouthUpperCoord)
        components.append(mouthUpperH - mouthUpperRadius*mouthUpperArch + mouthYoffset)

    # Pick one of the components
    i = numpy.random.randint( len(components), size = 1)
    return components[i]
    
def random_dataset( D, N,):
    """
        D: int
            number of features

        N: int
            number of samples 

        B: int
            number of blocks 
    """
    X = numpy.random.uniform(low=-1,high=1 ,size=(D,N))
    Y = numpy.random.uniform(low=-1,high=1,size=(D,N))
    A = numpy.zeros( (D,D) )
    return X,Y,A    
def orthogonalize_matrix(w):
    U, s, V =  svd(w.T, full_matrices=False)
    S = numpy.diag(s)
    #print U.shape, V.shape, s.shape
    #print (numpy.dot(U,S)).shape
    #print np.allclose(w.T, np.dot(U, np.dot(S, V)))
    return numpy.dot(U,S).T
def imbalanced_synthetic_dataset_uniform(D, N, B, cluster_percentage = 1, within_noise = 0.5, between_noise = 0.5, association_type = 'parabola' ):
    """
        D: int
            number of features

        N: int
            number of samples 

        B: int
            number of blocks 
    """
    
    X = numpy.random.uniform(low=-1,high=1 ,size=(D,N))
    Y = numpy.random.uniform(low=-1,high=1,size=(D,N))
    
    common_base = numpy.random.uniform(low=-1,high=1 ,size=(B+1,N))
    X_base = numpy.random.uniform(low=-1,high=1 ,size=(D,N))
    Y_base = numpy.random.uniform(low=-1,high=1 ,size=(D,N))
    
    X_base = orthogonalize_matrix(X_base)
    Y_base = orthogonalize_matrix(Y_base)
    #X = orthogonalize_matrix(X)
    #Y = orthogonalize_matrix(Y)
    #common_base = orthogonalize_matrix(common_base)
    A = numpy.zeros( (D,D) )
    blockSize = int(round(D/B))
    #print D, B, blockSize
    '''for i in range(0,D,blockSize):
        for j in range(i,i+blockSize):
            if j < D:
                X[j]= [X[i,k]  + numpy.random.normal(0,.1,1) for k in range(len(X[j]))]
    '''
    print ("Number of features %s, number of samples: %s, number of clusters: %s, number of features with each cluster: %s")\
         %(D, N, B, blockSize)
    if association_type == "L":
        common_base = numpy.hstack((numpy.random.uniform(low=-1.0,high=-1.0 ,size=(B+1,N/2)), numpy.random.uniform(low=-1,high=50 ,size=(B+1,N/2))))
        for l in range(B+1):
            common_base[l]= numpy.random.permutation(common_base[l])
    else:
        common_base = numpy.random.uniform(low=-1,high=1 ,size=(B+1,N))
    if association_type == "L":
        common_base_Y = numpy.random.uniform(low=-1,high=1 ,size=(B+1,N))
        for l in range(B+1):
            common_base_Y[l] = [numpy.random.uniform(low=l,high=100, size=1)  if common_base[l,k] < -0.999 else numpy.random.uniform(low=l,high=l, size=1) for k in range(N)]
        common_base_Y = orthogonalize_matrix(common_base_Y)
    #common_base = orthogonalize_matrix(common_base)
    
    number_associated_blocks =  max([int(B * cluster_percentage) , 1])
    assoc1 = [[] for i1 in range(number_associated_blocks)]
    assoc2 = [[] for j1 in range(number_associated_blocks)]
    
    for i in range(int(D*cluster_percentage)):
        l= randrange(0,number_associated_blocks)
        numpy.random.seed()
        if association_type == "L":
                    X[i]= [common_base[l,k]+ within_noise * numpy.random.uniform(low=-.1,high=.1, size=1) for k in range(N)]
        elif association_type == "log":
            X[i]= [math.fabs(common_base[l,k])  + within_noise * numpy.random.uniform(low=0,high=1 ,size=1) for k in range(N)]
        else:    
            X[i]= [common_base[l,k]  + within_noise * numpy.random.uniform(low=-1,high=1 ,size=1) for k in range(N)]
        #X[i]= [common_base[l][k] +  within_noise * numpy.random.uniform(low=-1,high=1 ,size=1) for k in range(N)]
        assoc1[l].append(i)
    
    noise_num = numpy.random.randint(N, size=int(N*between_noise))
    for i in  range(int(D*cluster_percentage)):
        l= randrange(0,number_associated_blocks)
        #slope = 1.0 +numpy.random.random_sample()#slope = numpy.random.random_sample()
        numpy.random.seed()
        if association_type == "parabola":
            Y[i]= [common_base[l,k]*common_base[l,k]  + within_noise *math.sqrt(math.fabs(numpy.random.uniform(low=-1,high=1 ,size=1))) for k in range(N)]
        elif association_type == "line":
            Y[i]= [common_base[l,k]  + within_noise * numpy.random.uniform(low=-1,high=1 ,size=1) for k in range(N)]
        elif association_type == "sine":
            Y[i]= [.5* math.sin(math.pi * common_base[l,k]*1.5)  + within_noise * numpy.random.uniform(low=-1,high=1 ,size=1)  for k in range(N)]
        elif association_type == "log":
            Y[i]= [math.log(math.fabs(common_base[l,k]))  + within_noise *math.fabs(numpy.random.uniform(low=0,high=1 ,size=1)) for k in range(N)]
        elif association_type == "step":
            p1 = numpy.percentile(common_base[l], 25)
            p2 = numpy.percentile(common_base[l], 50)
            p3 = numpy.percentile(common_base[l], 75)
            #NOISE = within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)
            #Y[j]= [ 0 if common_base[l,k] <p1 else  1 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)\
            #     if common_base[l,k] <p2 else 2+within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)\
            #     if common_base[l,k] <p3 else 3 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)  for k in range(N)]
            Y[i]= [ 2.0 +within_noise *numpy.random.uniform(low=-1,high=1 ,size=1) if common_base[l,k] < p1 else  1.0 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)\
                if common_base[l,k] < p2 else  3.0 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1) if common_base[l,k] < p3  else
                0.0 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1) for k in range(N)]
        elif association_type == "L":
            Y[i] = [ common_base_Y[l,k] + within_noise * numpy.random.uniform(low=-1,high=1, size=1) for k in range(N)]
            #Y[j]= [ numpy.random.uniform(low=10,high=100, size=1) * common_base[l,k] if common_base[l,k] < -0.8 else numpy.random.uniform(low=.2,high=.5, size=1) for k in range(N)]
        elif association_type =="happyface":
            Y[i] = [ happyface(common_base[l,k]) + within_noise * numpy.random.uniform(low=-.1,high=.1, size=1) for k in range(N)]
            
        for index,b in enumerate(noise_num):
            Y[i][b] = Y[i][index]
        assoc2[l].append(i)

    for a in range(number_associated_blocks):
        #print assoc1[a], assoc2[a]
        for i, j in itertools.product(assoc1[a], assoc2[a]):
            A[i][j] = 1
    return X,Y,A

def balanced_synthetic_dataset_uniform(D, N, B, within_noise = 0.5, between_noise = 0.1, cluster_percentage = 1, association_type = 'parabola', number_of_cat = None ):
    """
        D: int
            number of features

        N: int
            number of samples 

        B: int
            number of blocks 
    """
    if association_type == "categorical-step" or association_type == "mixed":
        cat_X = numpy.empty((D, N), dtype=object)
        cat_Y = numpy.empty((D, N), dtype=object)
    
    X = numpy.random.uniform(low=-1,high=1,size=(D,N))
    Y = numpy.random.uniform(low=-1,high=1,size=(D,N))
    A = numpy.zeros( (len(X),len(Y)) )
    blockSize =  int(D/B) #int(round(D/B+ 0.4999999999999))
    print ("Number of features %s, number of samples: %s, number of clusters: %s, number of features within each cluster: %s")\
         %(D, N, B, blockSize)
    if association_type == "L":
        '''if N % 2 != 0:
            N +=1 
        else:
            pass # Odd'''
        common_base = numpy.hstack((numpy.random.uniform(low=-.1,high=0.1 ,size=(B,N/2)), numpy.random.uniform(low=-1.0,high=10.0 ,size=(B,N/2))))
        for l in range(B):
            common_base[l]= numpy.random.permutation(common_base[l])
    else:
        common_base = numpy.random.uniform(low=-1,high=1 ,size=(B, N))
    common_base = orthogonalize_matrix(common_base)
    
    assoc = [[] for i in range(B)]
    l = 0
    for i in range(0, int(D * cluster_percentage), blockSize):
        for j in range(i, i + blockSize):
            if j < D:
                numpy.random.seed(j)
                if association_type == "L":
                    X[j]= [common_base[l,k] + within_noise * numpy.random.uniform(low=-.1, high=.1, size=1) for k in range(N)]
                elif association_type == "log":
                    X[j]= [math.fabs(common_base[l,k]) + within_noise * numpy.random.uniform(low=0, high=1 ,size=1) for k in range(N)]
                else:    
                    #print i, j, l, len(X), len(X[j])
                    X[j] = [common_base[l, k] + within_noise * numpy.random.uniform(low=-1, high=1 ,size=1) for k in range(N)]
                    assoc[l].append(j)               
        l += 1
        if l>=B:
            break
    if association_type == "L":
        common_base_Y = numpy.random.uniform(low=-1,high=1 ,size=(B,N))
        for l in range(B):
            common_base_Y[l] = [numpy.random.uniform(low=l,high=10, size=1)  if common_base[l,k] < -0.99 else 
                                l+ within_noise * numpy.random.uniform(low=-.1,high=.1, size=1) for k in range(N)]
    l= 0
    for i in range(0,int(D*cluster_percentage),blockSize):
        numpy.random.seed()
        noise_num = numpy.random.randint(N, size=int(N*between_noise))
        for j in range(i,i+blockSize):
            if j < D:
                numpy.random.seed()
                if association_type == "parabola":
                    Y[j] = [common_base[l,k]*common_base[l,k]  + within_noise *math.sqrt(math.fabs(numpy.random.uniform(low=-1,high=1 ,size=1))) for k in range(N)]
                elif association_type == "line":
                    Y[j]= [common_base[l,k]  + within_noise * numpy.random.uniform(low=-1,high=1 ,size=1) for k in range(N)]
                elif association_type == "sine":
                    Y[j]= [.5* math.sin(math.pi * common_base[l,k]*1.5)  + within_noise * numpy.random.uniform(low=-1,high=1 ,size=1)  for k in range(N)]
                elif association_type == "log":
                    Y[j]= [math.log(math.fabs(common_base[l,k]))  + within_noise *math.fabs(numpy.random.uniform(low=0,high=1 ,size=1)) for k in range(N)]
                elif association_type == "step" or association_type == "categorical-step" or association_type == "mixed":
                    p1 = numpy.percentile(common_base[l], 25)
                    p2 = numpy.percentile(common_base[l], 50)
                    p3 = numpy.percentile(common_base[l], 75)
                    #NOISE = within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)
                    #Y[j]= [ 0 if common_base[l,k] <p1 else  1 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)\
                    #     if common_base[l,k] <p2 else 2+within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)\
                    #     if common_base[l,k] <p3 else 3 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)  for k in range(N)]
                    Y[j]= [ 2.0 +within_noise *numpy.random.uniform(low=-1,high=1 ,size=1) if common_base[l,k] < p1 else  1.0 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1)\
                        if common_base[l,k] < p2 else  3.0 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1) if common_base[l,k] < p3  else
                        0.0 + within_noise *numpy.random.uniform(low=-1,high=1 ,size=1) for k in range(N)]
                
                elif association_type == "L":
                    Y[j] = [ common_base_Y[l,k] + within_noise * numpy.random.uniform(low=-.1,high=.1, size=1) for k in range(N)]
                    #Y[j]= [ numpy.random.uniform(low=10,high=100, size=1) * common_base[l,k] if common_base[l,k] < -0.8 else numpy.random.uniform(low=.2,high=.5, size=1) for k in range(N)]
                elif association_type =="happyface":
                    Y[j] = [ happyface(common_base[l,k]) + within_noise * numpy.random.uniform(low=-.1,high=.1, size=1) for k in range(N)]
                    
                for index,b in enumerate(noise_num):
                    Y[j][b] = Y[j][index]
        l += 1
        if l>=B:
            break
    for r in range(B):
        for i, j in itertools.product(assoc[r], assoc[r]):
            A[i][j] = 1
    if association_type == "categorical-step":
        xdisc_value = stats.discretize(X, number_of_bins = number_of_cat)
        ydisc_value = stats.discretize(Y, number_of_bins = number_of_cat)
        for i in range(D):
            #try:
            cat_X[i] = [chr(int(value)+64) for value in xdisc_value[i]]
            cat_Y[i] = [chr(int(value)+64) for value in ydisc_value[i]]
                #print "X", cat_X[i]
            #except:
            #   print( 'issue with categorical data generation')
        return cat_X, cat_Y , A
    elif association_type == "mixed":
        xdisc_value = stats.discretize(X)
        ydisc_value = stats.discretize(Y)
        for i in range(D):
            #try:
            if i < math.trunc((D/10)):
                cat_X[i] = [chr(int(value)+64) for value in xdisc_value[i]]
                cat_Y[i] = [chr(int(value)+64) for value in ydisc_value[i]]
            else:
                cat_X[i] = X[i]
                cat_Y[i] = Y[i]
        return cat_X, cat_Y , A
    else:      
        return X,Y,A
def balanced_synthetic_dataset_norm( D, N, B, within_noise = 0.5, between_noise = 0.1, cluster_percentage = 1, association_type = 'parabola' ):
    """
        D: int
            number of features

        N: int
            number of samples 

        B: int
            number of blocks 
    """
    X = numpy.random.normal(0, 1,size=(D,N))
    common_base = numpy.random.normal(0, 1,size=(B+1,N))
    common_base = orthogonalize_matrix(common_base)
    Y = numpy.random.normal(0, 1,size=(D,N))
    A = numpy.zeros( (len(X),len(Y)) )
    blockSize = int(round(D/B+.5))
    print (D, B, blockSize)
    assoc = [[] for i in range((B+1))]
    l = 0
    for i in range(0, int(D*cluster_percentage), blockSize):
        numpy.random.seed()
        for j in range(i,i+blockSize):
            if j < D :
                X[j]= [common_base[l,k]  + numpy.random.normal(0, within_noise, size=1) for k in range(N)]
                assoc[l].append(j)
        l += 1
    l= 0
    for i in range(0, int(D*cluster_percentage), blockSize):
        #print N, int(N*between_noise)
        numpy.random.seed()
        noise_num = numpy.random.randint(N, size= int(N*between_noise) )
        #print noise_num
        for j in range(i,i+blockSize):
            if j < D :
                #print j, i
                Y[j]= [common_base[l,k]  + numpy.random.normal(0, within_noise,size=1) for k in range(N)]
                for index,b in enumerate(noise_num):
                    Y[j][b] = Y[j][index]#print l
        l += 1
        
    for r in range(0,B+1):
        for i, j in itertools.product(assoc[r], assoc[r]):
            A[i][j] = 1
    return X,Y,A    
def imbalanced_synthetic_dataset_norm(  D, N, B, within_noise = 0.5, between_noise = 0.1, association_type = 'parabola' ):
    """
        D: int
            number of features

        N: int
            number of samples 

        B: int
            number of blocks 
    """
    X = numpy.random.normal(0, 1,size=(D,N))
    X_base = numpy.random.normal(0, 1,size=(D,N))
    common_base = numpy.random.normal(0, 1,size=(B+1,N))
    common_base = orthogonalize_matrix(common_base)
    Y = numpy.random.normal(0, 1,size=(D,N))
    Y_base = numpy.random.normal(0, 1,size=(D,N))
    A = numpy.zeros( (len(X),len(Y)) )
    blockSize = int(round(D/B))
    #print D, B, blockSize
    
    
    number_associated_blocks = max([int(B/4) , 1])
    assoc1 = [[] for i in range(number_associated_blocks)]
    assoc2 = [[] for i in range(number_associated_blocks)]
    
    for i in range(D):
        r = randrange(0,B)
        X[i]= [X_base[r,k]  + within_noise * numpy.random.normal(0, within_noise, size=1) for k in range(N)]
        
    for i in range(D):
        r = randrange(0,B)
        Y[i]= [Y_base[r,k]  + within_noise * numpy.random.normal(0, within_noise, size=1) for k in range(N)]
        
    r = numpy.random.randint(D, size=int(blockSize* number_associated_blocks))
    for i in r:
        l= randrange(0,int(number_associated_blocks))
        X[i]= [common_base[l][k] +  within_noise * numpy.random.normal(0, within_noise, size=1) for k in range(N)]
        assoc1[l].append(i)
    
    noise_num = numpy.random.randint(N, size=int(N*between_noise))
    for i in r:
        numpy.random.seed()
        l= randrange(0,int(number_associated_blocks))
        #slope = 1.0 +numpy.random.random_sample()#slope = numpy.random.random_sample()
        if association_type == "parabola":
            Y[i]= [common_base[l][k] * common_base[l][k]  + within_noise * math.sqrt(math.fabs(numpy.random.normal(0, within_noise, size=1)))  for k in range(N)]
            for index,b in enumerate(noise_num):
                Y[i][b] = Y[i][index]
            assoc2[l].append(i)
        elif association_type == "linear":
            Y[i]= [common_base[l][k] + within_noise * numpy.random.normal(0, within_noise, size=1)  for k in range(N)]
            for index,b in enumerate(noise_num):
                Y[i][b] = Y[i][index]
            assoc2[l].append(i)

    for a in range(number_associated_blocks):
        #print assoc1[a], assoc2[a]
        for i, j in itertools.product(assoc1[a], assoc2[a]):
            A[i][j] = 1
    return X,Y,A
def create_dir(dir): 
    if os.path.isdir(dir):
        try:
            shutil.rmtree(dir)
            #shutil.rmtree(dir)
            #shutil.rmtree(discretized_dir)
        except EnvironmentError:
            sys.exit("Unable to remove directory: "+dir)
    
    # create new directory
    try:
        os.mkdir(dir)
        #os.mkdir(dir)
        #if not bypass_discretizing():
            #os.mkdir(discretized_dir)
    except EnvironmentError:
        sys.exit("Unable to create directory: "+dir)
def main( ):
    # Parse arguments from command line
    args=parse_arguments(sys.argv)
    call_data_generator(args)
if __name__ == "__main__":
    main( )   