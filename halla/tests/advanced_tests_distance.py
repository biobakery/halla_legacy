import sys
import unittest
import itertools

from halla import distance
from halla import stats

try:
    from numpy import array
except ImportError:
    sys.exit("Please install numpy")


class TestHAllADistanceFunctions(unittest.TestCase):
    """
    Test the functions found in halla.distance
    """
    
    def test_l2(self):
        """
        Test the l2 function
        """
        
        x = array([1,2,3])
        y = array([4,5,6])
        
        expected_result=5.196152422706632
        
        self.assertAlmostEqual(distance.l2(x,y),expected_result)
        
    
    def test_mi(self):
        """
        Test the mutual information function
        """
        
        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
        dx = stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
        dy = stats.discretize( y, iN = None, method = None, aiSkip = [1] )
        p = itertools.product( range(len(x)), range(len(y)) )
        
        expected_result={ 0: 1.0, 1: 0.311278124459, 2: 1.0, 3: 0.311278124459}
        
        for (i,j) in p:
            self.assertAlmostEqual(expected_result[i], distance.mi(dx[i],dy[j]))
            
    def test_ami(self):
        """
        Test the adjusted mutual information function
        """
        
        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
        dx = stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
        dy = stats.discretize( y, iN = None, method = None, aiSkip = [1] )
        p = itertools.product( range(len(x)), range(len(y)) )
        
        expected_result={ 0: 1.0, 1: 2.51758394487e-08, 2: 1.0, 3: -3.72523550982e-08}
        
        for (i,j) in p:
            self.assertAlmostEqual(expected_result[i], distance.ami(dx[i],dy[j]))
        
        
    def test_nmi(self):
        """
        Test the normalized mututal information function
        """
        
        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],[0.015625,0.125,0.421875,1.0]])
        dx = stats.discretize( x, iN = None, method = None, aiSkip = [1,3] )
        dy = stats.discretize( y, iN = None, method = None, aiSkip = [1] )
        p = itertools.product( range(len(x)), range(len(y)) )
        
        expected_result={ 0: 1.0, 1: 0.345592029944, 2: 1.0, 3: 0.345592029944}
        
        for (i,j) in p:
            self.assertAlmostEqual(expected_result[i], distance.nmi(dx[i],dy[j]))        

        
