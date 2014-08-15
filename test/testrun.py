'''
Created on Jun 6, 2014

@author: susa
'''
import unittest
import halla.hallaclass
#from halla  import HAllA

class Test(unittest.TestCase):


    def testName(self):
        X = [[1,2,4,6,8], [.1,.2,.4,.6,.8], [100,50,25,12,6],[10,5,2.5,1.2,.6]]
        Y = X 
        h = halla.HAllA(X, Y)
        h.run()
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()