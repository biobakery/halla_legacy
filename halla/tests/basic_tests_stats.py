import sys
import unittest

from halla import stats

try:
    from numpy import array
except ImportError:
    sys.exit("Please install numpy")

class TestHAllAStatsFunctions(unittest.TestCase):
    """
    Test the functions found in halla.stats
    """
    
    def test_discretize_tenths(self):
        """
        Test the discretize function on four values of tenths
        """
        
        expected_result=[0, 0, 1, 1]
        result=stats.discretize([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(expected_result,result)
        
    def test_discretize_squares(self):
        """
        Test the discretize function on four values of squares
        """
        
        expected_result=[0, 0, 1, 1]
        result=stats.discretize([0.01, 0.04, 0.09, 0.16])
        self.assertEqual(expected_result,result)
        
    def test_discretize_negatives(self):
        """
        Test the discretize function on all negative values
        """
        
        expected_result=[1, 1, 0, 0]
        result=stats.discretize([-0.1, -0.2, -0.3, -0.4])
        self.assertEqual(expected_result,result)
        
    def test_discretize_quarters(self):
        """
        Test the discretize function on four values of quarters
        """
        
        expected_result=[0, 0, 1, 1]
        result=stats.discretize([0.25, 0.5, 0.75, 1.00])
        self.assertEqual(expected_result,result)
        
    def test_discretize_eights(self):
        """
        Test the discretize function on four values of eigths
        """
        
        expected_result=[0, 0, 1, 1]
        result=stats.discretize([0.015625, 0.125, 0.421875, 1])
        self.assertEqual(expected_result,result)
        
    def test_discretize_zero(self):
        """
        Test the discretize function on an array containing a single zero
        """
        
        expected_result=[0]
        result=stats.discretize([0])
        self.assertEqual(expected_result,result)
        
    def test_discretize_one(self):
        """
        Test the discretize function on an array of [0,1]
        """
        
        expected_result=[0,0]
        result=stats.discretize([0,1])
        self.assertEqual(expected_result,result)
        
    def test_discretize_two_bins_two_values(self):
        """
        Test the discretize function two values with two bins
        """
        
        expected_result=[0,1]
        result=stats.discretize([0, 1], 2)
        self.assertEqual(expected_result,result)
        
    def test_discretize_two_bins_two_values_reverse(self):
        """
        Test the discretize function on two values (revered order) with two bins
        """
        
        expected_result=[1,0]
        result=stats.discretize([1, 0], 2)
        self.assertEqual(expected_result,result)
        
    def test_discretize_three_bins_three_values(self):
        """
        Test the discretize function on three values with three bins
        """
        
        expected_result=[1, 0, 2]
        result=stats.discretize([0.2, 0.1, 0.3], 3)
        self.assertEqual(expected_result,result)

    def test_discretize_one_bin_three_values(self):
        """
        Test the discretize function on three values with one bin
        """
        
        expected_result=[0, 0, 0]
        result=stats.discretize([0.2, 0.1, 0.3], 1)
        self.assertEqual(expected_result,result)
        
    def test_discretize_two_bins_three_values(self):
        """
        Test the discretize function on three values with two bins
        """
        
        expected_result=[0, 0, 1]
        result=stats.discretize([0.2, 0.1, 0.3], 2)
        self.assertEqual(expected_result,result)
        
    def test_discretize_two_bins_four_values_all_floats(self):
        """
        Test the discretize function on four values (all floats) with two bins
        """
        
        expected_result=[1, 0, 0, 1]
        result=stats.discretize([0.4, 0.2, 0.1, 0.3], 2)
        self.assertEqual(expected_result,result)
        
    def test_discretize_two_bins_four_values_one_int(self):
        """
        Test the discretize function on four values (one int) with two bins
        """
        
        expected_result=[1, 0, 0, 1]
        result=stats.discretize([4, 0.2, 0.1, 0.3], 2)
        self.assertEqual(expected_result,result)
        
    def test_discretize_five_values(self):
        """
        Test the discretize function on five values with default bins
        """
        
        expected_result=[1, 0, 0, 0, 1]
        result=stats.discretize([0.4, 0.2, 0.1, 0.3, 0.5])
        self.assertEqual(expected_result,result)
        
    def test_discretize_three_bins_five_values(self):
        """
        Test the discretize function on five values with three bins
        """
        
        expected_result=[1, 0, 0, 1, 2]
        result=stats.discretize([0.4, 0.2, 0.1, 0.3, 0.5], 3)
        self.assertEqual(expected_result,result)
        
    def test_discretize_six_values(self):
        """
        Test the discretize function on six values with default bins
        """
        
        expected_result=[1, 0, 1, 0, 0, 1]
        result=stats.discretize([0.4, 0.2, 0.6, 0.1, 0.3, 0.5])
        self.assertEqual(expected_result,result)
        
    def test_discretize_three_bins_six_values(self):
        """
        Test the discretize function six values with three bins
        """
        
        expected_result=[1, 0, 2, 0, 1, 2]
        result=stats.discretize([0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 3)
        self.assertEqual(expected_result,result)
        
    def test_discretize_zero_bins_six_values(self):
        """
        Test the discretize function on six values with zero bins
        """
        
        expected_result=[3, 1, 5, 0, 2, 4]
        result=stats.discretize([0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 0)
        self.assertEqual(expected_result,result)
        
    def test_discretize_six_bins_six_values(self):
        """
        Test the discretize function on six values with six bins
        """
        
        expected_result=[3, 1, 5, 0, 2, 4]
        result=stats.discretize([0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 6)
        self.assertEqual(expected_result,result)
        
    def test_discretize_sixty_bins_six_values(self):
        """
        Test the discretize function on six values with sixty bins
        """
        
        expected_result=[3, 1, 5, 0, 2, 4]
        result=stats.discretize([0.4, 0.2, 0.6, 0.1, 0.3, 0.5], 60)
        self.assertEqual(expected_result,result)
        
    def test_discretize_two_bins_eight_values(self):
        """
        Test the discretize function on eight values with two bins
        """
        
        expected_result=[0, 0, 0, 0, 0, 0, 1, 1]
        result=stats.discretize([0, 0, 0, 0, 0, 0, 1, 2], 2)
        self.assertEqual(expected_result,result)
        
    def test_discretize_three_bins_ten_values(self):
        """
        Test the discretize function on ten values with three bins
        """
        
        expected_result=[0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
        result=stats.discretize([0, 0, 0, 0, 1, 2, 2, 2, 2, 3], 3)
        self.assertEqual(expected_result,result)
        
    def test_discretize_nine_values(self):
        """
        Test the discretize function on nine values which are mostly zero
        """
        
        expected_result=[1, 0, 0, 0, 0, 0, 0, 0, 0]
        result=stats.discretize([0.1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(expected_result,result)
        
    def test_discretize_fifty_one_values(self):
        """
        Test the discretize function on a large set of values
        """
        
        expected_result=[3, 6, 6, 5, 5, 0, 2, 2, 3, 5, 
            2, 4, 4, 2, 3, 5, 0, 4, 0, 6, 
            0, 1, 6, 1, 5, 3, 0, 3, 2, 1, 
            3, 0, 6, 3, 2, 0, 6, 5, 1, 3, 
            6, 4, 1, 1, 4, 5, 0, 4, 2, 4, 1]
        input_values=[0.992299, 1, 1, 0.999696, 0.999605, 0.663081, 0.978293, 
            0.987621, 0.997237, 0.999915, 0.984792, 0.998338, 0.999207, 0.98051, 
            0.997984, 0.999219, 0.579824, 0.998983, 0.720498, 1, 0.803619, 
            0.970992, 1, 0.952881, 0.999866, 0.997153, 0.014053, 0.998049, 
            0.977727, 0.971233, 0.995309, 0.0010376, 1, 0.989373, 0.989161, 
            0.91637, 1, 0.99977, 0.960816, 0.998025, 1, 0.998852, 0.960849, 
            0.957963, 0.998733, 0.999426, 0.876182, 0.998509, 0.988527, 
            0.998265, 0.943673]
        result=stats.discretize(input_values)
        self.assertEqual(expected_result,result)
        
    def test_discretize_array_skip_one(self):
        """
        Test the discretize function with a numpy array and one skip
        """
        

        expected_result=array([
            [ 1.,  1.,  0.,  0.],
            [ 1.,  1.,  0.,  0.],
            [ 0.,  0.,  1.,  1.],
            [ 0.,  0.,  1.,  1.]])
        y = array([[-0.1,-0.2,-0.3,-0.4],[1,1,0,0],[0.25,0.5,0.75,1.0],
            [0.015625,0.125,0.421875,1.0]])
        result=stats.discretize(y, aiSkip = [1])
        self.assertEqual(expected_result.all(),result.all())
        
    def test_discretize_array_skip_two(self):
        """
        Test the discretize function with a numpy array and two skips
        """
        
        expected_result=array([
            [ 0.,  0.,  1.,  1.],
            [ 1.,  1.,  1.,  0.],
            [ 0.,  0.,  1.,  1.],
            [ 0.,  0.,  0.,  1.]])
        x = array([[0.1,0.2,0.3,0.4],[1,1,1,0],[0.01,0.04,0.09,0.16],[0,0,0,1]])
        result=stats.discretize(x, aiSkip = [1,3])
        self.assertEqual(expected_result.all(),result.all())
        
