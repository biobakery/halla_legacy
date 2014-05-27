"""
#a1, a2 = numpy.array( [1,2,3,4,5] ), numpy.array( [6,7,8,9,10] )

a1,a2 = map( numpy.array, ([random.randint(0,4) for i in range(10)] for j in range(2)) )

CE = CEuclideanDistance( a1, a2 )
print CE.get_distance()
#print CE.get_distance_type()
#print CE.get_data_type()

CMI = CMutualInformation( a1, a2 )
print CMI.get_distance()
#print CMI.get_distance_type()
#print CMI.get_data_type()


CNMI = CNormalizedMutualInformation( a1, a2 )
print CNMI.get_distance()
#print CNMI.get_distance_type()
#print CNMI.get_data_type()

CAMI = CAdjustedMutualInformation( a1, a2 )
print CAMI.get_distance() 

print "you can use it like so: ", CNormalizedMutualInformation( a1, a2 ).get_distance()

print "or like this:"
normalized_mi = lambda x,y : CNormalizedMutualInformation(x,y).get_distance()

print normalized_mi( a1, a2 )

print "inverted distance:" , CNormalizedMutualInformation( a1, a2 ).get_inverted_distance( "1mflip" )
""" 

import csv
import sys

import numpy
from numpy.random import normal

from test import *


pVec = numpy.array( uniformly_spaced_gaussian( 32 ) )
pOut = numpy.vstack( [[pVec * normal()] for _ in range(100)] ).T 

csvw = csv.writer(sys.stdout, csv.excel_tab)

for item in pOut:
	csvw.writerow( item )
