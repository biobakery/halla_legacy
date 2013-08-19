import hierarchy as h 
import copy

data1, data2 = [1,2,3], [4,5,6]

tree1,tree2 = [h.CHallaTree() for i in range(2)]

tree1.add_data(data1); tree2.add_data(data2)

tree1.add_child(tree2) 

next(tree1)
next(tree1)
