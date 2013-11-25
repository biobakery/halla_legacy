#!/usr/bin/env R 

sessionInfo()

inputargs <- commandArgs(TRUE) 


# N x D transposed matrix, with dummy headers "Var1, Var2, ...".
# Remember to make sure the categories are strings ("characters") 
# Read in as tab-delimited file. 

input.file <-  inputargs[1] 
output.file <- inputargs[2]

library( FactoMineR )

input.matrix <- read.csv( input.file, sep="\t" ) 

#the residues 
res.mca <- MCA( input.matrix ) 

output.matrix <- res.mca$var$eta2 

# Output is actually D x N -- no need to transpose 
write.csv( output.matrix, output.file, sep="\t" ) 
