

### Baseline tests for HALLA implemented in R 

require("entropy")
require("energy")
require("combinat")

### Set up test data 

generate.test <- function(number.instance, suf.stat1=0, suf.stat2=1, noise.ratio=0.1){
 N <- number.instance 
 theta1 <- suf.stat1 
 theta2 <- suf.stat2 

 vec.noise <- rnorm( N, 0,1 )*noise.ratio  

 x0 <- rnorm( N, theta1, theta2 ) #without noise parameter  
 x1 <- (1+ vec.noise)*x0 #linear transformation with noise 
 x2 <- (1+vec.noise)*x0^2
 x3 <- (1+vec.noise)*x0^3 

 return(rbind(x0,x1,x2,x3)) 
}

meta.generate.test <- function(number.copies, number.instance, suf.stat1=0, suf.stat2=1, noise.ratio=0.1){
 
 P <- number.copies
 N <- number.instance 
 
 stopifnot( number.copies > 0 && number.instance > 0 )

 matrix.out <- generate.test( N, suf.stat1, suf.stat2, noise.ratio )  
 P <- P-1  

 while(P > 0){
  matrix.out <- rbind( matrix.out, generate.test( N, suf.stat1, suf.stat2, noise.ratio ) )
  P <- P-1 
 }

 return(matrix.out)

}

pairwise.test <- function( matrix.in, group.size=2 ){
 vec.dim <- dim( matrix.in )
 number.row <- vec.dim[1]
 number.col <- vec.dim[2]
 
 vec.comb <- combn(number.row, group.size)
 comb.size <- dim(vec.comb)[2]
 for(i in seq(comb.size)){
  pOne, pTwo <- vec.comb[,i]
 }

}

