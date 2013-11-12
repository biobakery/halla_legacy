
### Baseline tests for HALLA implemented in R 
## Use cor, dcor, mutual information with FDR correction 
## Relevant metric: time, power 

## stats/method cor spearman  mi  mic dcor  
## power
## time  

require("entropy")
require("energy")
require("combinat")

### Set up test data 

generate.test <- function(number.instance, suf.stat1=0, suf.stat2=1, noise.ratio=0.1){
 N <- number.instance 
 theta1 <- suf.stat1 
 theta2 <- suf.stat2 

 vec.noise <- function(float.noise){ rnorm( N, 0,1 )*float.noise}

 x0 <- rnorm( N, theta1, theta2 ) #without noise parameter  
 x1 <- (1+vec.noise(noise.ratio))*x0 #linear transformation with noise 
 x2 <- (1+vec.noise(noise.ratio))*x0^(1/3)
 x3 <- (1+vec.noise(noise.ratio))*x0^2 

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

get.ami <- function( matrix.x, matrix.y ){
 x <- matrix.x 
 y <- matrix.y 
 bin.x <- sqrt( length(x) )
 print("length of bin.x is")
 print(bin.x)
 bin.y <- sqrt( length(y) )
 print("length of bin.y is")
 print(bin.y)
 dxy <- discretize2d(x,y, bin.x, bin.y)  
 
 H.x <- entropy.empirical( discretize( x, bin.x ), unit="log2" )
 H.y <- entropy.empirical( discretize( y, bin.y ), unit="log2" )
 correction.factor <- sqrt( H.x * H.y )
 
 return( mi.empirical( dxy, unit="log2" )/correction.factor )

}

grouped.test <- function( matrix.in, group.size=2 ){
 # Output table 
 # method/pair.index (1,1) (1,2) ... 
 # cor
 # dcor 
 # mi-adjusted 
 # ... 
  
 M <- matrix.in 
 
 vec.dim <- dim( matrix.in )
 number.row <- vec.dim[1]
 number.col <- vec.dim[2]
 
 matrix.out <- NULL #initialize 
 vec.colnames <- NULL 
 vec.rownames <- c("pearson", "spearman", "kendall", "dcor", "ami")
 #vec.rownames <- c("pearson", "spearman", "kendall", "dcor")
 tmp <- NULL 
 
 print("the number of rows is")
 print(number.row)
 
 vec.comb <- combn(number.row, group.size)
 print(vec.comb)
 comb.size <- dim(vec.comb)[2]
 for(i in seq(comb.size)){
   tmp$pearson <- NULL
   tmp$spearman <- NULL 
   tmp$kendall <- NULL 
   tmp$dcor <- NULL
   tmp$ami <- NULL

  index.one <- vec.comb[,i][1]
  index.two <- vec.comb[,i][2]
  print("index one is")
  print(index.one)
  print("index two is")
  print(index.two)
  vec.x <- M[index.one,]
  vec.y <- M[index.two,]
  label.pair <- paste( "(",index.one,",",index.two,")" ,sep="") 
  tmp$pearson <- cor( vec.x, vec.y )
  tmp$spearman <- cor( vec.x, vec.y, method="spearman")
  tmp$kendall <- cor( vec.x, vec.y, method="kendall")
  tmp$dcor <- dcor(vec.x,vec.y)
  tmp$ami <- get.ami( vec.x,vec.y ) 
  
  vec.colnames <- c(vec.colnames, label.pair)
  matrix.out <- cbind( matrix.out, c(tmp$pearson,tmp$spearman,tmp$kendall,tmp$dcor,tmp$ami))
 }
  
  df.out <- data.frame( matrix.out, row.names=vec.rownames )
  #df.out <- data.frame( matrix.out )
  
  colnames(df.out) <- vec.colnames 
 
  return( df.out )
}

### Runtime 



