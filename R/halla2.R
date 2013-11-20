
### Baseline tests for HALLA implemented in R 
## Use cor, dcor, mutual information with FDR correction 
## Relevant metric: time, power 

## stats/method cor spearman  mi  mic dcor  
## power
## time  

require("entropy")
require("energy")
require("combinat")
require("rJava")
require("ggplot2")
require("matrixStats")

## External Dependencies 
# load MINE 

#source("MINE.r")
# It's saved like this: "<specified.prefix>,allpairs,cv=0.0,B=n^0.6,Results.csv"

### Tibshirani wrapper for MINE (adapted from http://www-stat.stanford.edu/~tibs/reshef/script.R)

get.MINE <-function(x,y){
  xx=cbind(x,y)
  write("x,y",file="test.csv")
  write(t(xx),sep=",",file="test.csv",ncol=2,append=T)
  command <- 'java -jar ../java/MINE.jar "test.csv" -allPairs'
  system(command)
  res=scan("test.csv,allpairs,cv=0.0,B=n^0.6,Results.csv",what="",sep=",")
  val=as.numeric(res[11])
  return(val)
}

### Set up test data 

generate.test <- function(){
  
 table.predictor <- read.table("~/hg/halla/input/predictor.txt", header=FALSE, sep="\t")
 table.response <- read.table("~/hg/halla/input/response.txt", header=FALSE, sep="\t")
 
 x1 <- as.numeric( table.predictor[1,] ) 
 x2 <- as.numeric( table.predictor[2,] )
 x3 <- as.numeric( table.predictor[3,] )
 
 y1 <- as.numeric( table.response[1,] )
 y2 <- as.numeric( table.response[2,] )
 y3 <- as.numeric( table.response[3,] )
 
 
 return(rbind(x1,x2,x3,y1,y2,y3)) 
  
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
 x <- matrix( matrix.x )
 print("x is")
 print(x)
 y <- matrix( matrix.y )
 print("y is")
 print(y)
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
 vec.rownames <- c("pearson", "spearman", "kendall", "dcor", "ami", "mine")
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
  tmp$mine <- get.MINE( vec.x, vec.y)
  
  vec.colnames <- c(vec.colnames, label.pair)
  matrix.out <- cbind( matrix.out, c(tmp$pearson,tmp$spearman,tmp$kendall,tmp$dcor,tmp$ami, tmp$mine))
 }
  
  df.out <- data.frame( matrix.out, row.names=vec.rownames )
  #df.out <- data.frame( matrix.out )
  
  colnames(df.out) <- vec.colnames 
 
  return( df.out )
}

### Runtime 

M <- generate.test( )
df.M <- grouped.test( M )

### Set up 
#par(mfrow=c(2,1))
#layout(matrix(c(1,0,0,2),4,1, byrow=TRUE))

### 1. 
boxplot(t(abs(df.M)))

# Mean and Variance 

vec.mean <- rowMeans( df.M )
vec.var <- rowVars( df.M )

summary.table <- data.frame( rbind( vec.mean, vec.var ) )
rownames(summary.table) <- c("mean", "variance")

### 2. 

grid.table(t(summary.table))
#heatmap(summary.table)

plot.new()
frame() 

par(mfrow=c(3,2))
### 3. 
plot( M[1,], M[2,] )
### 4. 
plot( M[1,], M[3,] )
### 5. 
plot( M[1,], M[4,] )
### 6. 
plot( M[1,], M[5,] )
### 7. 
plot( M[1,], M[6,] )
### 8. 
#plot( M[1,], M[7,] )





