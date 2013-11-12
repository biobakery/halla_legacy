
## We write the following function because MIC glitches a small percentage of the time, and we do not wish to average over those trials

notNA.greater <- function(a,b){
  ind <- which(!is.na(a))
  pow <- sum(a[ind] > b)/length(ind)
  return(pow)
}

# download from  mine.jar exploredata.net
#install.packages("energy")
library("energy")

## This is a short R wrapper which uses system calls to call MIC
# (we wrote our own wrapper, as the one provided by the authors was difficult to use)

mymine=function(x,y){
 xx=cbind(x,y)
 write("x,y",file="test.csv")
 write(t(xx),sep=",",file="test.csv",ncol=2,append=T)
 command <- 'java -jar MINE.jar "test.csv" -allPairs'
    system(command)
 res=scan("test.csv,B=n^0.6,k=15.0x,Results.csv",what="",sep=",")
 val=as.numeric(res[11])
return(val)
}

set.seed(1)

# Here we define parameters which we use to simulate the data

nsim=500                           # The number of null datasets we use to estimate our rejection reject regions for an alternative with level 0.05
nsim2=500                           # Number of alternative datasets we use to estimate our power

num.noise <- 30                     # The number of different noise levels used
noise <- 3                          # A constant to determine the amount of noise

n=320                               # Number of data points per simulation

val.cor=val.dcor=val.mine=rep(NA,nsim)              # Vectors holding the null "correlations" (for pearson, dcor and mic respectively) for each of the nsim null datasets at a given noise level

val.cor2=val.dcor2=val.mine2= rep(NA,nsim2)              # Vectors holding the alternative "correlations" (for pearson, dcor and mic respectively) for each of the nsim2 alternative datasets at a given noise level

power.cor=power.dcor=power.mine= array(NA, c(8,num.noise))                # Arrays holding the estimated power for each of the "correlation" types, for each data type (linear, parabolic, etc...) with each noise level

## We loop through the noise level and functional form; each time we estimate a null distribution based on the marginals of the data, and then use that null distribution to estimate power

## We use a uniformly distributed x, because in the original paper the authors used the same

for(l in 1:num.noise){
  for(typ in 1:8){

    ## This next loop simulates data under the null with the correct marginals (x is uniform, and y is a function of a uniform with gaussian noise)
    
    for(ii in 1:nsim){
      x=runif(n)
      
      if(typ==1){
        y=x+ noise *(l/num.noise)* rnorm(n)
      }
                                        #parabolic+noise
      if(typ==2){
        y=4*(x-.5)^2+  noise * (l/num.noise) * rnorm(n)
      }
                                        #cubic+noise
      if(typ==3){
        y=128*(x-1/3)^3-48*(x-1/3)^3-12*(x-1/3)+10* noise  * (l/num.noise) *rnorm(n)
      }
                                        #sin+noise
      if(typ==4){
        y=sin(4*pi*x) + 2*noise * (l/num.noise) *rnorm(n)
      }
                                              #their sine + noise
      if(typ==5){
        y=sin(16*pi*x) + noise * (l/num.noise) *rnorm(n)
      }
                                        #x^(1/4) + noise
      if(typ==6){
        y=x^(1/4) + noise * (l/num.noise) *rnorm(n)
      }
                                        #circle
      if(typ==7){
        y=(2*rbinom(n,1,0.5)-1) * (sqrt(1 - (2*x - 1)^2)) + noise/4*l/num.noise *rnorm(n)
      }
                                        #step function
      if(typ==8){
        y = (x > 0.5) + noise*5*l/num.noise *rnorm(n)
      }
      
      x <- runif(n)                       # We resimulate x so that we have the null scenario
      
      val.cor[ii]=(cor(x,y))^2            # Calculate the correlation
      val.dcor[ii]=dcor(x,y)              # Calculate dcor
      val.mine[ii]=mymine(x,y)            # Calculate mic
    }

    val.mine <- val.mine[which(!is.na(val.mine))]                 # we remove the mic trials which glitch

    ## Next we calculate our 3 rejection cutoffs
    
    cut.cor=quantile(val.cor,.95)
    cut.dcor=quantile(val.dcor,.95)
    cut.mine=quantile(val.mine,.95)

    ## Next we simulate the data again, this time under the alternative
    
    for(ii in 1:nsim2){
      x=runif(n)

                                        #lin+noise
      if(typ==1){
        y=x+ noise * (l/num.noise) *rnorm(n)
      }
                                        #parabolic+noise
      if(typ==2){
        y=4*(x-.5)^2+  noise * (l/num.noise)*rnorm(n)
      }
                                        #cubic+noise
      if(typ==3){
        y=128*(x-1/3)^3-48*(x-1/3)^3-12*(x-1/3)+10* noise * (l/num.noise) *rnorm(n)
      }
                                        #sin+noise
      if(typ==4){
        y=sin(4*pi*x) + 2*noise * (l/num.noise) *rnorm(n)
      }
                                        #their sine + noise
      if(typ==5){
        y=sin(16*pi*x) + noise * (l/num.noise) *rnorm(n)
      }
                                        #x^(1/4) + noise
      if(typ==6){
        y=x^(1/4) + noise * (l/num.noise) *rnorm(n)
      }
                                        #circle
      if(typ==7){
        y=(2*rbinom(n,1,0.5)-1) * (sqrt(1 - (2*x - 1)^2)) + noise/4*l/num.noise *rnorm(n)
      }
                                        #step function
      if(typ==8){
        y = (x > 0.5) + noise*5*l/num.noise *rnorm(n)
      }

      ## We again calculate our "correlations"
      
      val.cor2[ii]=(cor(x,y))^2
      val.dcor2[ii]=dcor(x,y)
      val.mine2[ii]=mymine(x,y)
      
    }

    ## Now we estimate the power as the number of alternative statistics exceeding our estimated cutoffs
    
    power.cor[typ,l] <- sum(val.cor2 > cut.cor)/nsim2
    power.dcor[typ,l] <- sum(val.dcor2 > cut.dcor)/nsim2
    power.mine[typ,l] <- notNA.greater(val.mine2, cut.mine)
  }
}

save.image()

## The rest of the code is for plotting the image

pdf("power.pdf")
par(mfrow = c(4,2), cex = 0.45)
plot((1:30)/10, power.cor[1,], ylim = c(0,1), main = "Linear", xlab = "Noise Level", ylab = "Power", pch = 1, col = "black", type = 'b')
points((1:30)/10, power.dcor[1,], pch = 2, col = "green", type = 'b')
points((1:30)/10, power.mine[1,], pch = 3, col = "red", type = 'b')
legend("topright",c("cor","dcor","MIC"), pch = c(1,2,3), col = c("black","green","red"))

plot((1:30)/10, power.cor[2,], ylim = c(0,1), main = "Quadratic", xlab = "Noise Level", ylab = "Power", pch = 1, col = "black", type = 'b')
points((1:30)/10, power.dcor[2,], pch = 2, col = "green", type = 'b')
points((1:30)/10, power.mine[2,], pch = 3, col = "red", type = 'b')
legend("topright",c("cor","dcor","MIC"), pch = c(1,2,3), col = c("black","green","red"))

plot((1:30)/10, power.cor[3,], ylim = c(0,1), main = "Cubic", xlab = "Noise Level", ylab = "Power", pch = 1, col = "black", type = 'b')
points((1:30)/10, power.dcor[3,], pch = 2, col = "green", type = 'b')
points((1:30)/10, power.mine[3,], pch = 3, col = "red", type = 'b')
legend("topright",c("cor","dcor","MIC"), pch = c(1,2,3), col = c("black","green","red"))

plot((1:30)/10, power.cor[5,], ylim = c(0,1), main = "Sine: period 1/8", xlab = "Noise Level", ylab = "Power", pch = 1, col = "black", type = 'b')
points((1:30)/10, power.dcor[5,], pch = 2, col = "green", type = 'b')
points((1:30)/10, power.mine[5,], pch = 3, col = "red", type = 'b')
legend("topright",c("cor","dcor","MIC"), pch = c(1,2,3), col = c("black","green","red"))

plot((1:30)/10, power.cor[4,], ylim = c(0,1), main = "Sine: period 1/2", xlab = "Noise Level", ylab = "Power", pch = 1, col = "black", type = 'b')
points((1:30)/10, power.dcor[4,], pch = 2, col = "green", type = 'b')
points((1:30)/10, power.mine[4,], pch = 3, col = "red", type = 'b')
legend("topright",c("cor","dcor","MIC"), pch = c(1,2,3), col = c("black","green","red"))

plot((1:30)/10, power.cor[6,], ylim = c(0,1), main = "X^(1/4)", xlab = "Noise Level", ylab = "Power", pch = 1, col = "black", type = 'b')
points((1:30)/10, power.dcor[6,], pch = 2, col = "green", type = 'b')
points((1:30)/10, power.mine[6,], pch = 3, col = "red", type = 'b')
legend("topright",c("cor","dcor","MIC"), pch = c(1,2,3), col = c("black","green","red"))

plot((1:30)/10, power.cor[7,], ylim = c(0,1), main = "Circle", xlab = "Noise Level", ylab = "Power", pch = 1, col = "black", type = 'b')
points((1:30)/10, power.dcor[7,], pch = 2, col = "green", type = 'b')
points((1:30)/10, power.mine[7,], pch = 3, col = "red", type = 'b')
legend("topright",c("cor","dcor","MIC"), pch = c(1,2,3), col = c("black","green","red"))

plot((1:30)/10, power.cor[8,], ylim = c(0,1), main = "Step function", xlab = "Noise Level", ylab = "Power", pch = 1, col = "black", type = 'b')
points((1:30)/10, power.dcor[8,], pch = 2, col = "green", type = 'b')
points((1:30)/10, power.mine[8,], pch = 3, col = "red", type = 'b')
legend("topright",c("cor","dcor","MIC"), pch = c(1,2,3), col = c("black","green","red"))
dev.off()
