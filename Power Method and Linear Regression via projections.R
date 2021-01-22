# Extend the power method to provide the first k eigenvectors and eigenvalues.
# Initialize x1 in R^n and set x0 = x1 + 1
library(pracma)
n = 3

A = matrix(c(7, 4, 1,
             4, 4, 4,
             1, 4, 7), n, n, byrow = TRUE)
# Note: A must be symmetric
powerMethod <- function(A, x1 = rnorm(ncol(A)), tol=1e-6){
  x1 <- x1/Norm(x1, p = 2)
  x0 <- x1 + 1
  i <- 0 
  while((sum(abs(x1 - x0)) > tol) && (i < 1e3)){
    x0 <- x1
    x1 <- A%*%x1
    x1 <- x1/Norm(x1, p = 2) 
    i <- i+1
  }
  v <- x1
  lambda <- dot((A%*%x1), x1)/dot(x1, x1)
  return(list("v" = v, "lambda" = lambda))}

#powerMethod(A)

powerEigs <- function(A, n, tol=1e-6){
  pairs <- list()
  for (i in 1:n){
    key <- toString(i)
    out <- powerMethod(A, tol=tol)
    val <- round(out$lambda, 4)
    vec <- round(as.matrix(out$v), 4)
    pairs[[key]] <- out
    A <- A - (val*(vec%*%t(vec)/as.numeric(t(vec)%*%vec)))
  }
  return(pairs)
}

powerEigs(A, ncol(A), tol=1e-6)

# Determine the minimum number of variables required to predict the zip code 
# number to the greatest extent possible suing only a linear model.
zipcodes <- read.table("C:/zip.train.gz", sep=" ", header=F)
n <- length(zipcodes[,1])
# Split the data into 80% training and 20% testing data
n.train <- as.integer(n*0.80)
# Randomly select the rows of the training example
train.id <- sample(1:n, n.train)
# Split the data
train.data <- zipcodes[train.id,]
test.data <- zipcodes[-train.id,]
# Response vectors
y.train <- train.data[,1]
y.test <- test.data[,1]
# Remove column of NAs from the data matrix
x.train <- as.matrix(train.data[,2:257])
x.test <- as.matrix(test.data[,2:257])

x <- x.train
y <- y.train

center_scale <- function(x) {
  x <- scale(x, scale = FALSE)
}
# PCA is sensitive to data centering
# Center data
x <- center_scale(x)
# Obtain the svd
S <- svd(x)
# d is the diagonal matrix with non-negative singular
# values of x
d <- S$d
# find orthonormal set of eigenvectors
pval <- svd(t(x)%*%x)$v
# w is the matrix of principal components
u <- svd(t(x)%*%x)$u
w <- x%*%pval

e <- prcomp(x, center = T, scale = T)
head(e$rotation)
# column variances
cvars <- apply(w,2,var)
tot_var_exp <- sum(cvars)

# ratios of variance explained for each principal comp
ratio <- cvars/tot_var_exp

# cumulative sums of the variance explained ratios 
csums <- cumsum(ratio)

# get that plot
plot(1:length(csums), csums)

# get where the components start explaining >= 90% of variance
# That is we need just k variables to best predict zip codes
k <- length(csums) - length(csums[which(csums > .9)])

# Compute the PCR estimator gam
# gam is the vector of estimated regression coefficients 
# obtained by ordinary least squares regression of the 
# response vector on the data matrix 

wtw <- t(w)%*%w[,1:k]
wtw.inv <- solve(wtw)
gam <- wtw.inv%*%(t(w[,1:k])%*%y)
# The final PCR estimator of beta.hat based
# on the first k principal components
beta.hat <- t(gam)%*%pval
beta.hat.k <- pval[,1:k]%*%gam
# Test: (good)
dat <- data.frame(x)
check <- lm(y ~ ., data = dat)
coef <- check$coefficients

# feature vector (rotation vector)
fv <- t(pval)
# project test data onto pcs 
z <- x.test %*% fv 

# get first k 
z <- z[,1:k]

# predict 
y.hat.pca <- z%*%gam
# TODO: project back into the space of the original data
