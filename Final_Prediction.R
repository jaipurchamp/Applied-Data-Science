source("imputation_2.0.1.tar.gz")
install.packages("imputation",source = imputation_2.0.1.tar.gz)
# Installing imputation package which basically fills missing values using Singular Value Thresholding,kNN,tree,mean, SVD or linear imputation in a data frame
install.packages("TimeProjection","locfit","gbm")
install.packages("imputation_2.0.1.tar.gz",repos=NULL,type='source')
install.packages("ROCR")
# Installing packages "TimeProjection","locfit" and "gbm" which will be used in below process
# TimeProjection is useful in extracting time components of a data object like day, week, holiday and put that in data frame.
# Locfit basically represents local regression, likelihood and density estimation.
# gbm includes regression models for least squares, absolute loss, logistic, Poisson, etc.
library(e1071)
library(rpart)
library(ada)
library(adabag)
library(ggplot2)
library(randomForest)
library(imputation)
library(GA)
library(ROCR)
library(caTools)
# Initializing all the packages installed above 

bank_train<-read.csv("cs-training.csv")
bank_test <- read.csv("cs-test.csv")
# Loading Training and Testing Datasets

plot.bank_data <- function(bank_data) { plot1 <- ggplot(bank_data) + 
    geom_bar(aes(x = factor(SeriousDlqin2yrs), y = ..count../sum(..count..))) + 
    ylab("Percentage") + xlab("Classification of Data")
  plot1
}
# This is a function used to plot-data  

bank_data_split <-function(bank_data, bank_data_size) {
  no_of_rows <- nrow(bank_data)
  probability <- ifelse(bank_data$SeriousDlqin2yrs == 1, 0.20, 0.015)
  sample_probability <- bank_data[sample(no_of_rows, no_of_rows / bank_data_size, prob = probability, replace = TRUE), ] 
  return(sample_probability)
}

# This is a function used to split data , generate samples of data for random-forest algorithm

table(bank_train$age)
table(bank_train$NumberOfTime30.59DaysPastDueNotWorse)
table(bank_train$NumberOfTimes90DaysLate)
table(bank_train$NumberOfTime60.89DaysPastDueNotWorse)


#Checking the table and finding out the trends.

boxplot(bank_train$RevolvingUtilizationOfUnsecuredLines, horizontal = T)
boxplot(bank_train$NumberOfTime30.59DaysPastDueNotWorse, horizontal = T, main = "Num. Times 30-59 Days Late")
# boxplot of each variable. Note that 98/96 are the outliers for the categories
# mentioned above.
apply(bank_train, 2, boxplot)

# We should think about ways to weight observations where the person did 
# default since defaulting obs. are underrepresented in the data.
plot.bank_data(bank_train)

# We should impute all the data before we begin training/testing with it. 
# Here is a naive attempt that removes the offending observations from our data.
# We use a boosted regression tree imputation method
combine_observations <- rbind(bank_train, bank_test)[ , -1]  # Gets rid of the "X" column

# filter nonsense variables (Data- Cleaning)
combine_observations$age <- ifelse(combine_observations$age > 0, combine_observations$age, NA)

combine_observations$RevolvingUtilizationOfUnsecuredLines <- ifelse(combine_observations$RevolvingUtilizationOfUnsecuredLines <= 1,
                                                                    combine_observations$RevolvingUtilizationOfUnsecuredLines, NA)

combine_observations$NumberOfTime30.59DaysPastDueNotWorse <- ifelse(combine_observations$NumberOfTime30.59DaysPastDueNotWorse < 90,
                                                                    combine_observations$NumberOfTime30.59DaysPastDueNotWorse, NA)


combine_observations$NumberOfTimes90DaysLate <- ifelse(combine_observations$NumberOfTimes90DaysLate < 90, 
                                                       combine_observations$NumberOfTimes90DaysLate, NA)

combine_observations$NumberOfTime60.89DaysPastDueNotWorse <- ifelse(combine_observations$NumberOfTime60.89DaysPastDueNotWorse < 90,
                                                                    combine_observations$NumberOfTime60.89DaysPastDueNotWorse, NA)

combine_observations$DebtRatio <- ifelse(is.na(combine_observations$MonthlyIncome), NA, combine_observations$DebtRatio)

combine_observations$NumberOfDependents[is.na(bank_train$NumberOfDependents)] <- 0

summary(combine_observations)
# impute data using test and training data; ignore response
clean_impute_data <- gbmImpute(combine_observations[, -1], cv.fold = 5, max.iters = 3)

bank_train_full <- cbind(SeriousDlqin2yrs = bank_train$SeriousDlqin2yrs, # add reponse back to dataframe
                         clean_impute_data$x[1:nrow(bank_train), ]) 
bank_test_full <- cbind(SeriousDlqin2yrs = bank_test$SeriousDlqin2yrs,
                        clean_impute_data$x[c(-1:-nrow(bank_train)), ])

# If training takes too long, split training data up:
bank_data_unskewed <- bank_data_split(bank_train, 1)
head(bank_data_unskewed)
# let's check how defaults are distributed in these:
plot.bank_data(bank_data_unskewed)
plot.bank_data(combine_observations[0:nrow(bank_train), ])

# TODO: Someone should tweak the settings of this random forest to see if
# it gives us any better predictions. May also consider how we weight predictions

#Trying Linear Regression
lin_bank <- lm(SeriousDlqin2yrs ~ .,data = bank_train_full)
summary(lin_bank)

lin_bank <- lm(SeriousDlqin2yrs ~ RevolvingUtilizationOfUnsecuredLines + age + NumberOfTime30.59DaysPastDueNotWorse + MonthlyIncome + NumberOfOpenCreditLinesAndLoans + NumberRealEstateLoansOrLines + NumberOfTime60.89DaysPastDueNotWorse, data = bank_train_full)



# this gives us and what Randomforest gives us.
rf_bank <- randomForest(as.factor(SeriousDlqin2yrs) ~ ., data = bank_train_full)
#rf_bank <- randomForest(as.factor(SeriousDlqin2yrs) ~ ., data = bank_train_full,do.trace=TRUE,importance=TRUE,ntree=250,forest=TRUE)
#rf_bank <- randomForest(as.factor(SeriousDlqin2yrs) ~ ., data = bank_train_full,do.trace=TRUE,importance=TRUE,ntree=500,forest=TRUE)
#rf_bank <- randomForest(as.factor(SeriousDlqin2yrs) ~ ., data = bank_train_full,do.trace=TRUE,importance=TRUE,ntree=1000,forest=TRUE)
#rf_bank <- randomForest(as.factor(SeriousDlqin2yrs) ~ ., data = bank_train_full,do.trace=TRUE,importance=TRUE,ntree=500,forest=TRUE)

# Naive Bayes
nb_bank <- naiveBayes(as.factor(SeriousDlqin2yrs) ~ ., data = bank_data_unskewed)

predict_forest_bank_train <- predict(rf_bank, bank_train_full, type = "prob" )
predict_bayes_bank_train <- predict(nb_bank, bank_train_full, type = "raw")

predict_all_bank_train<- data.frame(predict_bayes_bank_train[, 2], predict_forest_bank_train[ ,2])
head(predict_all_bank_test)


# The fitness function for the genetic algorithm
# calculates the AUC 
# w1,w2,w3 are weights
bank_fit <- function(z, bank_data_guess) {
  v <- z / sum(z)
  predict_fit <- prediction(as.matrix(bank_data_guess) %*% v , bank_train_full[, 1])
  auc_bank <- performance(predict_fit, "auc")
  auc_bank <- unlist(slot(auc_bank, "y.values"))
  return (1 - auc_bank)
}

# The genetic algorithm finds the optimal weights
# The range of w1,w2 is [0,0.5]
# There are quite a few parameters we can play with
Genetic_algorithm_bank <- ga(type = "real-valued",fitness = bank_fit, maxiter = 10, bank_data_guess = predict_all_bank_train,
         min = c(0, 0), max = c(1, 1))

summary(Genetic_algorithm_bank)
GAweights <- c(0.156985,0.5970209) # Complete
#GAweights <- c(0.2300269,0.8750585) #250 -Trees
#GAweights <- c(0.1886652,0.7173049) #500
#GAweights <- c(0.1399561,0.5324322)  #1000

GAweights

predict_forest_bank_test <- predict(rf_bank, bank_test_full, type = "prob" )
predict_bayes_bank_test <- predict(nb_bank, bank_test_full, type = "raw")
predict_all_bank_test<- data.frame(predict_bayes_bank_test[, 2],predict_forest_bank_test[ , 2])

head(predict_all_bank_test)

predict_final_bank_test <- as.matrix(predict_all_bank_test) %*% GAweights

results_bank <- data.frame(Id = 1:nrow(bank_test), Probability = predict_final_bank_test)
write.table(results_bank, "imputed_final.csv", quote=F, row.names=F, sep=",")
