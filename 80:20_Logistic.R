library(ggplot2)
library(pROC)

setwd("S:/Advance data Mining/Project")


data <- read.csv("Bank_Personal_Loan_Modelling.csv")
head(data)


missing_counts <- colSums(is.na(data))
print(missing_counts)

#there is null value.

set.seed(530)
Random.seed <- c('Mersenne-Twister', 530)



data$Education <- factor(data$Education)
data$Family <- factor(data$Family)





ggplot(data) + geom_bar(aes(x = factor(Personal.Loan)), fill='lightblue', color='black', stat='count') + geom_text(stat='count',aes(x = factor(Personal.Loan), label=..count..), vjust=2) + xlab('Personal Loan') + ylab('Count')

# Plot the education counts
ggplot(data) + geom_bar(aes(x = Education), fill='lightblue', color='black', stat='count') +
  geom_text(stat='count',aes(x = Education, label=..count..), vjust=5) +
  xlab('Education') + ylab('Count') + theme()



#distributing the family


ggplot(data) + geom_bar(aes(x = Family), fill='lightblue', color='black', stat='count') +
  geom_text(stat='count',aes(x = Family, label=..count..), vjust=5) +
  xlab('Family') + ylab('Count') + theme()


# Plot histograms and scatter plots for Age, Experience, Income, CCAvg, Mortgage
ggplot(data) + geom_histogram(aes(x=Age, y=..density..), color="black",fill="lightblue", binwidth=1) + geom_density(aes(x=Age), color="blue")


ggplot(data) + geom_histogram(aes(x=Experience, y=..density..), color="black",fill="lightblue", binwidth=1) + geom_density(aes(x=Experience), color="blue")
ggplot(data) + geom_histogram(aes(x=Income, y=..density..), color="black",fill="lightblue", binwidth=5) + geom_density(aes(x=Income), color="blue")
ggplot(data) + geom_histogram(aes(x=CCAvg, y=..density..), color="black",fill="lightblue", binwidth=0.1) + geom_density(aes(x=CCAvg), color="blue")
ggplot(data) + geom_histogram(aes(x=Mortgage, y=..density..), color="black",fill="lightblue", binwidth=100) + geom_density(aes(x=Mortgage), color="blue")

#######################################################################
#(80/20)

indices <-sample(1:nrow(data), 0.8 * nrow(data), replace = FALSE)
train <-data[indices,]
test <-data[-indices,]



logit.train <- glm(Personal.Loan ~ Age + Experience + Income + Family + CCAvg + Education + Mortgage + Securities.Account + CD.Account + Online + CreditCard, data = train, family = "binomial")

summary(logit.train)


#from this we can see that in 80% distribution we need to include Intercept, Income, Family3, Family4, CCAvg, Eductaion2, Education 3, Securities.Account, CD.Account, Online, CreditCard because there P value is less then 0.05
#1.	A one-unit increase in Income is associated with an increase in the log-odds of Personal.Loan by 6.229e-02
#2.	A one-unit increase in Family3 is associated with an increase in the log-odds of Personal.Loan by 1.959e+00
#3.	A one-unit increase in Family4 is associated with an increase in the log-odds of Personal.Loan by 6.229e-02
#4. A one-unit increase in CCAvg is associated with an increase in the log-odds of Personal.Loan by 1.965e-01
#5.	A one-unit increase in Education2 is associated with an increase in the log-odds of Personal.Loan by 3.913e+00
#6.	A one-unit increase in Education3 is associated with an increase in the log-odds of Personal.Loan by 4.079e+00
#7.	A one-unit increase in Securities.Account is associated with an decrease in the log-odds of Personal.Loan by -7.667e-01
#8.	A one-unit increase in CD.Account is associated with an increase in the log-odds of Personal.Loan by 3.632e+00
#9.	A one-unit increase in Online is associated with an decraese in the log-odds of Personal.Loan by -6.926e-01
#10.A one-unit increase in CreditCard is associated with an decraese in the log-odds of Personal.Loan by -9.406e-01


# Calculate loglikelihood this will tell if the model with 80% train is better or 50% train is better
#the lower the liklyhood the better the result is.
logLik(logit.train)
# 'log Lik.' -451.3085 (df=15)

#test the significance of the model
with(logit.train, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))


#intercept model
logit.none <- glm(Personal.Loan ~ 1, data = data, family = "binomial")
with(logit.none, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))
summary(logit.none)

#this shows that 80% model works best and we dont need to consider intercept model.
#                             Intercept only	80% model
#AIC (the smaller the better)	3164	          932.62
#p-value	                    1	              0




#Testing set
predictions <- predict(logit.train, test, type="response")
# Discretize predictions; set probabilities >0.5 to 1 and <=0.5 to 0 
predictions.binary <- ifelse(predictions > 0.5, 1, 0)

# Create confusion table
cftable <- table(predictions.binary, test$Personal.Loan)
cftable

# Calculate Accuracy
accuracy <- sum(diag(cftable))/sum(cftable)
accuracy
# Calculate Sensitivity
sensitivity<-cftable[1]/(cftable[1] + cftable[2])
sensitivity
# Calculate Specificity
specificity <- cftable[4]/(cftable[3] + cftable[4])
specificity

# Calculate Positive Predictive Value
ppv <- cftable[1]/(cftable[1] + cftable[3])
ppv

# Calculate Negative Predictive Value
npv <- cftable[4]/(cftable[2] + cftable[4])
npv


#Training set
pred.train <- predict(logit.train, train, type="response")
pred.train.binary <- ifelse(pred.train > 0.5, 1, 0)

# Create confusion table
cft.train <- table(pred.train.binary, train$Personal.Loan)
cft.train

# Calculate Accuracy
accuracy <- sum(diag(cft.train))/sum(cft.train)
accuracy
# Calculate Sensitivity
sensitivity <- cft.train[1]/(cft.train[1] + cft.train[2])
sensitivity
# Calculate Specificity
specificity <- cft.train[4]/(cft.train[3] + cft.train[4])
specificity
# Calculate Positive Predictive Value
ppv <- cft.train[1]/(cft.train[1] + cft.train[3])
ppv
# Calculate Negative Predictive Value
npv <- cft.train[4]/(cft.train[2] + cft.train[4])
npv



#S-curve
#income
Income_mdl <- glm(Personal.Loan ~ Income, data = data, family = "binomial")
plot(Personal.Loan ~ Income, data = data, 
     col = "darkorange", pch = "|", xlim = c(-25, 250), ylim = c(0, 1),
     main = "Using Logistic Regression for Classification")

abline(h = 0, lty = 3)
abline(h = 1, lty = 3)
abline(h = 0.5, lty = 2)
curve(predict(Income_mdl, data.frame(Income = x), type = "response"), 
      add = TRUE, lwd = 3, col = "dodgerblue")
abline(v = -coef(Income_mdl)[1] / coef(Income_mdl)[2], lwd = 3)



#CCAvg
CCAvg_mdl <- glm(Personal.Loan ~ CCAvg, data = data, family = "binomial")
plot(Personal.Loan ~ CCAvg, data = data, 
     col = "darkorange", pch = "|", xlim = c(-10, 20), ylim = c(0, 1),
     main = "Using Logistic Regression for Classification")

abline(h = 0, lty = 3)
abline(h = 1, lty = 3)
abline(h = 0.5, lty = 2)
curve(predict(CCAvg_mdl, data.frame(CCAvg = x), type = "response"), 
      add = TRUE, lwd = 3, col = "dodgerblue")
abline(v = -coef(CCAvg_mdl)[1] / coef(CCAvg_mdl)[2], lwd = 3)

#Securities.Account
Securities.Account_mdl <- glm(Personal.Loan ~ Securities.Account, data = data, family = "binomial")
plot(Personal.Loan ~ Securities.Account, data = data, 
     col = "darkorange", pch = "|", xlim = c(-10, 30), ylim = c(0, 1),
     main = "Using Logistic Regression for Classification")

abline(h = 0, lty = 3)
abline(h = 1, lty = 3)
abline(h = 0.5, lty = 2)
curve(predict(Securities.Account_mdl, data.frame(Securities.Account = x), type = "response"), 
      add = TRUE, lwd = 3, col = "dodgerblue")
abline(v = -coef(Securities.Account_mdl)[1] / coef(Securities.Account_mdl)[2], lwd = 3)

#ROC AND AUC curve
#train
train.prob = predict(logit.train, newdata = train, type = "response")
roc(train$Personal.Loan, train.prob)

#test
test.prob = predict(logit.train, newdata = test, type = "response")

test.roc = roc(test$Personal.Loan, test.prob)
plot.roc(test.roc, col=par("fg"),plot=TRUE,print.auc = FALSE, legacy.axes = TRUE, asp =NA)
test.roc

plot.roc(smooth(test.roc), col="blue", add = TRUE,plot=TRUE,print.auc = TRUE, legacy.axes = TRUE, asp =NA)
legend("bottomright", legend=c("Empirical", "Smoothed"),
       col=c(par("fg"), "blue"), lwd=2)



