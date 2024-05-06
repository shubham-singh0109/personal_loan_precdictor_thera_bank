
data <- read.csv('Bank_Personal_Loan_Modelling.csv')
# Load required packages
install.packages("rpart")
install.packages("rpart.plot")
install.packages("caret")
install.packages("pROC")
library(rpart)
library(rpart.plot)
library(caret)
library(pROC)



head(data)

# Using summary() function
summary(data)




# Set the seed for reproducibility
set.seed(530)
Random.seed <- c("Mersenne-Twister", 530)


# Perform train-test split
data.split <- sample(1:nrow(data), size = nrow(data) * 0.2)
train.data <- data[data.split, ]
test.data <- data[-data.split, ]


# Constructing CART Model
formula <- Personal.Loan ~ Age + Experience + Income + ZIP.Code + Family + CCAvg + Education + Mortgage + Securities.Account + CD.Account + Online + CreditCard

# Fit CART model
cart_model <- rpart(formula = formula,
                    data = train.data,
                    method = "class",
                    control = rpart.control(minbucket = 2, xval = 10))


# Output graphical representation of the tree
prp(cart_model, roundint = FALSE)

# View the performance at each level of the tree
cart_model$cptable





cp.param <- cart_model$cptable

# Initialize vectors to store MSE values
train_err <- double(4)
cv_err <- double(4)
test_err <- double(4)



# Loop through each row of the complexity parameter table
for (i in 1:nrow(cp.param)) {
  alpha <- cp.param[i, 'CP']  # Extract the complexity parameter

  # Train the pruned decision tree model using the current complexity parameter
  pruned_cart_model <- prune(cart_model, cp = alpha)

  # Calculate classification error (misclassification rate) for training data
  train_cm <- table(train.data$Personal.Loan, predict(pruned_cart_model, newdata = train.data, type = 'class'))
  train_err[i] <- 1 - sum(diag(train_cm)) / sum(train_cm)

  cv_err[i] <- cp.param[i, 'xerror'] * cp.param[i, 'rel error']

  # Calculate classification error (misclassification rate) for testing data
  test_cm <- table(test.data$Personal.Loan, predict(pruned_cart_model, newdata = test.data, type = 'class'))
  test_err[i] <- 1 - sum(diag(test_cm)) / sum(test_cm)
  }

# Print classification error (1 – accuracy) values
train_err
test_err





# Plot training, CV and testing errors at # of Splits/depth

matplot(cp.param[,'nsplit'], cbind(train_err, cv_err, test_err), pch=19, col=c("red", "black", "blue"), type="b", ylab="Loss/error", xlab="Depth/# of Splits")
legend("right", c('Train', 'CV', 'Test') ,col=seq_len(3),cex=0.8,fill=c("red", "black", "blue"))



plotcp(cart_model)


# Calculate predictions using the pruned decision tree model on the testing dataset
predictions <- predict(cart_model, type = 'class', newdata = test.data)

# Create a confusion matrix
conf.mat.tree <- table(test.data$Personal.Loan, predictions)

# Print the confusion matrix
conf.mat.tree


# Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
tp <- conf.mat.tree[2, 2] # Predicted positive and actually positive
tn <- conf.mat.tree[1, 1] # Predicted negative and actually negative
fp <- conf.mat.tree[1, 2] # Predicted positive but actually negative
fn <- conf.mat.tree[2, 1] # Predicted negative but actually positive

# Calculate Sensitivity (True Positive Rate)
sensitivity <- tp / (tp + fn)

# Calculate Specificity (True Negative Rate)
specificity <- tn / (tn + fp)

# Print TP, TN, FP, FN, Sensitivity, and Specificity
print(paste("True Positives (TP):", tp))
print(paste("True Negatives (TN):", tn))
print(paste("False Positives (FP):", fp))
print(paste("False Negatives (FN):", fn))
print(paste("Sensitivity (True Positive Rate):", sensitivity))
print(paste("Specificity (True Negative Rate):", specificity))



acc <- sum(diag(conf.mat.tree))/sum(conf.mat.tree)
acc




# Print pruned and unpruned train, test, and all accuracy
# Error at the maximum splits
train.acc <- 1 - train_err[4] # Unpruned train accuracy
train.acc

test.acc <- 1 - test_err[4] # Unpruned test accuracy
test.acc

cv.acc <- 1 - cv_err[4] # Unpruned all accuracy
cv.acc



