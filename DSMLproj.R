setwd("C:/Users/julia/OneDrive/Escritorio")

library(caret)
library(randomForest)
library(lubridate)
library(ggplot2)
library(gridExtra)

training <- read.csv('pml-training.csv')
testing <- read.csv('pml-testing.csv')

# Since 19216 represents approximately 98% of the observations
# variables with that amount of NAs are excluded
training2 <- within(training, {
    classe <- as.factor(classe)
    new_window <- as.factor(new_window)
    cvtd_timestamp <- dmy_hm(cvtd_timestamp)
})

testing2 <- within(testing, {
    new_window <- as.factor(new_window)
    cvtd_timestamp <- dmy_hm(cvtd_timestamp)
})

# The X and user_name variables are also removed since
# they just serve as a subject identifier
training2 <- subset(training2, select=-c(X, user_name))
testing2 <- subset(testing2, select=-c(X, user_name, problem_id))

to_numeric <- function(df){
    for(col in names(df)){
        col_type <- class(df[,c(col)])[1]
        # If character convert to numeric because all character tables
        # in the table contain character values
        if(col_type == "character" | col_type == "logical"){
            df[,c(col)] <- as.numeric(df[,c(col)])
        }
    }

    df
}

# Convert character to numeric
training2 <- to_numeric(training2)
testing2 <- to_numeric(testing2)

clean_dat <- function(df){
    exclude_cols <- character()
    total_rows <- nrow(df)

    for(col in names(df)){

        # Count # of NAs for a particular col
        total_nas <- sum(is.na(df[,c(col)]))
        na_proportion <- total_nas / total_rows

        # Check if it reaches the threshold
        if (total_nas > .97){
            exclude_cols <- append(exclude_cols, col)
        }
    }

    exclude_cols

}

exclude_cols <- clean_dat(training2)

# Remove cols from both training and test sets
training_clean <- training2[, !(names(training2) %in% exclude_cols)]
testing_final <- testing2[, !(names(testing2) %in% exclude_cols)]

# Since there are 5 labels for the outcome
# variable and there are 92 features, a
# simple random forest model is trained
set.seed(9265)

# Create validation set
inBuild <- createDataPartition(y=training_clean$classe,
                               p=.75, list=FALSE)

training_final <- training_clean[inBuild,]
validation_final <- training_clean[-inBuild,]

set.seed(4250)
# Create random forest
simple_rf <- randomForest(classe ~ ., data=training_final,
                          importance=TRUE)
# Measure accuracy
pred_simple_rf <- predict(simple_rf, validation_final[,-58])

# Accuracy: 0.9992
xtab <- table(pred_simple_rf, validation_final$classe)
confusionMatrix(xtab)

# Create PCA random forest
preProc <- preProcess(training_final[,-58], method = "pca",
                      pcaComp = 3)
trainPC <- predict(preProc, training_final[,-58])
pca_rf <- randomForest(training_final$classe ~ ., data=trainPC,
                       importance=TRUE)

valPC <- predict(preProc, validation_final[,-58])
# Measure accuracy
pred_pca_rf <- predict(pca_rf, valPC)

xtab2 <- table(pred_pca_rf, validation_final$classe)

# Accuracy: 0.73
confusionMatrix(xtab2)

# Merge the two previous models
predDF <- data.frame(pred_simple_rf, pred_pca_rf, classe=validation_final$classe)

combRf <- randomForest(classe ~ ., data=predDF,
                       importance=TRUE)

# Measure accuracy
combPred <- predict(combRf, predDF)
xtab3 <- table(combPred, validation_final$classe)

# Accuracy: 0.9994
confusionMatrix(xtab3)

# Which model was the most accurate on the validation set?
# As we saw, it was the combined random forest, therefore, it will
# be the final model that will be used for the test set.
quick_df <- data.frame(combPred=combPred)

p1 <- ggplot(quick_df, aes(combPred, fill=combPred)) + geom_bar() +
    ggtitle("Final model predictions") + theme(legend.position = "none")

p2 <- ggplot(validation_final, aes(classe, fill=classe)) + geom_bar() +
    ggtitle("Observed values") + theme(legend.position = "none")

grid.arrange(p2, p1)


