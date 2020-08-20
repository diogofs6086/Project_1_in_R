## PART TWO

This script got the main tidying lines of part one to tidy the full 
training dataset, nominated train.csv.

``` r
# Removes all existing objects and packages from the current workspace
rm(list = ls())
# Working directory 
# setwd("~/project1")
# getwd()

# Packages
library(dplyr)
library(data.table)
library(caret)
library(randomForest)
library(DMwR)
```

Number of rows in the train dataset

The train dataset named train.csv can be found on the web site

https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

``` r
n_rows <- fread(file = 'train.csv', header = T, select = 'is_attributed')
n_rows <- nrow(n_rows)
n_rows    # 184.903.890 rows
## [1] 184903890
gc()
##            used  (Mb) gc trigger   (Mb)  max used   (Mb)
## Ncells  3023909 161.5   14360160  767.0  22293634 1190.7
## Vcells 20116531 153.5  379220964 2893.3 508565428 3880.1
``` r
Calculating the number of batches
``` r
for (i in c(15:100)) {
  if (n_rows%%i == 0) {
    print(c(i, n_rows/i))
  }
}             # 15 seems better for my computer capacity
## [1]      15 12326926
## [1]      30 6163463
## [1]      73 2532930

rm(i)
``` r

Batches
``` r
n = 15
train_set <- data.frame(is_attributed = c(),
                        app = c(),
                        channel = c(),
                        repetitions_fac = c(),
                        app_fac = c())
```

The training dataset transformation

``` r
for (i in c(0:(n-1))) {
  if (i == 0) {
    # The train dataset named train.csv can be found on the web site
    # https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
    train <- fread(file = 'train.csv', header = T, 
                 skip = n_rows/n*i, nrows = n_rows/n,
                  select = c('is_attributed', 'ip', 'app', 'channel'))
                 } else {
    train <- fread(file = 'train.csv', header = F, 
                   skip = n_rows/n*i, nrows = n_rows/n,
                   select = c(8,1,2,5))
    names(train) <- c('is_attributed', 'ip', 'app', 'channel')
  }
 
  # ip feature
  # Repeated ips in order
  n_dupl_ips <- train %>%
    count(ip, wt = n(), name = 'repetitions') %>%
    arrange(desc(repetitions))
  
  # Number of duplicate ips column
  train <- left_join(train, n_dupl_ips, by = 'ip')
  train$ip <- NULL
  
  # repetitions classes
  train$repetitions_fac <- cut(train$repetitions,
                               breaks = c(0,5,nrow(train)), 
                               labels = c(1, 2))
  train$repetitions <- NULL
  
  # app classes
  train$app_fac <- cut(train$app,
                       breaks = c(0, 3, 12, 18, nrow(train)),
                       right = F, labels = c(1, 2, 3, 4))
  
  # is_attributed classes
  train <- train %>%
    mutate(is_attributed = factor(is_attributed, levels = c(1,0)))
  head(train_set)
  
  # Balancing the target class
  train <- SMOTE(is_attributed ~ ., data  = train)
  
  # Binding the train dataset
  train_set <- rbind(train_set, train)
  
  rm(n_dupl_ips, train)
  gc()
  print(i)
}
## [1] 0
## [1] 1
## [1] 2
## [1] 3
## [1] 4
## [1] 5
## [1] 6
## [1] 7
## [1] 8
## [1] 9
## [1] 10
## [1] 11
## [1] 12
## [1] 13
## [1] 14

# training data set dimension
dim(train_set)
## [1] 3197922  5

# Number of downloads, indicated by "1"
table(train_set$is_attributed) 
##        1       0
##  1370538 1827384
```

Saving the tidy train dataset 

``` r
write.csv(x = train_set, file = 'train_set.csv')
```

#### Continue on part three, filename project_click_fraud_3_predictions_with_the_test_dataset.R
