### PART FOUR

In this script, the trained model was applied to the provided test dataset, 
test.csv. Afterward, the predicted results were matched with the click_id 
to produce the submission file.

The test dataset is similar to the training dataset, with the following 
differences:
* click_id: reference for making predictions
* s_attributed: not included



``` r
# Removes all existing objects and packages from the current workspace
rm(list = ls())
# Working directory 
# setwd("~/project1")
# getwd()
```

``` r
# Packages
library(dplyr)
library(data.table)
library(randomForest)
```

``` r
# Loading the model
model15s <- readRDS("model15.rds")
``` 


Loading the test file

The test dataset named test.csv can be found on the web site

https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

``` r
test_set <- fread(file = 'test.csv', header = T, 
                  select = c('click_id', 'ip', 'app', 'channel'))
```

``` r
# ip feature
# Repeated ips in order
n_dupl_ips <- test_set %>%
  count(ip, wt = n(), name = 'repetitions') %>%
  arrange(desc(repetitions))
``` 

``` r
# Number of duplicate ips column
test_set <- left_join(test_set, n_dupl_ips, by = 'ip')
test_set$ip <- NULL
rm(n_dupl_ips)
```

``` r
# repetitions classes
test_set$repetitions_fac <- cut(test_set$repetitions,
                                breaks = c(0,5,nrow(test_set)), 
                                labels = c(1, 2))
test_set$repetitions <- NULL
``` 

``` r
# app classes
test_set$app_fac <- cut(test_set$app,
                        breaks = c(0, 3, 12, 18, nrow(test_set)),
                        right = F, labels = c(1, 2, 3, 4))
gc()
##            used  (Mb) gc trigger   (Mb)  max used   (Mb)
## Ncells  5566151 297.3   13128697  701.2  13128697  701.2
## Vcells 99736171 761.0  547515494 4177.3 680144856 5189.1
``` 


Predictions of the machine learning model
``` r
# Predictions using the model 15s
predictions15 <- predict(model15s, test_set, type = "prob")
head(predictions15)
##   1 0
## 1 0 1
## 2 0 1
## 3 0 1
## 4 0 1
## 5 0 1
## 6 0 1
```


The submission file with the calculated probabilities 
``` r
# for the is_attributed variable
test_set_results <- data.frame(click_id = test_set$click_id, 
                               is_attributed = predictions15[,1])
                               
head(test_set_results)
##   click_id is_attributed
## 1        0             0
## 2        1             0
## 3        2             0
## 4        3             0
## 5        4             0
## 6        5             0

dim(test_set_results)
## [1] 18790469     2
``` 


Saving the submission file
``` r
write.csv(x = test_set_results, file = 'submission_file.csv', row.names = F)
```

``` r
# Number yes (1) or no (0) is_attributed variable
table(round(test_set_results[,2]))
##
##        0        1
## 18346343   444126
```

``` r
# Cleaning the house
rm(list = ls())
gc()
##            used  (Mb) gc trigger   (Mb)  max used   (Mb)
## Ncells  5684916 303.7   13202667  705.1  13202667  705.1
## Vcells 39908833 304.5  546488472 4169.4 671447441 5122.8
```

# THE END
