### PART THREE

In this script, the tidying training dataset was taken with the best model 
acquired in part one to train the model, but the number of the trees of the 
random forest model was reduced due to my notebook capacity.

The test dataset is similar to the training dataset, with the following 
differences:
click_id: reference for making predictions
is_attributed: not included

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
library(caret)
library(randomForest)
library(DMwR)
```

``` r
# Reading the training dataset
train_set <- fread(file = 'train_set.csv', header = T) %>% select(-V1)
nrow(train_set)
``` 

``` r
# Changing some features to factor
train_set <- train_set %>%
  mutate(is_attributed = factor(is_attributed, levels = c(1,0))) %>%
  mutate(repetitions_fac = factor(repetitions_fac, levels = c(1,2))) %>%
  mutate(app_fac = factor(app_fac, levels = c(1,2,3,4))) 
str(train_set)
gc()
``` 

``` r
# Random forest model
model15 <- randomForest(is_attributed ~ repetitions_fac * app + 
                          channel * app_fac, 
                        data = train_set, 
                        ntree = 10,
                        nodesize = 1)
``` 

``` r
# Saving the model
saveRDS(model15, file = "model15.rds")
``` r

#### Continue on part three, filename project_click_fraud_4_training_the_model.R
