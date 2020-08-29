## PART THREE

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
## [1] 3197922
``` 

``` r
# Changing some features to factor
train_set <- train_set %>%
  mutate(is_attributed = factor(is_attributed, levels = c(1,0))) %>%
  mutate(repetitions_fac = factor(repetitions_fac, levels = c(1,2))) %>%
  mutate(app_fac = factor(app_fac, levels = c(1,2,3,4))) 
  
str(train_set)
## Classes ‘data.table’ and 'data.frame':	3197922 obs. of  5 variables:
##  $ is_attributed  : Factor w/ 2 levels "1","0": 2 2 2 2 2 2 2 2 2 2 ...
##  $ app            : num  2 46 3 12 24 3 1 15 11 8 ...
##  $ channel        : num  469 347 137 178 178 489 134 140 173 145 ...
##  $ repetitions_fac: Factor w/ 2 levels "1","2": 2 2 2 2 2 2 2 2 2 2 ...
##  $ app_fac        : Factor w/ 4 levels "1","2","3","4": 1 4 2 3 4 2 1 3 2 2 ...
##  - attr(*, ".internal.selfref")=<externalptr> 

gc()
##            used  (Mb) gc trigger  (Mb) max used  (Mb)
## Ncells  2472645 132.1    4384280 234.2  3547667 189.5
## Vcells 23403555 178.6   51165512 390.4 42571260 324.8
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
``` 

#### Continue on part three, filename project_click_fraud_4_training_the_model.R
