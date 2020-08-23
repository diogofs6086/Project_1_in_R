## Project 1 in R

The objective of this project is to predict whether a user will download an app after clicking a mobile app advertisement. The datasets are from Kaggle, https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data. This project is part of the Data Science course formation of Data Science Academy from Brazil.


## Pipeline of the given solution

The solution to this problem was divided into four parts. The first part is 
in this script. It deals with the data munging and the testing of many machine 
learning models using the train_sample.csv file and testing with 1E+07 rows of 
the train.csv. The data of this file was used as the test dataset because the 
dataset provided did not have the target variable.

The second part of the solution got the main tidying lines of part one to tidy 
the full training dataset, nominated train.csv. In the third part, the tidying 
training dataset was taken with the best model acquired in part one to train 
the model, but the number of the trees of the random forest model was reduced 
due to my notebook capacity. In the fourth part, the trained model was applied 
to the provided test dataset, test.csv. Afterward, the predicted results were 
matched with the click_id to produce the submission file.

A script parts are below:
* [Part 1 - Data munging and testing models](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md)
* [Part 2 - Tidying the training dataset](source_githubio/project_click_fraud_2_tidying_in_the_train_dataset.md)
* [Part 3 - Trainig the choosen model](source_githubio/project_click_fraud_3_training_the_model.md)
* [Part 4 - Predictions](source_githubio/project_click_fraud_4_predictions_with_the_test_dataset.md)

```
