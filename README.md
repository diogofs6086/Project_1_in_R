# Project 1 in the R languange: Predictions whether a user will download an app after clicking a mobile app advertisement in R
<center> <h3>Diogo F. dos Santos</h3> </center>
<center><h4>August 9th, 2020</h4></center>


The objective of this project is to predict whether a user will download an app after clicking a mobile app advertisement. The datasets are from Kaggle, [click here](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data) to see. This project is part of the Data Science course formation of Data Science Academy from Brazil.


## Pipeline of the given solution

The solution to this problem was divided into four parts. The first part deals
with the data munging and the testing of many machine learning models using 
the train_sample.csv file and testing with 1E+07 rows of the train.csv. The 
data of this file was used as the test dataset because the provided dataset
did not include the target variable. The second part of the solution got the 
main tidying lines of part one to tidy the full training dataset, nominated 
train.csv. In the third part, the tidying training dataset was taken with the 
best model acquired in part one to train the model, but the number of the trees 
of the random forest model was reduced due to my notebook capacity. In the 
fourth part, the trained model was applied to the provided test dataset, 
test.csv. Afterward, the predicted results were matched with the click_id to 
produce the submission file.



A script parts are below:

* [PART 1 - Data munging and testing models](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#part-one)
  * [Data fields](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#data-fields)
  * [Exploratory data analysis](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#exploratory-data-analysis)
    * [Ips analysis](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#ips-analysis)
    * [Click features](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#click-features)
    * [Attributed time features](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#attributed-time-features)
    * [App id feature](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#app-id-feature)
    * [OS feature](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#os-feature)
    * [Channel feature](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#channel-feature)
    * [Features with missing values](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#features-with-missing-values)
    * [Reducing the quantity of not downloaded to balance the train target feature](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#reducing-the-quantity-of-not-downloaded-to-balance-the-train-target-feature)
  * [Models](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#models)
    * [Model 1 - Logistic regression model](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#logistic-regression-model)
    * [Model 2 - Logistic regression model with the most significant variables](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#logistic-regression-model-with-the-most-significant-variables)
    * [Model 3  - KSVM model with rbf kernel](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#ksvm-model-with-rbf-kernel)
    * [Model 4  - KSVM model with rbf kernel and the most significant variables](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#ksvm-model-with-rbf-kernel-and-the-most-significant-variables)
    * [Model 5  - KSVM model with vanilladot Linear kernel](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#ksvm-model-with-vanilladot-linear-kernel)
    * [Model 6  - KSVM model with vanilladot Linear kernel and the most significant variables](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#ksvm-model-with-vanilladot-linear-kernel-and-the-most-significant-variables)
    * [Model 7  - SVM model with radial kernel](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#svm-model-with-radial-kernel)
    * [Model 8  - SVM model with radial kernel and the most significant variables](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#svm-model-with-radial-kernel-and-the-most-significant-variables)
    * [Model 9  - SVM model with linear kernel](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#svm-model-with-linear-kernel)
    * [Model 10 - SVM model with linear kernel and the most significant variables](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#svm-model-with-linear-kernel-and-the-most-significant-variables)
    * [Model 11 - Regression Trees model](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#regression-Trees-model)
    * [Model 12 - Regression Trees model with the most significant variables](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#regression-trees-model-with-the-most-significant-variables)
    * [Model 13 - Another Regression Trees model](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#another-regression-trees-model)
    * [Model 14 - Another Regression Trees model with the most significant variables](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#another-regression-trees-model-with-the-most-significant-variables)
    * [Model 15 - Random Forest model](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#random-forest-model)
    * [Model 15 - Random forest model balanced by reducing the major target class](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#random-forest-model-balanced-by-reducing-the-major-target-class)
    * [Model 15 - Random forest model balanced by increasing the minor target class](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#random-forest-model-balanced-by-increasing-the-minor-target-class)
    * [Model 15 - Random forest model balanced by SMOTE](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#random-forest-model-balanced-by-smote)
    * [Model 15 - Random forest model balanced by ROSE](source_githubio/project_click_fraud_1_data_munging_and_testing_models_in_a_sample.md/#random-forest-model-balanced-by-rose)
  
  
* [PART 2 - Tidying the training dataset](source_githubio/project_click_fraud_2_tidying_in_the_train_dataset.md/#part-two)


* [PART 3 - Trainig the choosen model](source_githubio/project_click_fraud_3_training_the_model.md/#part-three)


* [PART 4 - Predictions](source_githubio/project_click_fraud_4_predictions_with_the_test_dataset.md/#part-four)
