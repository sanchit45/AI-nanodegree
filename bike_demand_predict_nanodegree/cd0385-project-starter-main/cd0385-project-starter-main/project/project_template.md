# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Sanchit Baweja

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?

I realized it is necessary to drop negative values from predictions and replace them with 0 , since kaggle does not accept submissions with negative value.

### What was the top ranked model that performed?

My top ranked model was model with additional features with a kaggle score of 0.65651.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?

In EDA i observed histogram of each feature distribution relative with data.I observed datetime histogram was Plateau Distribution, (temp,atemp,humidity) histogram was  truncated distribution and (windspeed,count) were right skewed distribution.

I added hour,day and month features from "datetime" column e.g, df["hour"]=df["hour"].dt.hour in order to improve performance.

### How much better did your model preform after adding additional features and why do you think that is?
Model with additional features performed 63.35 % better than the initial model. I believe its because of the more information gain from the new features i.e, hours,day,month generated from datetime column that made perform better.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Model with additional features performed 58.96% better than the initial model.

### If you were given more time with this dataset, where do you think you would spend more time?
If given time , i would spend more time in training model with autogluon since it is mostly automated and require negligible intervention , and i will also spend time on hyperparameters optimization.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|time_limit|presets|eval_metric|score|
|--|--|--|--|--|
|initial|600|best_quality|root_mean_squared_error|1.79171| 			
|add_features|600|best_quality|root_mean_squared_error|0.65651| 	
|hpo|720|high_quality|r2|0.73156|			

### Create a line plot showing the top model score for the three (or more) training runs during the project.



![model_train_score.png](cd0385-project-starter/project/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.



![model_test_score.png](cd0385-project-starter/project/model_test_score.png)

## Summary
In this project data from bike share demand competition was used to predict demand of bike containing features such as temperature, humidity, datetime, windspeed etc .I did exploratory data analysis on data and observed histogram of features distribution with respect to data and make additional features like hours,day,month from datetime feature. One model was trained with default parameters ,second was with additional features and third was hyperparameter optimised with a kaggle score of 1.79171,0.65651 and 0.73156 as depicted by second graph. As we can see in first graph model with hyperparameters optimized had highest performance for score but still medium kaggle score which i believe is due to overfitting.
