# COM SCI M148 Final Project Report

Morgan Mason, Andrew Zhang, Pallavi Srinivas 
Github Link: https://github.com/morganmason0606/24F_CSM148_Final


# Overview
## I) Dataset
Our team chose to use the Student performance Dataset. This dataset consists of synthetic data, with the observational unit being one student and each observation measuring: Hours_Studied, Attendance, Parental_Involvement, Access_to_Resources, Extracurricular_Activities, Sleep_Hours, Previous_Scores, Motivation_Level, Internet_Access, Tutoring_Sessions, Family_Income, Teacher_Quality, School_Type, Peer_Influence, Physical_Activity, Learning_Disabilities, Parental_Education_Level, Distance_from_Home, Gender, Exam_Score.
## II) Objective
Our team wanted to explore which factors contribute the greatest to improvements in student performance. We wished to see how this dataset’s features could best predict if a student would see an improvement from Previous_Scores to Exam_Scores (a feature we called Score_Change). By analyzing the relationship between exam scores and other factors that are present in a students life, we can make conclusions about what changes students can make in their current routines to improve their academic performance.

## III) Methodology
We explored different models to predict if a student would improve their score or not: logistic regression, KNN, Random Forests, and Neural Networks. Other methods were also used to help us understand and explore the data: linear regression with lasso regularization was used to identify irrelevant variables, PCA was explored for processing the data, and clustering was used to evaluate the structure of our data. We eventually decided that logistic regression and Neural Networks were the most successful ways to answer our problem: they are very accurate and there are methodologies to quickly explain the relative importance of features on the output.

## IV) Results
We wished to predict if a student would improve their test score or not. Because this is a classification problem, we chose to use classification evaluation metrics like accuracy, precision, recall, AUC scores, f1 scores, and confusion matrixes to evaluate our models. Our two most successful models were our logistic regression model and neural network who were able to both achieve f1 scores of around 98%. 

Our neural network was singe-hidden-layer neural network with Binary Cross Entropy Loss and an Adam optimizer that takes all features (which are normalized) except exam score and classifies if a given student will see any improvement. Using cross validation, we chose a learning rate from the range of 1e-5 to 1 and a hidden layer size from 40 to 6 (between feature size and square root of feature size). Finally, we created model with the discovered hyperparameters. 

Our logistic regression model was trained to predict if the score would increase given the full normalized data set except for the final exam score. We also performed cross validation to pick an appropriate l2 regularization constant. Our logistic regression model is our most useful because it is easiest to analyze. From our logistic regression models’ parameters we note the following
- Studying and attending class are associated students improving their scores: an increase in 1 standard deviation in the number of hours studied per week lead to an increase of 1.5 log odds and an increase in 1 standard deviation in percent attendance increases the log odds by 2 percent.
- Higher parental involvement, access to resources, motivation, family income, teacher quality, and parental education level are all better than being low or medium.
- Having a higher previous score seriously hurts one's chance of improvement: one hypothesis as to why is that students with higher scores will have less room for improvement.
	
Our main model is our logistic regression model because of its effectiveness and ease of explanation. However, one weakness is that a logistic regression model assumes a linear relation. As such, we treat our neural network as a secondary model because although it is less explainable, it can better handle nonlinear relationships between data. In any case, our logistic regression model performs very well and helps illuminate that students should continue to study and attend class and parents and communities should place more support into providing high quality resources.

## V) Using Our Code
Our code is available at the github link at the top of the page. In the notebook, there is information about the python libraries and data needed to run the notebook. Each section of the notebook can be run independently, but you must have “transformed_data.csv”, which is created by running “ii)Data Preprocessing”, in the same directory as the notebook before running any section after Section II.
# Appendix
## I) Exploratory Data Analysis
Part of our exploratory data analysis included calculating the 5-Number Summary (smallest data point, lower quartile, median, upper quartile, largest data point), mean, standard deviation, skewness, and kurtosis for each of the variables we chose to analyze. We also visualize correlations between features through a correlation heatmap, finding that most of our predictor variables are not correlated with each other, and that Hours_Studied, Attendance, and Previous Scores are the most correlated with Score_Change. For quantitative data we plotted a histogram and box plot. For categorical variables, we plotted the boxplot of exam scores per each category. Finally, we explored the distribution of observations with missing data. By comparing the population with the subset missing a feature, we decided that during data processing we would drop missing values because they do not significantly affect the whole dataset.

## II) Data Pre-Processing and Feature Engineering
We began our data preprocessing by cleaning our data and removing any NaN values that may have been present across all of the feature variables present in the original dataset. Through this process, we saw that the dataset did not contain very many NaN values relative to its size. Data preprocessing methods we decided to implement include removing rows that contained NaN values, rows with Exam_Score greater than 100, one-hot encoding categorical variables, renaming columns, and making a column to represent the improvement between Exam_Score and Previous_Score (Score_Change). We will use this transformed dataset to calculate future metrics.

## III) Regression Analysis
Regression analysis was used in our project to begin exploring building logistic regression models and identifying potentially irrelevant variables.

Our first model attempted to predict a Score_Change from a subset of our variables (Hours_Studied_Per_Week, Attendance, Sleep_Hours, and Motivation_Level). We normalized our data, fit the model, and then calculated the mean squared error (MSE), mean absolute error (MAE), and R2 scores. Looking at these values, the model does not seem well suited, as our MSE is high and our R2 score is close to 0. To address this, our second model utilized  all the feature variables from the transformed dataset and standardize them. We will only standard-deviation scale our response variable, Score_Change, because we want to see whether or not student exam performance is improving or not. Performing Linear Regression on this new dataframe, we see that while the R2 score is much better but our coefficients are very high. To infer more information, we can apply Lasso Regression to decrease the coefficients to identify the most important features in the dataset.

Our third model fit our test dataset to a model that used  lasso regression and cross validation to pick a regularization constant. This resulted in much more reasonable coefficients while maintained a high R2 score and a low MSE. From our Lasso Regression Model, we learned that hours of sleep, gender, and private vs. public schools had little to no impact on our model as they all had either coefficients that were either 0 or very small. Likewise, previous scores, hours studied, and attendance have the largest coefficients, indicating that they have the largest impact on exam performance.

## IV) Logistic Regression Analysis
We fit our training dataset on our Logistic Regression model to predict if a student's exam score will improve (high Score_Change datapoint). We scaled our data, similarly to how it was scaled in our linear regression model, and also performed cross validation to pick an appropriate regularization constant. After selecting and training our model, we received the coefficients for all of the feature variables, created a confusion matrix and used several measures to evaluate our model: true negative rate, prediction error, accuracy, F1 score, and ROC curve/AUC score. Our model performed very well as we have a high accuracy, recall, true negative rate, and a strong ROC curve.

One benefit of using a logistic regression model is that we can easily analyze the effects of our features. For each feature, an increase in one of feature corresponds to an increase in coefficient-amount log odds. Furthermore, since it has been trained on mean-centered and scaled data, we can also compare coefficients between features and their relative importance.

## V) KNN and Random Forest
We used KNN and random forests as another way to predict if students will improve their scores. 
For random forests, we used cross-validation to find the ideal depth to use. We then evaluated the random forest (through training) and generated a confusion matrix to visualize our data and other classification evaluation metrics. From this, we could see that our random forest's performance is comparable to logistic regression, albeit slightly worse.

With KNN we used cross-validation to pick the number of clusters. We scaled the data using StandardScaler() and ran KNN on this scaled data. We then generated a confusion matrix to visualize our findings, and from it, we could see that the KNN method was similar to the random forest but was also still slightly worse than the logistical regression. 

Though both models were very accurate, logistic regression is not only more accurate, but it is also easier to explain. As a result, we did not end up using KNN or Random Forest.

## VI) PCA and Clustering
How were PCA and clustering applied on your data? What method worked best for your data and why was it good for the problem you were addressing? 

We explored using PCA on our dataset. After performing principal component analysis, we noticed that one principal component contributed to 34% of the variability and the rest contributed to less than 6% each. We (arbitrarily) chose to only consider PC’s that contributed more than 2% of the data’s variability, leaving us with 16 principal components. We plotted our transformed data on the first 3 principal components and noted that the data seemed to have separated, indicating using transformed data could be useful for separating data.

Because PCA had visually helped separate the data, we explored using the PCA transformed data in clustering. We first explored KMeans clustering. Because we already know that we have two labels, we chose to separate the data into two clusters. Likewise, we also explored Agglomerative clustering, choosing to build clusters until we had two clusters. While we were creating our clusters, we also used the labels to check how well the clusters aligned with the actual label; we found that our clustering methods were only slightly better than random chance. After creating both clusterings we calculated the silhouette scores of  each method and the rand score between them. Our silhouette scores were positive but small, implying that the data is loosely clustered but clustered well. Our rand score was .67, implying that the clusters were somewhat similar

Though PCA did seem useful initially, there doesn’t seem to be much justification for using it: our correlation heat map did not indicate high correlation and our normally scaled data was able to produce good results. Similarly, because we already have labels for our data, clustering was not very useful beyond gaining a better understanding of how our data is separated.

## VII) Neural Network
Our project uses a neural network as a secondary way to predict outcomes. We created a simple neural network model  with a single hidden layer and trained with Binary Cross Entropy loss and the Adam optimizer.. We used cross validation to pick a learning rate and hidden layer size before training a full neural net. 

After training our network, we tested it by looking at its confusion matrix and calculating the accuracy, precision, recall, f1 score, ROC AUC score, and graphed the ROC graph. We discovered that our neural network was performing comparably to our logistic regression model. We ended up keeping our neural network model to complement our logistic regression; neural networks can represent nonlinear relations that logistic regression cannot, but it is easier to understand and explain the impact of features through logistic regression’s coefficients.

## VIII) Hyperparameter Tuning
For our neural network, an example of hyperparameter tuning that we applied was using different learning rates for optimization. For this, we tested our neural network model by training and validating our data with different learning rates and utilizing 5-fold cross validation to check how our model performed across different subsets of the data.

Another example of hyperparameter tuning was using hidden dimensions. Hidden dimensions define the number of units/neurons in the hidden layer of our neural network model, and for this we would vary the hidden dimension. 

From this, we were able to see the results with different learning rates and different hidden dimensions and thus select the best combination of each that gives the best cross-validation score.
