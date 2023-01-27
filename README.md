# Wine_Quality_Prediction

This document is written to provide aid in understanding the project.

Contents of the document - 
1. Understanding the problem statement 
2. About the dataset
3. Machine learning 
4. Types of machine learning models with examples 
5. Machine learning algorithm used for the model - Random Forest Model 
6. NumPy library 
7. Pandas library 
8. Scikit-learn library 
9. Exploratory data analysis 
10. Fixing missing values in the dataset 
11. Data visualisation - Seaborn 
12. Train-test split 
13. Model evaluation - accuracy 

What is the problem statement for the machine learning algorithm ?

Build a machine learning algorithm to predict the quality of wine on the basis of given features - 
1. volatile acidity - gaseous acids present in wine
2. fixed acidity - primary fixed acids found in wine are tartaric, succinic, citric and malic
3. residual sugar - amount of sugar left after fermentation 
4. citric acid - it is weak organic acid, found in citric fruit naturally 
5. chlorides - amount of salt present in wine
6. total sulphur dioxide - it is used for prevention of spoilage of wine by oxidation and micro-organisms
7. pH - it is used for checking acidity
8. density 
9. sulphates - added sulfites preserve freshness and protect wine from oxidation, and bacteria 
10. alcohol - percentage of wine present in wine 
11. quality - score between 0 to 10 

About the dataset - 

The dataset contains the above mentioned features.
It contains 1600 data points and 12 features. 

Machine learning - 

Machine learning enables the processing of sonar signals and target detection. Machine Learning is a subset of Artificial Intelligence. This involves the development of computer systems that are able to learn by using algorithms and statistical measures to study data and draw results from it. Machine learning is basically an integration of computer systems, statistical mathematics and data.

Machine Learning is further divided into three classes - Supervised learning, Unsupervised learning and Reinforcement Learning. 

Supervised learning is a machine learning method in which models are trained using labelled data. In supervised learning, models need to find the mapping function and find a relationship between the input and output. In this, the user has a somewhat idea of what the output should look like. It is of two types - regression (predicts results with continuous output. For example, given the picture of a person, we have to predict their age on the basis of the given picture) and classification (predict results in a discrete output. For example, given a patient with a tumor, we have to predict whether the tumor is malignant or benign.) 

Unsupervised learning is a method in which patterns are inferred from the unlabelled input data. It allows us to approach problems with little or no idea what the results should look like. We can derive structure from the data where we don’t necessarily know the effect of variables. We can derive the structure by clustering the data based on relationships among the variables in the data. With unsupervised learning there is no feedback on the prediction results. It is of two types - clustering (model groups input data into groups that are somehow similar or related by different variables. For example, clustering data of thousands of genes into groups) and non-clustering (models identifies individual inputs. It helps us find structure in a chaotic environment. For example, the cocktail party problem where we need to identify different speakers from a given audiotape.)

Reinforcement learning is a feedback-based machine learning technique. It is about taking suitable action to maximise reward in a particular situation. For example, a robotic dog learning the movement of his arms or teaching self-driving cars how to depict the best route for travelling. 

For the project, we use Random Forest model - 

It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.

Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset. Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output. The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.

Why Random Forest?
1. It takes less training time as compared to other algorithms.
2. It predicts output with high accuracy, even for the large dataset it runs efficiently.
3. It can also maintain accuracy when a large proportion of data is missing.
4. It is capable of handling large datasets with high dimensionality.
5. It enhances the accuracy of the model and prevents the overfitting issue.

Hyperparameters - 
1. n_estimators– number of trees the algorithm builds before averaging the predictions.
2. max_features– maximum number of features random forest considers splitting a node.
3. mini_sample_leaf– determines the minimum number of leaves required to split an internal node.
4. n_jobs– it tells the engine how many processors it is allowed to use. If the value is 1, it can use only one processor but if the value is -1 there is no limit.
5. random_state– controls randomness of the sample. The model will always produce the same results if it has a definite value of random state and if it has been given the same hyperparameters and the same training data.
6. oob_score – OOB means out of the bag. It is a random forest cross-validation method. In this one-third of the sample is not used to train the data instead used to evaluate its performance. These samples are called out of bag samples.

Python libraries used in the project - 

NumPy  

It is a python library used for working with arrays. It has functions for working in the domain of linear algebra, fourier transform, and matrices. It is the fundamental package for scientific computing with python. NumPy stands for numerical python. 

NumPy is preferred because it is faster than traditional python lists. It has supporting functions that make working with ndarray very easy. Arrays are frequently used where speed and resources are very important. NumPy arrays are faster because it is stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently. This is locality of reference in computer science. 

Pandas - 

Pandas is made for working with relational or labelled data both easily and intuitively. It provides various data structures and operations for manipulating numerical data and time series. 

It has a lot of advantages like - 
1. Fast and efficient for manipulating and analyzing data
2. Data from different file objects can be loaded 
3. Easy handling of missing data in data preprocessing 
4. Size mutability 
5. Easy dataset merging and joining 
6. Flexible reshaping and pivoting of datasets 
7. Gives time-series functionality 

Pandas is built on top of NumPy library. That means that a lot of structures of NumPy are used or replicated in Pandas. The data produced by pandas are often used as input for plotting functions of Matplotlib, statistical analysis in SciPy, and machine learning algorithms in Scikit-learn. 

Scikit-Learn - 

It provides efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction. It has numerous machine learning, pre-processing, cross validation and visualisation algorithms. 

Matplotlib - 

Used for 2D array plots. It includes wide range of plots, such as scatter, line, bar, histogram and others that can assist in delving deeper into trends.

Exploratory data analysis - 

Exploratory data analysis is the process of performing initial investigation on the data to discover patterns or spot anomalies. It is done to test the hypothesis and to check assumptions with the help of summary statistics and graphical representations. 

‘describe()’ method returns description of data in DataFrame. It tells us the following information for each column - 
- Count - number of non-empty values
- Mean - the average (mean) value  
- Std - standard deviation
- Min - minimum value
- 25% - the 25 percentile 
- 50% - the 50 percentile 
- 75% - the 75 percentile
- Max - maximum value

Missing values - 

Missing values are common when working with real-world datasets. Missing data could result from a human factor, a problem in electrical sensors, missing files, improper management or other factors. Missing values can result in loss of significant information. Missing value can bias the results of model and reduce the accuracy of the model. There are various methods of handling missing data but unfortunately they still introduce some bias such as favoring one class over the other but these methods are useful. 

In Pandas, missing values are represented by NaN. It stands for Not a Number. 

Reasons for missing values - 
1. Past data may be corrupted due to improper maintenance
2. Observations are not recorded for certain fields due to faulty measuring equipments. There might by a failure in recording the values due to human error. 
3. The user has not provided the values intentionally. 

Why we need to handle missing values - 
1. Many machine learning algorithms fail if the dataset contains missing values. 
2. Missing values may result in a biased machine learning model which will lead to incorrect results if the missing values are not handled properly. 
3. Missing data can lead to lack of precision. 

Types of missing data - 

Understanding the different types of missing data will provide insights into how to approach the missing values in the dataset. 
1. Missing Completely at Random (MCAR) 
There is no relationship between the missing data and any other values observed or unobserved within the given dataset. Missing values are completely independent of other data. There is no pattern. The probability of data being missing is the same for all the observations. 
The data may be missing due to human error, some system or equipment failure, loss of sample, or some unsatisfactory technicalities while recording the values.
It should not be assumed as it’s a rare case. The advantage of data with such missing values is that the statistical analysis remains unbiased.   
2. Missing at Random (MAR)
The reason for missing values can be explained by variables on which complete information is provided. There is relationship between the missing data and other values/data. In this case, most of the time, data is not missing for all the observations. It is missing only within sub-samples of the data and there is pattern in missing values. 
In this, the statistical analysis might result in bias. 
3. Not MIssing at Random (NMAR)
Missing values depend on unobserved data. If there is some pattern in missing data and other observed data can not explain it. If the missing data does not fall under the MCAR or MAR then it can be categorized as MNAR. 
It can happen due to the reluctance of people in providing the required information. 
In this case too, statistical analysis might result in bias. 

How to handle missing values - 

isnull().sum() - shows the total number of missing values in each columns 

We need to analyze each column very carefully to understand the reason behind missing values. There are two ways of handling values - 
1. Deleting missing values - this is a simple method. If the missing value belongs to MAR and MCAR then it can be deleted. But if the missing value belongs to MNAR then it should not be deleted. 
The disadvantage of this method is that we might end up deleting useful data. 
You can drop an entire column or an entire row. 
2. Imputing missing values - there are various methods of imputing missing values
3. Replacing with arbitrary value 
4. Replacing with mean - most common method. But in case of outliers, mean will not be appropriate
5. Replacing with mode - mode is most frequently occuring value. It is used in case of categorical features. 
6. Replacing with median - median is middlemost value. It is better to use median in case of outliers. 
7. Replacing with previous value - it is also called a forward fill. Mostly used in time series data. 
8. Replacing with next value - also called backward fill. 
9. Interpolation 

Data visualisation - 

Datasets often come in csv files, spreadsheets, table form etc. Data visualisation provides a good and organized pictorial representation of data which makes it easier to observe, understand and analyze. 
Python provides various libraries that come with different features for visualizing data. All these libraries have different features and can support various types of graphs. 
1. Matplotlib - for 2D array plots. It includes wide range of plots, such as scatter, line, bar, histogram and others that can assist in delving deeper into trends. 
2. Seaborn - it is used for creating statistical representations based on datasets. It is built on top of matplotlib. It is built on top of pandas’ data structures. The library conducts the necessary modelling and aggregation internally to create insightful visuals.
3. Bokeh - it is a modern web browser based interactive visualization library. It can create engaging plots and dashboards with huge streaming data. The library contains many intuitive graphs. It has close relationship with PyData tools. The library is ideal for creating customized visuals.  
4. Plotly - python visualization library that is interactive, accessible, high-level and browser-based. Scientific graphs, 3D charts, statistical plots, and financial charts. Interaction and editing options are available. 

Data visualisation library and plot used in the project - 

Seaborn - 

Seaborn aims to make visualization the central part of exploring and understanding data. It provides dataset-oriented APIs, so that we can switch between different visual representations for same variables for better understanding of dataset.

Catplot - 

Parameters - 
1. x, y, hue:  names of variables in data
2. data:  DataFrame
3. row, col:   names of variables in data
4. kind:  str
The kind of plot to draw, corresponds to the name of a categorical axes-level plotting function.
5. color:  matplotlib color
6. palette:  palette name, list, or dict
7. kwargs:  key, value pairings

Barplot - 

A barplot is used for categorical data according to some methods and by default it’s the mean. It can also be understood as a visualization of the group by action. To use this plot we choose a categorical column for the x-axis and a numerical column for the y-axis, and we see that it creates a plot taking a mean per categorical column.

It has similar parameters. 

Heatmap - 

Heatmap is graphical representation of data using colors to visualize value of the matrix. It is a plot of rectangular data as color-encoded matrix. 

Parameters:
1. data: 2D dataset that can be coerced into an ndarray.
2. vmin, vmax: Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
3. cmap: The mapping from data values to color space.
4. center: The value at which to center the colormap when plotting divergent data.
5. annot: If True, write the data value in each cell.
6. fmt: String formatting code to use when adding annotations.
7. linewidths: Width of the lines that will divide each cell.
8. linecolor: Color of the lines that will divide each cell.
9. cbar: Whether to draw a colorbar.

corr() - 

corr() method finds the pairwise correlation of each column in a dataframe. 

There are two types of correlation - 
1. Positive correlation - when the columns are directly proportional to each other 
2. Negative correlation - when the columns are inversely proportional to each other. 


Train-test split - 

The entire dataset is split into training dataset and testing dataset. Usually, 80-20 or 70-30 split is done. The train-test split is used to prevent the model from overfitting and to estimate the performance of prediction-based algorithms. We need to split the dataset to evaluate how well our machine learning model performs. The train set is used to fit the model, and statistics of training set are known. Test set is for predictions. 

This is done by using scikit-learn library and train_test_split() function. 

Parameters - 
1. *arrays: inputs such as lists, arrays, data frames, or matrices
2. test_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our test size. its default value is none.
3. train_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our train size. its default value is none.
4. random_state: this parameter is used to control the shuffling applied to the data before applying the split. it acts as a seed.
5. shuffle: This parameter is used to shuffle the data before splitting. Its default value is true.
6. stratify: This parameter is used to split the data in a stratified fashion.

Model evaluation - 

Model evaluation is done to test the performance of machine learning model. It is done to determine whether the model is a good fit for the input dataset or not. 

In this case, we use accuracy. Accuracy is a performance metrics that is used to test a binary classification model. Accuracy measures the proportion of correct prediction to total data points.

Accuracy = ( tp + tn) / ( tp + fp + tn + fn )

- Tp - true positive. This refers to the total number of observations that belong to the positive class and have been predicted correctly. 
- Tn - true negatives. It is total number of observations that belong to the negative class and have been predicted correctly 
- Fp - false positives. It total number of observations that have been predicted to belong to positive class, but instead belong to the negative class. 
- Fn - false negatives. It is total number of observations that have been predicted to be a part of negative class but instead belong to the positive class.
