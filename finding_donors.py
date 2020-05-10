#!/usr/bin/env python
# coding: utf-8

# # Data Scientist Nanodegree
# ## Supervised Learning
# ## Project: Finding Donors for *CharityML*
# 
# <div class="alert alert-info">
#     
# **Note:** This notebook was created with Python 3.7.7, scikit-learn 0.22.1, and numpy 1.18.1
# 
# </div>

# Welcome to the first project of the Data Scientist Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Please specify WHICH VERSION OF PYTHON you are using when submitting this notebook. Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. 
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The dataset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

# ----
# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, `'income'`, will be our target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database.

# In[60]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))


# ### Implementation: Data Exploration
# A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \$50,000. In the code cell below, you will need to compute the following:
# - The total number of records, `'n_records'`
# - The number of individuals making more than \$50,000 annually, `'n_greater_50k'`.
# - The number of individuals making at most \$50,000 annually, `'n_at_most_50k'`.
# - The percentage of individuals making more than \$50,000 annually, `'greater_percent'`.
# 
# ** HINT: ** You may need to look at the table above to understand how the `'income'` entries are formatted. 

# In[61]:


# TODO: Total number of records
n_records = len(data.index)

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = len(data[data.income == '>50K'].index)

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = len(data[data.income == '<=50K'].index)

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = 100*n_greater_50k/n_records

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))


# ** Featureset Exploration **
# 
# * **age**: continuous. 
# * **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
# * **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
# * **education-num**: continuous. 
# * **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
# * **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
# * **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
# * **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. 
# * **sex**: Female, Male. 
# * **capital-gain**: continuous. 
# * **capital-loss**: continuous. 
# * **hours-per-week**: continuous. 
# * **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# ----
# ## Preparing the Data
# Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

# ### Transforming Skewed Continuous Features
# A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number.  Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. 
# 
# Run the code cell below to plot a histogram of these two features. Note the range of the values present and how they are distributed.

# In[62]:


# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)


# For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.
# 
# Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed. 

# In[63]:


# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)


# ### Normalizing Numerical Features
# In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.
# 
# Run the code cell below to normalize each numerical feature. We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this.

# In[64]:


# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))


# ### Implementation: Data Preprocessing
# 
# From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.
# 
# |   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |
# | :-: | :-: |                            | :-: | :-: | :-: |
# | 0 |  B  |  | 0 | 1 | 0 |
# | 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
# | 2 |  A  |  | 1 | 0 | 0 |
# 
# Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively. In code cell below, you will need to implement the following:
#  - Use [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) to perform one-hot encoding on the `'features_log_minmax_transform'` data.
#  - Convert the target label `'income_raw'` to numerical entries.
#    - Set records with "<=50K" to `0` and records with ">50K" to `1`.

# When no `columns` parameter is specified, `pd.get_dummies` will convert all columns with dtype `object` and `category`. Let's check this first:

# In[65]:


print(features_log_minmax_transform.dtypes)


# Alright, it's safe to use `get_dummies` directly:

# In[66]:


# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
# 
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.map({'<=50K': 0, '>50K': 1})

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
# print(encoded)


# ### Shuffle and Split Data
# Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.
# 
# Run the code cell below to perform this split.

# In[67]:


# Import train_test_split
# from sklearn.cross_validation import train_test_split # scikit-learn < 0.18
from sklearn.model_selection import train_test_split # scikit-leran >= 0.18

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ----
# ## Evaluating Model Performance
# In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a *naive predictor*.

# ### Metrics and the Naive Predictor
# *CharityML*, equipped with their research, knows individuals that make more than \\$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \\$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performance would be appropriate. Additionally, identifying someone that *does not* make more than \\$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \\$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:
# 
# $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
# 
# In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).
# 
# Looking at the distribution of classes (those who make at most \\$50,000, and those who make more), it's clear most individuals do not make more than \\$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \\$50,000"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \\$50,000, *CharityML* would identify no one as donors. 
# 
# 
# #### Note: Recap of accuracy, precision, recall
# 
# **Accuracy** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).
# 
# **Precision** tells us what proportion of messages we classified as spam, actually were spam.
# It is a ratio of true positives (words classified as spam, and which are actually spam) to all positives (all words classified as spam, irrespective of whether that was the correct classification), in other words it is the ratio of
# 
# `[True Positives/(True Positives + False Positives)]`
# 
# **Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam.
# It is a ratio of true positives (words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of
# 
# `[True Positives/(True Positives + False Negatives)]`
# 
# For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average(harmonic mean) of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score (we take the harmonic mean as we are dealing with ratios).

# ### Question 1 - Naive Predictor Performace
# * If we chose a model that always predicted an individual made more than $50,000, what would  that model's accuracy and F-score be on this dataset? You must use the code cell below and assign your results to `'accuracy'` and `'fscore'` to be used later.
# 
# ** Please note ** that the the purpose of generating a naive predictor is simply to show what a base model without any intelligence would look like. In the real world, ideally your base model would be either the results of a previous model or could be based on a research paper upon which you are looking to improve. When there is no benchmark model set, getting a result better than random choice is a place you could start from.
# 
# ** HINT: ** 
# 
# * When we have a model that always predicts '1' (i.e. the individual makes more than 50k) then our model will have no True Negatives(TN) or False Negatives(FN) as we are not making any negative('0' value) predictions. Therefore our Accuracy in this case becomes the same as our Precision(True Positives/(True Positives + False Positives)) as every prediction that we have made with value '1' that should have '0' becomes a False Positive; therefore our denominator in this case is the total number of records we have in total. 
# * Our Recall score(True Positives/(True Positives + False Negatives)) in this setting becomes 1 as we have no False Negatives.

# In[68]:


# Counting the ones as this is the naive case.
# Note that 'income' is the 'income_raw' data encoded to numerical values done
# in the data preprocessing step.
TP = np.sum(income)
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case

# TODO: Calculate accuracy, precision and recall
accuracy = (TP+TN)/income.count()
recall = TP/(TP+FN)
precision = TP/(TP+FP)

# TODO: Calculate F-score using the formula above for beta = 0.5 and
# correct values for precision and recall.
beta = 0.5
fscore = (1+np.power(beta,2))*precision*recall/(np.power(beta,2)*precision + recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# ###  Supervised Learning Models
# **The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent Classifier (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Question 2 - Model Application
# List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen
# 
# - Describe one real-world application in industry where the model can be applied. 
# - What are the strengths of the model; when does it perform well?
# - What are the weaknesses of the model; when does it perform poorly?
# - What makes this model a good candidate for the problem, given what you know about the data?
# 
# ** HINT: **
# 
# Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer.

# **Answer:**
# 
# #### RandomForest
# 
# **Real-world application**
# 
# Due to the high accuracy of Random Forests, they are often used in medical research ([7], [17]). They are also, for example, used in malware detection (section 4.4. in [4]).
# 
# **Strengths**
# 
# * Very fast, even on large data sets [15]
# * Small to no risk of overfitting [15]
# * Comparatively very good accuracy [16]
# * Can identify important features (see sklearn documentation)
# * Robust to outliers and noise [16]
# 
# **Weaknesses**
# 
# * More samples don't necessarily lead to higher accuracy [18]
# * Time-consuming and memory-intensive to construct many deep trees [19]
# * An ensemble model is inherently less interpretable than an individual decision tree [19]
# * Predictions may be slow, which is a problem in applications that require very low latency [19]
# 
# **Why a good candidate for our problem?** 
# 
# First and foremost, since they can produce models with high accuracy, they match one of the requirements in the CharityML use case.
# 
# And they are fast, which should be helpful since our dataset is large (with lots of samples).
# 
# More generally, since Random Forests can winnow out important features in the data set, that might help in the future to elicit more relevant data to increase the accuracy further.
# 
# #### AdaBoost
# 
# **Real-world application**
# 
# * Smart city applications [8], e.g. "SignFinder" (facilitating the navigation of blind and visually impaired people in city streets) [20] or detecting indoor/outdoor places with WiFi signals [21]
# 
# **Advantages**
# 
# * Aimed at improving accuracy
# * Fast with large feature sets [8]
# * Reduced training time [8]
# * No prior knowledge of weak learners required ([8], [25])
# 
# **Disadvantages**
# 
# * Susceptible to overfitting, low margins [8]
# * "can fail to perform well given insufficient data, overly complex weak hypotheses or weak hypotheses which are too weak" [25]
# * Sensistive to uniform noise ([8], [25])
# 
# **Why a good candidate for our problem?**
# 
# In general, we are aiming for high accuracy, and AdaBoost is specifically intended to achieve that. Also, the approach of intelligently combining weak learners should help weed out unimportant features (if any).
# 
# #### Support Vector Machines
# 
# **Real-world application**
# 
# One typical application is Text Categorization (see ch. 15 in [5], also [6], [11]). Spam detection is a ubiquitous problem and comes to mind first. Another area could be automatic classification of web pages (e.g. sports, news, finance, etc.) so that ad servers can place context-relevant ads in them. In text categorization, the feature vector representing a text document (web page, etc.) is a vector of word counts (the vector is normalized to unit length to abstract different document lengths). Since every word (or, more likely, word stem) represents one dimension, the problem is very high-dimensional (easily in the tens of thousands). According to [11], SVMs "ability to learn can be _independent of the dimensionality of the feature space_. At the same time, the feature vectors are very sparse, for which SVMs are also well suitable (also mentioned in [11]).
# 
# Other areas in which SVMs are mentioned are
# 
# * Malware Detection (see section 4.4 in [4])
# * Smart City Applications [8]
# * BioInformatics [9]
# 
# **Advantages**
# 
# * Requires full labeling of input data [12]
# * Works well with high-dimensional, sparse feature spaces [11], [14]
# * Simultaneously minimize the classification error and maximize the geometric margin [12]
# 
# **Disadvantages**
# 
# * SVM as such is only directly applicable for two-class tasks. [12] (_Multiclass SVM_ usually boils down to reducing a multiclass problem to several binary decision problems. [13])
# * Sensitive to feature scaling [10]. This is fine since we already scaled the data.
# * Runtime complexity. The algorithm is, data-set dependent, quadratic or cubic in the number of samples (multipled by the number of features). But note that implementations for the linear case are just O(n). [3]
# * Probability estimates don't come out of the box, but require cross-validation, which is computationally expensive [14]
# 
# 
# **Why a good candidate for our problem?** 
# 
# First of all, since we have very many samples, we can expect a high running time, but let's try it out. On the plus side, our feature space is very sparse, and this SVM handles well.
# 
# Given our use case, where we are aiming at high accuracy and high precision, intuitively, it feels that the maximizing the margin is good in this scenario.
# 
# #### What not to use
# 
# Here are some approaches which I don't think are appropraite:
# 
# * Any of sklearn's **Naive Bayes** implementations (Gaussian, Multinomial, Bernoulli, etc.). Those won't work (or produce a bad result) because our features are of mixed types. We have a lot of categorical values (one-hot encoded, so binary), but also some continuous values. We could categorize the continuous values along percentiles, or we could separately model those features, but let's not overcomplicate things. (https://stackoverflow.com/questions/14254203/mixing-categorial-and-continuous-data-in-naive-bayes-classifier-using-scikit-lea )
# * Simple **Decision Trees**, those are just too simple an approach and too prone for overfitting
# 
# Finally, I haven't tried **Stochastic Gradient Descent**. This approach might be interesting, it is simple and very efficient on large data sets (>10^5 examples and/or >10^5 features) [2], but we only had to choose three. Also, even though it can be used for classification, it was a bit hard to find examples (most references refer to using SGD to optimize hyperparameters). 
# 
# 
# 
# #### References
# 
# * [1] Freund, Shapire, "An Introduction to Boosting", Journal of Japanese Society for Artificial Intelligence, 14(5):771-780, September, 1999, https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf
# * [2] https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use
# * [3] https://scikit-learn.org/stable/modules/svm.html#complexity
# * [4] "On the Security of Machine Learning in Malware C&C Detection: A Survey", ACM Computing Surveys, December 2016 Article No.: 59, https://doi.org/10.1145/3003816
# * [ 5] Manning et al, "Introduction to Information Retrieval", Cambridge University Press 2008
# * [6] "Machine learning in automated text categorization", ACM Computing Surveys, March 2002, https://doi.org/10.1145/505282.505283
# * [7] "Predicting Breast Cancer Recurrence Using Machine Learning Techniques: A Systematic Review", ACM Computing Surveys, October 2016, Article No.: 52, https://doi.org/10.1145/2988544
# * [8] "A Survey on Big Multimedia Data Processing and Management in Smart Cities", ACM Computing Surveys, June 2019 Article No.: 54, https://doi.org/10.1145/3323334
# * [9] Olsen et al, "Data-driven advice for applying machine learning to bioinformatics problems", Biocomputing 2018, pp. 192-203, https://doi.org/10.1142/9789813235533_0018
# * [10] https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
# * [11] Joachims, "Text categorization with Support Vector Machines: Learning with many relevant features.". European Conference on Machine Learning (ECML) 1998, pp 137-142, https://doi.org/10.1007%2FBFb0026683
# * [12] https://en.wikipedia.org/wiki/Support-vector_machine#Properties, retrieved on 2020-05-09
# * [13] https://en.wikipedia.org/wiki/Support-vector_machine#Multiclass_SVM, retrieved on 2020-05-09
# * [14] https://scikit-learn.org/stable/modules/svm.html
# * [15] https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#overview, retrieved on 2020-05-09
# * [16] Breiman, "Random Forests", Machine Learning, 45, 5-32, 2001, https://doi.org/10.1023/A:1010933404324
# * [17] Rahman et al, "Functional random forest with applications in dose-response predictions", Scientific Reports 9, Article no. 1628 (2019), https://www.nature.com/articles/s41598-018-38231-w.pdf
# * [18] Henrik Strøm, https://www.quora.com/What-are-the-advantages-and-disadvantages-for-a-random-forest-algorithm, retrieved on 2020-05-09
# * [19] Jansen, "Hands-On Machine Learning for Algorithmic Trading", Dec 2018, ISBN 9781789346411
# * [20] X. Chen and A. L. Yuille, "A time-efficient cascade for real-time object detection: With applications for the visually impaired.", Proceedings of the Computer Society Conference on Computer Vision and Pattern Recognition (2005), 28–28
# * [21] O. Canovas, P. E. Lopez de Teruel, and A. Ruiz, "Detecting indoor/outdoor places using WiFi signals and Ad-aBoost", IEEE Sens. J.17, 5 (Mar. 2017), 1443–1453

# ### Implementation - Creating a Training and Predicting Pipeline
# 
# To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
# In the code block below, you will need to implement the following:
#  - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Fit the learner to the sampled training data and record the training time.
#  - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
#    - Record the total prediction time.
#  - Calculate the accuracy score for both the training subset and testing set.
#  - Calculate the F-score for both the training subset and testing set.
#    - Make sure that you set the `beta` parameter!

# In[69]:


# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    # using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# ### Implementation: Initial Model Evaluation
# In the code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
#   - Use a `'random_state'` for each model you use, if provided.
#   - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Calculate the number of records equal to 1%, 10%, and 100% of the training data.
#   - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.
# 
# **Note:** Depending on which algorithms you chose, the following implementation may take some time to run!

# In[70]:


# TODO: Import the three supervised learning models from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


# TODO: Initialize the three models
clf_A = RandomForestClassifier(random_state = 0)
clf_B = AdaBoostClassifier(random_state = 0)
clf_C = SVC(cache_size = 1000, random_state = 0)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(samples_100/10)
samples_1 = int(samples_100/100)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


# ----
# ## Improving Results
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 

# ### Question 3 - Choosing the Best Model
# 
# * Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000. 
# 
# ** HINT: ** 
# Look at the graph at the bottom left from the cell above(the visualization created by `vs.evaluate(results, accuracy, fscore)`) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:
# * metrics - F score on the testing when 100% of the training data is used, 
# * prediction/training time
# * the algorithm's suitability for the data.

# **Answer:**
# 
# As expected, SVM takes the longest to train, but it's still something one can wait for. For all approaches we have: The larger the training set, the better the result on the test set (but not necessarily on the training set!).
# 
# AdaBoost is the clear winner. Not only is it more accurate (bottom-middle diagram) with a higher F-score (bottom-right diagram) than both Random Forest and SVC, regardless of the training size, it is also the algorithm requiring the least training and prediction time, at least for the larger training sizes (at 10% it is head-to-head with Random Forest, and with the full training set it is the fastest) (bottom-left diagram).

# ### Question 4 - Describing the Model in Layman's Terms
# 
# * In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.
# 
# ** HINT: **
# 
# When explaining your model, if using external resources please include all citations.

# **Answer:**
# 
# The model we are using is _AdaBoost,_ and it is a so-called _Ensemble Method_. This means it combines several basic classifiers (so called _weak learners_ ) such that the combined result is more accurate than each single result (and hence it is called a _strong learner_ ).
# 
# The basic classifier is often a _decision tree_. For example, one might say "Everyone with a certain education level, or above, earns more than 50K". Now one can simply look at every education level in our data set and the one level that creates the most accurate result. And this can be done with several features: "Someone who has a certain education level (or higher) _and_ a certain work experience (or higher) earns more than 50k". So the idea is to take some features and, based on those, find those "boundary values" that classify the data as good as possible. 
# 
# AdaBoost generates a series of such weak learners (here, decision trees), where each new learner takes into account the errors the previous learner made, trying to avoid them. This works by emphasising the errors made by the previous learner, so that the next learner has a higher incentive of avoiding them. Now, the next learner is still just as simple as the previous one: It will not be steadily more accurate, it will just make _other_ errors than the previous learner. The key is that AdaBoost generates many such simple models, and combines them intelligently: One simple way could be majority vote: For example, say we have generated 5 weak learners, now we look at each data point and assign it either ">50k" or "<=50k", depending on what the majority of the 5 weak learners has decided. AdaBoost does not use a majority vote, but goes one step further and weighs the importance of a weak learner in the final result by the _information gain_ reflected by the model: A learner that predicts 50% correct, 50% incorrect is not better than guessing - this learner can be ignored completely. At the other extreme, a model that classifies 100% correct (or 100% incorrect!) is very valuable and is weighted high in the final result.

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Initialize the classifier you've chosen and store it in `clf`.
#  - Set a `random_state` if one is available to the same state you set before.
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
#  - **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!
# - Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
# - Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.
# 
# **Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!

# In[71]:


# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

# TODO: Initialize the classifier
clf = AdaBoostClassifier(random_state = 0)

# TODO: Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {"n_estimators": [10, 100, 200, 400, 600],
              "learning_rate": [.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 10, 20]
             }

# TODO: Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scorer, verbose=2, n_jobs=-1)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# In[72]:


print(grid_fit.best_params_)


# ### Question 5 - Final Model Evaluation
# 
# * What is your optimized model's accuracy and F-score on the testing data? 
# * Are these scores better or worse than the unoptimized model? 
# * How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  
# 
# **Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.

# #### Results:
# 
# |     Metric     | Unoptimized Model | Optimized Model |
# | :------------: | :---------------: | :-------------: | 
# | Accuracy Score |     0.8576        |      0.8660     |
# | F-score        |     0.7246        |      0.7415     |
# 

# **Answer: **
# 
# As can be seen in the table above, both accuracy and F-score were better by around 1% and about 2%, respectively. In my opinion, this is a marginal improvement, I'd hoped for more (but not sure what to expect). In either case, compared to the naive model (accuracy 0.2478, F-score 0.2917), AdaBoost provides a significant improvement.

# ----
# ## Feature Importance
# 
# An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.
# 
# Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.

# ### Question 6 - Feature Relevance Observation
# When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

# **Answer:**
# 
# I would think that the _four_ most relevant are,
# 
# * education level
# * occupation
# * age
# * sex
# 
# (The fifth is explained at the end.)
# 
# **Education level** to me seems obvious since it is often the foundation to higher earnings later on. Notwithstanding the known exceptions, namely those famous billionaires who quit a prestigious school to start a business, education level may be seen as a proxy for intelligence and persistence (and, alas, probably also as a predictor for socio-economic status, which again makes it easier for people to take risks if they're high up on the ladder). Having said here, it is difficult to estimate whether _education level_ or _education num_ is more relevant.
# 
# **Occupation** comes second since some types of work (e.g., "Exec-managerial") simply pay more than others (e.g., "Handlers-cleaners").
# 
# **Age** is relevant since, generally spoken, age equals experience, and the more experience, the more valuable the person in their job.
# 
# **Sex** - well there's the glass ceiling and other (probably undeserved) factors that I'd expect make males earn more than women, on average.
# 
# As to the fifth one? Difficult. I don't expect it to be **marital status**, but rather, alas, **race** or (in the same vein) **native-country**. I think **hours per week** is an obvious predictor, but without further exploratory data analysis, I assume most records report a standard 40h week. I am not sure how to interpret capital-gain or capital-loss. If that is the current wealth, then yes it may be (very) helpful to predict income, but I am not sure.
# 
# My vote for the fifth goes to **race**.

# ### Implementation - Extracting Feature Importance
# Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
# 
# In the code cell below, you will need to implement the following:
#  - Import a supervised learning model from sklearn if it is different from the three used earlier.
#  - Train the supervised model on the entire training set.
#  - Extract the feature importances using `'.feature_importances_'`.

# In[73]:


# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import AdaBoostClassifier

# TODO: Train the supervised model on the training set using .fit(X_train, y_train)
model = AdaBoostClassifier(random_state=0)
model = model.fit(X_train, y_train)

# TODO: Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)


# ### Question 7 - Extracting Feature Importance
# 
# Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.  
# * How do these five features compare to the five features you discussed in **Question 6**?
# * If you were close to the same answer, how does this visualization confirm your thoughts? 
# * If you were not close, why do you think these features are more relevant?

# **Answer:**
# 
# Well interestingly enough, I am not close at all with my prediction. As said, I am not sure how to understand capital gain and capital loss, but apart from a qualitative discussion, what is striking here is that the most important features are all of the continuous nature, as opposed to the one-hot encoded categorical ones.
# 
# And intuitively, that feels right: No single category value is very much decisive, but that's all that's left after one-hot encoding. Inside a decision tree of a given depth, a category preserved as a whole is much more discriminative. I googled and found https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769 which goes in the same direction.
# 
# I am tempted to try again without one-hot encoding. In fact I am wondering whether AdaBoost (or RandomForest) was a good choice given the one-hot encoded data, but in any case I'd have to further research it.

# ### Feature Selection
# How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. 

# In[74]:


# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


# ### Question 8 - Effects of Feature Selection
# 
# * How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
# * If training time was a factor, would you consider using the reduced data as your training set?

# **Answer:**
# 
# The final model on the reduced data produces a slightly worse result (accuracy 0.8427 vs 0.8660, and F-score 0.7048 vs. 0.7415). I think one could make a case to use the reduced data as a training set, but then you have to find the important features somehow before you can use them. And we found them by fitting on the full data, so in a way that seems like a circular argument.
# 
# As said, I'd rather try AdaBoost again without the one-hot encoded data.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
