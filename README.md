## Topics to explore

- Embeddings creation techniques (transfer Learning) like ELMo, ULMFiT, BERT
- Sequential modelling (CRF, LSTM etc)
- Word embeddings (Glove, Word2vec)
- look-alike modelling
- Optimization Softwares Gurobi or CPLEX
- Optimization Solutions using Linear or Mixed integer programming





# My Notes
My notes on datascience.

https://towardsdatascience.com/20-machine-learning-interview-practice-problems-3c86a572eeec


R-squared explains to what extent the variance of one variable explains the variance of the second variable. So, if the R2 of a model is 0.50, then approximately half of the observed variation can be explained by the model's inputs.


heteroskedasticity (or heteroscedasticity) happens when the standard errors of a variable, monitored over a specific amount of time, are non-constant


A1. Differentiate between univariate, bivariate and multivariate analysis. 

Univariate analysis is used where the data contains only one variable, bivariate analysis when there are two variables, and multivariate analysis is implemented when there are more than two variables. b) The purpose of univariate analysis is to describe the data and discover patterns. While bivariate analysis discovers the relationship or correlation between the two variables. The multivariate analysis technique is used when you need to predict a certain outcome with a given set of multiple dependent variables

A2. What is p-value?

P-value is the probability value of the occurrence of a given event, measured by assigning number between 0 and 1.

When performing a statistical hypothesis or null hypothesis (H0) of a study, the p-value helps to determine the strength of the results. The null hypothesis is the inference about a population of statistics. Generally, the p-value of 0.05 is used as a threshold. A low p-value (< 0.05) indicates strength against the null hypothesis, which means the null hypothesis can be rejected, and the alternative hypothesis accepted. A high p-value (> 0.05) indicates the evidence against the null hypothesis is not strong enough, and the null hypothesis cannot be rejected.

A4. When is resampling done?

When it is required to interface two systems that have different sampling rates; When you need to test the models under small samples; When drawing randomly with replacement from a set of data points; When performing significance tests or exact tests by substituting labels on data
points;

To validate models by using random subsets. A5. What are the types of biases that can occur during sampling?

Selection bias.. Under coverage bias. Survivorship bias. Explain selection bias.

Selection bias occurs when there is a bias in sample selection. The sample is not representative of the population under analysis, as proper randomisation was not achieved during sample selection. An error is introduced due to the non-random population sample, creating a distortion in statistical analysis.

A6. What is logistic regression? When is it used? Give an example, when you have

used logistic regression.

Logistic regression is a statistical method for analysing a dataset in which one or more independent variables determine an outcome.

It is used in predictive algorithms, when you need to predict a binary outcome of a linear combination of predictor or independent variables.

Example. To predict whether a particular customer will buy an online product or not, the outcome of the prediction is binary (yes/no), and the predictor variables are the previous spend data of the customer, buying or browsing behaviour, cart abandonment rate, etc.

A7. What methods do you use to identify outliers within a data set? What call do you

take when outliers are identified?

The answer demonstrates your analytical skills. So explain the process briefly with examples, to display your understanding.

Use visualisation techniques like histogram and scatter plot for univariate or bivariate data Use simple statistical techniques, where the process includes -> sorting the data -> calculating the medians of the first half (Q1) and second half (Q3) of the data -> deriving the difference between the interquartile range (Q3 – Q1) -> identifying outliers by the position of data point.

A data point that falls outside the inner fence is a minor outlier, and a data point that falls outside the outer fence is a major outlier.

Once the outliers are identified, they have to be examined in the context of the nature of the data set, data validation protocols, and the behaviour of the variable being analysed. If the outlier is uncharacteristic but correct, like a large cash withdrawal, it is left untouched. However, if the outlier is unrealistic like a peak city temperature of 250 degrees, it is incorrect and has to be dealt with. When feasible, a resurvey is done for fresh data collection, or otherwise if not feasible, the data point is deleted.

A8. What is the goal of A/B Testing?

A/B Testing is a statistical hypothesis used when there are two variables, A and B. The objective of A/B testing is to generate insights by testing both variables A against B, to discover which performs better. A/B testing is done in testing two versions of a web page, and to detect what changes in each web page can maximise or increase an outcome, like better conversion rate for a page.

B. DATA ANALYTICS & MANAGEMENT

B1. What is root cause analysis?

As the name suggests, root cause analysis is a problem-solving technique used to identify the root causes of faults or problems. It adopts a structured approach to eliminate the root causes of an incident so that the most effective solutions can be identified and implemented.

B2. Explain the steps in making a decision tree.

Identify the decision to be made i.e. the problem to be solved or the question to be answered. Gather relevant information. Take the entire data set as input (root node). Look for a split that separates the dataset. Apply the split to the input data. Re-apply steps c) and d) to the divided data. Add more branches and leaves. (Branches connect to leaf nodes, containing questions or criteria to be answered). Stop when a stopping criteria are met. Clean up the tree if there are too many splits. This pruning (opposite of splitting) removes sections of the tree that add little value to the classification and improves predictive accuracy by reduction of overfitting. Verify accuracy. B3.What is data preparation? Data preparation is the process of making sure that the data used meets the needs of the analysis, is of high quality, precise, describable, and in a format that can be used by the data scientist.

B4. When must an algorithm be updated?

In instances, when

The underlying data source is changing, You want the model to evolve as data streams through the infrastructure, There is a case of non-stationarity, where the statistical properties like the mean, variance or autocorrelation are changing. B5. How does data cleaning play a vital role in the analysis? Data is often collected from multiple sources and is always in a raw format. Datasets come in various sizes and are different in nature.

Data cleaning is essential when data comes from heterogeneous sources, as the time taken to clean the data increases exponentially with an increase in the number of disparate data sources, dataset sizes and the volume of data generated. It helps to transform and refine data into a usable format, that data scientists can work with. Cleaning up data removes errors and inconsistencies, and improves the quality of data for robust analysis. For instance, removal of duplicate or invalid entries, refines the analysis. Data cleaning reduces analysis time by nearly 80%. It improves the accuracy and performance of the model, as biased information can alter business decisions. C. MACHINE LEARNING (ML)

C1. What is the difference between supervised and unsupervised machine learning? Supervised machine learning uses a full set of labelled data, i.e. data is tagged with the answer the algorithm should come up with on its own. Unsupervised machine learning doesn’t required labelled data. In supervised machine learning, the machine learning task is trained for every input with a corresponding target or response variable. In unsupervised machine learning the machine learning task is trained only with a set of inputs with no target variable, or specific desired outcome. Supervised learning is useful in classification and regression problems. Unsupervised learning is used in problems of clustering, anomaly detection, associations, and in autoencoders. C2. What is ‘Naive’ in a Naive Bayes?

Naive means the algorithm used to classify objects is ‘naive’ or uniformed, as it makes assumptions that may or may not be correct.

C3. Explain Decision Tree algorithm in detail.Decision tree is a supervised machine learning algorithm chiefly used for regression and classification. The dataset is continually split up into smaller subsets of similar value, to develop a decision tree incrementally. The result is a decision tree where each node represents a feature (attribute), each branch represents a decision (rule) and each leaf represents an outcome (categorical or continuous value).

C4. What do you understand by the term recommender systems? Where are they used?

Recommender systems are a kind of information filtering systems, to predict ratings or preferences based on content and collaboration.

Recommender systems are commonly used in ecommerce, movie review platforms, music downloads, dedicated apps, news curation, and so on.

C5. What are the different types of collaborative filtering, and what are the common methods used?

Memory based approach uses the entire database for prediction. Common methods are classification, neighbourhood and item-to-item. Model based approach develops models using various data mining and machine learning algorithms to predict users’ rating. Common algorithms are Bayesian networks, clustering models, latent semantic models such as singular value decomposition (SVD), probabilistic latent semantic analysis, and Markov decision process based models. Hybrid approach combines the memory-based and the model-based algorithms, to overcome limitations like sparsity and loss of information, as well as to improve predictability performance. D. DEEP LEARNING

D1. When does a neural network model become a deep learning model?

When you add more hidden layers and increase depth of neural network.

D2. In a neural network, what steps can prevent overfitting?

Adding more data, using Data Augmentation, Batch Normalisation, Reducing architecture complexity, Regularisation, and Dropout.

D3. For an image recognition problem (like recognising a human in an image), which

architecture of neural network is best suited to solve the problem?

The Convolutional Neural Network is best suited for image related problems because of its inbuilt nature of factoring changes in nearby locations of an image.

D4. Which gradient technique works better when the data is too big to handle in RAM

simultaneously?

Stochastic Gradient Descent, to get the best possible neural network.

D5. Suppose the problem you are trying to solve has a small amount of data. You

have a pre-trained neural network used on a similar problem. Which method would

you choose to make use of this pre-trained network, and why?

The answer demonstrates your problem-solving skills. So explain the process briefly with examples, to display your understanding.

If the data is mostly similar, the best method would be to freeze all the layers and re-train only the last layer; because the previous layers work as feature extractors.

E. TOOL / LANGUAGE

It is not all about mentioning the projects you have worked on, or tools used. At a data science job interview, you will be assessed on your understanding of why you chose an algorithm or a technique, and why you reached the conclusion. Generally, a sound, hands-on knowledge of Python, R and SQL are considered must-haves. So you can expect to be grilled on the same before you are tested for other tools and languages mentioned on your resume! E1. Give examples of aggregate functions in SQL.

COUNT() function returns the number of rows that match a specified criteria. AVG() function returns the average value of a numeric column. SUM() function returns the total sum of a numeric column. MIN() function returns the smallest value in the table. MAX() function returns the largest value in the table. DISTINCT function returns distinct or different values, allowing you to omit duplicates. E2. Consider you have a column ‘A’ in table1 with three values (1,2,3). This is a

primary key and is referenced to column ‘B’ in table2. How to insert more values

without getting an error?

Any value except duplicate values can be inserted in column A of table 1. However, because of foreign key integrity (column B in table2 referenced by the column A), values other than 1, 2 and 3 cannot be inserted in column B.

E3. Python or R – Which would you prefer for text analytics? The answer demonstrates your understanding of the two programming languages and their application in real-world scenarios. So explain the reason why you opt for one vis-a-vis the other. Better still, if you can demonstrate your knowledge with examples.

Python would be preferred because:

It performs faster for all types of text analytics. Can be used further for data manipulation and repetitive tasks, like say, social engineering techniques. It has the Pandas library that provides easy-to-use data structures and high-performing analysis tools. It has many NLP libraries and other dedicated packages like Gensim for Topic Analysis, It can also be used to explore Deep Networks using Long Short Term Memory (LSTM) for more refined results from a vast dataset. E4. What are negative indexes and why are they used?The sequences in Python are indexed, i.e. in an ordered list with both positive and negative numbers. Positive numbers use ‘0’ as first index, ‘1’ as second index and so on. The index for the negative number, however starts from ‘-1’ and continuing below. Negative indexes use counting from the last element in the list or the penultimate element, so you count from the right instead of the left.

Negative indexes are used to

Remove any new-line spaces from the string, and allow the string to except the last character that is shown as S[:-1]. Show the index to represent the string in the correct order. Reduce the time spent, in writing, to access the last item of the list. F. GUESS-ESTIMATES /GUESSTIMATES

A guesstimate is a portmanteau of guess and estimate, used to make a rough approximation pending a more accurate estimate, or just an educated guess. Guess questions are common in interviews for data science roles. You are judged on how structured your approach is, how good you are with numbers and mental calculations, and if you are able to quickly analyse using different methods.

EXAMPLE 1. Guess-estimate the quantity of cheese consumed in Bengaluru in a day.

The guess-estimate considers either the consumption side (end customer, grams consumed per person, types of cheese available in the market, etc.) or the production side (brands in India, quantity sold, etc.).

Let’s say, you are considering the consumption side as the approach:

Break down the end consumers into age groups, consumer types (regulars/occasional), consumers with diet/medical restrictions (diabetes, heart problems, high cholesterol, etc.), and so on.

Work out the population of Bengaluru in this age group, the statistics related to users and diseases, and other variables. Assign percentages and appropriate weightage to make a guess-estimate!

EXAMPLE 2. How many red coloured Swift cars are there in Delhi?

Consider Swift as a youth brand, and red Swift as a sporty vehicle more than a family car. Going by this logic, consider the numbers in the age group 25–40 as buyers of a red Swift. Further, assuming that Swift has an approx. 10% market share in the car segment, and assuming than 5% of the people in Delhi in the age group 25-40 can afford a car, you can get your values. At the end, think of how many Swift red cars you see on the road (one of every six?), and derive your final numbers of red Swifts in Delhi!

EXAMPLE 3. Guess-estimate the number of people in India that would use a cricket kit of an elite brand [Fractal Analytics]

So think aloud, as you connect numbers and links, and work your way through the puzzle before you find your Eureka moment!

Jagdish Chaturvedi, Director, Clinical Innovations at InnAccel, however, has a unique but perhaps useful take if you want to be thinking on your feet. According to his comment on Quora, “there are very few lateral thinkers and everyone wants to hire them. The reason for these questions often is to elicit some lateral thinking and not meticulous and detailed logic.” So he says, you have the option of an “Akbar-Birbal answer” if you find yourself in a tight spot. Ultimately hirers want fast and smart thinkers!

G. CASE STUDIES

The answers demonstrate your analytical skills. So walk through your reasoning. Begin with understanding what the company does. What is the business problem? Why are they applying a certain technique? These questions help you find solutions to the problem being considered.

EXAMPLE1. Optimise pricing for an e-commerce product, where variables are

Market Price/Unit Cost/Unit Profit/Unit Average Number of Units sold Increase in total customer response rate for every 10% drop in unit price Increase in sales volume with every 10% drop in unit price. A price hike of up to 20% is allowed.

EXAMPLE2. Route optimisation for a school bus with given two alternate routes (X, Y) where

Average permitted speed is 25 km/hr The two routes are of length 5km (X) and 7.8 km (Y) Traffic congestion on route X, which also has a traffic signal configured for 120 seconds time. EXAMPLE3. How would you investigate a drop in user engagement?
