# Application of Time Perspective in Opioid Addiction Study Using Natural Language Processing

## Summary
### Objective
The objective of this project is to leverage natural language processing techniques and machine learning to extract time perspective from social media posts.

### Technologies Used
* Programming Language: `Python`
* Libraries: `scikit-learn`, `NLTK`, `Gensim`, `Matplotlib`, , `pycorenlp (StanfordCoreNLP)`, `pandas`, `numpy`

## Introduction
Opioid addiction continues to be a contemporary health crisis in the United States. In a 2020 study, an estimated 10.1 million people aged 12 or older misused opioids in the previous year [1]. With the societal stigma against opioid misuse, it is challenging to model the epidemic of opioid addiction. In recent years, however, advancements in machine learning and big data can facilitate the understanding and prediction of opioid use through social media posts. In this project, I used machine learning, specifically natural language processing (NLP), in extracting time perspectives from social media posts. Time perspective (TP) is the manner in which individuals divide the flow of their human experience into different time zones (past, present, future). Social science and psychology studies have shown that a person’s TP has a profound impact on different aspects of their personhood, including their health [2]. Given this context, the use of time perspective in the opioid addiction study can be potentially significant in analyzing the current landscape of opioid use and abuse, which in turn can help inform future public health policies, interventions, and treatment programs. In this paper, I explore how TP can help model the epidemic through natural language processing techniques using Twitter and Reddit social media posts. First, I discuss my problem formulation, followed by an idea of an approach. I will then canvass prior and related work, my methodology and models used (description of work), experimental evaluations, and conclusion and future work.

## II. Proposed Problem Formulation and Data Collection
Given opioid users' social media posts, I aim to identify and characterize each post of its temporal orientation relative to the user’s opioid addiction state. For each post, my goal is to determine its label: past, present, future (for clarity these temporal orientation classes are italicized hereafter).  

The project is based on publicly available data from Twitter and Reddit. For the Twitter data collection, I used tweepy, a Python library for accessing the Twitter API. For the Reddit data, I used PushiftAPI and collected data from two subreddits, r/Opiates and r/OpiatesRecovery. Reddit posts are typically longer than Twitter posts because of Twitter's 280-character limit. To solve this discrepancy, I favored Reddit posts with less than or equal to 280 characters. I amassed a total of 1,010 tweets, 49,452 Reddit posts from r/Opiates, and  13,852 Reddit posts from r/OpiatesReovery. After I independently analyzed and annotated these posts with labels future, past, and present, I have a final dataset of 648 annotated social media posts. Evidently, the project does not have any limitations on collecting public data; the constraints were primarily on the data annotation power of one annotator who is a graduate student with novelty in the subject at hand.

## III. The Idea

My idea has the following steps: (1) Data collection and data annotation. In this first step, I plan to analyze Twitter and Reddit social media posts and label them with their temporal orientation, future, past, and present, relevant to their opioid addiction state. (2) Data preprocessing and feature engineering. Using multiple text pre-processing and analysis libraries, I plan to clean the raw posts with regex and Python’s built-in string library. I will then generate numerical features based on the raw and clean posts using NLTK, TextBlob, StanfordCoreNLP, Scikit-learn, and gensim. (3) Building models. I will use Scikit-learn’s off-the-shelf algorithms Multinomial Naive Bayes, Logistic Regression, Random Forest, and Support Vector Machine to attempt to model the problem. (4) Evaluation. I will try a total of four experiments per algorithm and compare model performances using evaluation metrics precision, recall, f1, and accuracy scores. With the optimal model,  I will perform an error and strength analysis to determine the strengths and weaknesses of the approach.

## IV. Prior and Related Work

Among prior works, I read the papers Identifying and Characterizing Opioid Addiction States Using Social Media Posts (D. Jha, et. al.) and Resolution of grammatical tense into actual time, and its application in Time Perspective study in the tweet space (Sabyasachi Kamila et. al.). In the first paper, the authors extracted Reddit posts to use for identifying the progression of individuals through stages of opioid addiction. This paper advances the state-of-the-art, including the author’s use of word embeddings with bi-directional long short-term memory recurrent neural networks (Bi-LSTM) [2]. In the second paper, Sabyasachi Kamila et. al. collected Twitter data and built a temporal orientation classifier using Bi-LSTM in conjunction with an attention mechanism that can resolve syntactic tense to semantic time [3]. These papers have inspired my work to leverage time perspective in the context of the opioid crisis.

My work attempts to be foundational in the application of time perspective in the study of opioid addiction. Specifically, my work presents findings in the use of traditional machine learning algorithms, and how they perform in identifying temporal orientations given a relatively small dataset. 


## V. Description of Work

My work follows the same steps as my idea: (A) _Data collection and annotation_, (B) _Data preprocessing and feature engineering_, (C) _Building models_, and (D) _Evaluation._

### A. Data Collection and Annotation

As aforementioned, the dataset was collected from Twitter and Reddit using tweepy and PushiftAPI. A total of  64,314 social media posts were collected: 1,010 tweets, 49,452 Reddit posts from r/Opiates, and  13,852 Reddit posts from r/OpiatesReovery. After analyzing thousands of these data points, the final dataset consists of 648 annotated social media posts. 

For the Twitter data collection, I used opioid-related keywords from another study[4].

<table>
  <tr>
   <td><strong>Keyword</strong>
   </td>
   <td><strong>Common Misspellings</strong>
   </td>
  </tr>
  <tr>
   <td>Tramadol
   </td>
   <td>trammadol tramadal tramdol tramadols tramado tramedol tramadoll tramadole tramidol tamadol tranadol tramodol tremadol
   </td>
  </tr>
  <tr>
   <td>Heroin
   </td>
   <td>herione herroine heroins heroine heroin heorin herion
   </td>
  </tr>
  <tr>
   <td>Methadone
   </td>
   <td>methadones methadose methodone mehtadone metadone methadon methdone
   </td>
  </tr>
  <tr>
   <td>Oxycontin
   </td>
   <td>oxicontin oxcotin oycotin oxycotins oycontin oxycontins oxycoton oxicotin ocycotin oxycodin oxycottin oxycotine ocycontin
   </td>
  </tr>
  <tr>
   <td>Codeine
   </td>
   <td>codiene coedine codine codene codein
   </td>
  </tr>
  <tr>
   <td>Dilaudid
   </td>
   <td>delaudid dialudid dilaudad diluadid diaudid dilaudin dilauded dilauid dillaudid
   </td>
  </tr>
</table>


**Table 1: **Sample of opioid-related keywords and their common misspellings used in the Twitter data collection.

Of the final dataset, there are 140 (21.6%) future-oriented posts, 240 (37.0%) past-oriented posts, and 268 (41.4%) present-oriented posts. In Table 3, I present a further breakdown of the dataset across training and test set. 


<table>
  <tr>
   <td><strong>TP Orientation</strong>
   </td>
   <td><strong>Sample Post</strong>
   </td>
  </tr>
  <tr>
   <td>Future
   </td>
   <td>I’m like 20-ish days clean and the Brain zaps finally stopped.. my goal is to not use any hard drugs all of 2021 let’s make it happen 🙏🏽
   </td>
  </tr>
  <tr>
   <td>Past
   </td>
   <td>I was on 170mg and had been for two years. The physical withdrawals lasted a little over a month, the mental stuff went away after about 3.
   </td>
  </tr>
  <tr>
   <td>Present
   </td>
   <td>I am on day 8 as well and the mood swings are insane. I thought I was going crazy.
   </td>
  </tr>
</table>


**Table 2: **Sample of social media posts and their temporal orientation label.

In determining the temporal orientation of the posts, I created and followed these guidelines to determine the TP classification of the post:



1. Label a social media post as _future_ if the user is motivated by the anticipation of tomorrow or the underlying temporal connotation of the tweet refers to the _future _time relevant to the user’s opioid addiction state.
2. Label a social media post as _past_ if the user is motivated by memories of the past or the underlying temporal connotation of the tweet refers to the _past _time relevant to the user’s opioid addiction state.
3. Label a social media post as _present_ if the user is motivated by their current situation or the underlying temporal connotation of the tweet refers to the _past _time relevant to the user’s opioid addiction state.
### B. Data Preprocessing and Feature Engineering

#### 1. Data pre-processing
On each social media post, I performed the following pre-processing tasks: (1) _De-constructing contractions_. The informality of social media posts includes the use of contractions, the combination of two or more words in a shortened form (e.g. _I’ll _is the contraction of _I will_). I manually created a few of the most common contractions as rules with regex. (2) _Performed basic substitutions and cleaning_. This step includes converting _\n _and _\t _to white spaces, lowercasing the text, removing punctuations, stripping the text from extra white spaces, and removing hyperlinks. (3) _Tokenization._ I then separated the words of each post into a list. This step acts as a helper function to step four. (4) _Removing stopwords. _I modified the stopwords of NLTK, removing verbs and phrases that I determined might be useful for prediction (e.g. I removed the word _won’t _in the stopwords because it may add context in determining the TP of the post). After preprocessing, I then split the dataset into training and testing sets. 

<table>
  <tr>
   <td>
   </td>
   <td><strong>Training set</strong>
<p>
<strong>(n=434)</strong>
   </td>
   <td><strong>Testing set</strong>
<p>
<strong>(n = 214)</strong>
   </td>
  </tr>
  <tr>
   <td><strong>TP Orientation</strong>
   </td>
   <td><strong>Count (%)</strong>
   </td>
   <td><strong>Count (%)</strong>
   </td>
  </tr>
  <tr>
   <td>Future
   </td>
   <td>90 (20.74)
   </td>
   <td>50 (23.36)
   </td>
  </tr>
  <tr>
   <td>Past
   </td>
   <td>163 (37.56)
   </td>
   <td>77 (35.98)
   </td>
  </tr>
  <tr>
   <td>Present
   </td>
   <td>181 (41.71)
   </td>
   <td>87 (40.65)
   </td>
  </tr>
</table>


**Table 3:** Breakdown of the composition of the training and test sets. 



#### 2. **Feature Engineering**

For feature engineering, I employed a mixture of text analysis libraries to obtain numerical features and manually crafted features as informed by my prior work. A feature-engineered dataset that includes all features has a total of 367 features.



##### 2.1. Polarity and subjectivity scores

For polarity and subjectivity, I used the uncleaned text as inputs because punctuations, letter cases, and emojis can help in identifying the feature. I used the NLTK sentiment analyzer called Vader. Vader uses a compound score that calculates the sum of all the lexicon ratings normalized between -1 (extremely negative) to 1 (extremely positive) [5]. Vader’s polarity output presents four scores: a negative score (compound score >= 0.05, a neutral score (compound score >-0.05 and compound score &lt; 0.05), a positive score (compound score &lt;= 0.05), and the compound score. For subjectivity, I used TextBlob’s subjectivity method which looks into the intensity of words in the text. Its score ranges from 0 (least subjective) to 1 (most subjective).



##### 2.2. Time expressions count

Time expressions are words or phrases that communicate a period of time, the duration of something that happens, such as _now_, _today_,_ _and _February 28, 2020. _Per my prior work, I determined counting temporal expressions may help in the classification task. I used StanfordCoreNLP’s SUTime, a library for processing temporal expressions. 



##### 2.3. Temporal modifiers count

Per my prior work and data collection, I noticed that in some of the posts that I have read, the temporal modifiers that exist in those posts have helped me identify their temporal orientation. With this in mind, I created a list of temporal adjectives and adverbs and classified them by their temporal type.

<table>
  <tr>
   <td><strong>Temporal Type</strong>
   </td>
   <td><strong>Adverbs and Adjectives</strong>
   </td>
  </tr>
  <tr>
   <td>Futuristic
   </td>
   <td>finally, later, next, soon, yet, forward
   </td>
  </tr>
  <tr>
   <td>Past
   </td>
   <td>then, before, formerly, last, late, previously, past, former
   </td>
  </tr>
  <tr>
   <td>Past-present
   </td>
   <td>yesterday, already, early, earlier, since, earlier
   </td>
  </tr>
  <tr>
   <td>Present
   </td>
   <td>now, today, just, lately, recently
   </td>
  </tr>
  <tr>
   <td>Present-futuristic
   </td>
   <td>tomorrow, tonight, later
   </td>
  </tr>
  <tr>
   <td>Undetermined
   </td>
   <td>daily, fortnightly, hourly, monthly, nightly
   </td>
  </tr>
</table>


**Table 4**: Some of the temporal adverbs and adjectives used in feature engineering.



##### 2.4. Part-of-Speech tag count

In addition to the temporal modifiers count, I incorporated a general part-of-speech tag feature that counts the instances of nouns, verbs, interjections, adjectives, and adverbs (not limited to temporal modifiers). For this task, I used NLTK part-of-speech tagging.



##### 2.5. Word2Vec and TF-IDF

Finally, I used two different word representation techniques to obtain word associations. For Word2Vec, I accounted for bi-grams and tri-grams and I created a word2vec model with 300 dimensions. For TF-IDF, I used Scikit-learn’s TFIDFVectorizer with max_features of 10,000, and an n-gram range of (1,2). I decreased the dimensionality of the overall TFIDF sparse matrix using Scikit-learn’s chi-square feature selection with a p-value limit of 0.80. From 10,000 features, I reduced the TFIDF matrix to 46 most relevant features.

## C. Building Models

For the model creation, I compared and used four traditional statistical algorithms: Multinomial Naive Bayes, Logistic Regression, Random Forest, and Support Vector Machine. For each algorithm, I ran 4 different models: 

* Model 0 uses the following features hereby known as “simple features”: polarity and subjectivity scores, time expression count, temporal modifier count, and part of speech count.
* Model 1 uses all the features: simple features, the Word2Vec matrix, and the TF-IDF matrix.
* Model 2 uses simple features and Word2Vec.
* Model 3 uses simple features and the TF-IDF matrix. 

I performed grid-search cross-validation to obtain the optimal hyperparameters for each model. The following table displays the hyperparameters I used for each algorithm.

<table>
  <tr>
   <td><strong>Algorithm</strong>
   </td>
   <td><strong>Dictionary of Hyperparameters</strong>
   </td>
  </tr>
  <tr>
   <td>Multinomial Naive Bayes (NB)
   </td>
   <td>{
<p>
'alpha' : np.linspace(0.5, 1.5, 6),
<p>
'fit_prior' : [True, False]
<p>
}
   </td>
  </tr>
  <tr>
   <td>Logistic Regression (LR)
   </td>
   <td>{
<p>
'C' : [0.1, 1, 10, 100],
<p>
'solver' : ['lbfgs', 'liblinear'],
<p>
'penalty': ['l1', 'l2'],
<p>
'max_iter': [200,400,600]
<p>
}
   </td>
  </tr>
  <tr>
   <td>Random Forest (RF)
   </td>
   <td>{
<p>
'n_estimators' : [100,200,500,1000],
<p>
'min_samples_leaf':[1, 2, 5, 10,20],
<p>
'max_features' : [int(0.5*np.sqrt(len_features)), int(np.sqrt(len_features)), int(2*np.sqrt(len_features))]
<p>
}
   </td>
  </tr>
  <tr>
   <td>Support Vector Machine (SVM)
   </td>
   <td>{
<p>
'kernel': ['rbf', 'linear'],
<p>
'gamma': [1e-3, 1e-4],
<p>
'C': [1, 10, 100, 1000]
<p>
}
   </td>
  </tr>
</table>


**Table 5**: Algorithm and hyperparameters used in GridSearchCV. 



### D. Experimental Evaluations

#### 1. **Results of the GridSearch Cross Validation: Optimal Hyperparameters**

<table>
  <tr>
   <td colspan="2" >
<strong>Model 0: (Simple Features) Polarity + Subjectivity Score, Count of Temporal Expressions, Count of Temporal Modifiers, POS Tags</strong>
   </td>
  </tr>
  <tr>
   <td>NB
   </td>
   <td>{'alpha': 0.7, 'fit_prior': False}
   </td>
  </tr>
  <tr>
   <td>LR
   </td>
   <td>{'C': 1,
<p>
'max_iter': 200,
<p>
'penalty': 'l1',
<p>
'solver': 'liblinear'}
   </td>
  </tr>
  <tr>
   <td>RF
   </td>
   <td>{'max_features': 2,
<p>
'min_samples_leaf': 2,
<p>
'n_estimators': 500}
   </td>
  </tr>
  <tr>
   <td>SVM
   </td>
   <td>{'C': 1000, 'gamma': 0.001, 'kernel': 'linear'}
   </td>
  </tr>
  <tr>
   <td colspan="2" ><strong>Model 1: (All features) simple_features + tfidf + w2v</strong>
   </td>
  </tr>
  <tr>
   <td>NB
   </td>
   <td>{'alpha': 1.1, 'fit_prior': False}
   </td>
  </tr>
  <tr>
   <td>LR
   </td>
   <td>{'C': 1,
<p>
'max_iter': 200,
<p>
'penalty': 'l2',
<p>
'solver': 'liblinear'}
   </td>
  </tr>
  <tr>
   <td>RF
   </td>
   <td>{'max_features': 8,
<p>
'min_samples_leaf': 1,
<p>
'n_estimators': 100}
   </td>
  </tr>
  <tr>
   <td>SVM
   </td>
   <td>{'C': 1, 'gamma': 0.001, 'kernel': 'linear'}
   </td>
  </tr>
  <tr>
   <td colspan="2" ><strong>Model 2: Simple features + w2v</strong>
   </td>
  </tr>
  <tr>
   <td>NB
   </td>
   <td>{'alpha': 1.5, 'fit_prior': True}
   </td>
  </tr>
  <tr>
   <td>LR
   </td>
   <td>{'C': 1,
<p>
'max_iter': 400,
<p>
'penalty': 'l2',
<p>
'solver': 'lbfgs'}
   </td>
  </tr>
  <tr>
   <td>RF
   </td>
   <td>{'max_features': 8,
<p>
'min_samples_leaf': 2,
<p>
'n_estimators': 200}
   </td>
  </tr>
  <tr>
   <td>SVM
   </td>
   <td>{'C': 1, 'gamma': 0.001, 'kernel': 'linear'}
   </td>
  </tr>
  <tr>
   <td colspan="2" ><strong>Model 3: Simple features + tfidf</strong>
   </td>
  </tr>
  <tr>
   <td>NB
   </td>
   <td>{'alpha': 0.5, 'fit_prior': False}
   </td>
  </tr>
  <tr>
   <td>LR
   </td>
   <td>{'C': 100,
<p>
'max_iter': 200,
<p>
'penalty': 'l2',
<p>
'solver': 'liblinear'}
   </td>
  </tr>
  <tr>
   <td>RF
   </td>
   <td>{'max_features': 4,
<p>
'min_samples_leaf': 1,
<p>
'n_estimators': 500}
   </td>
  </tr>
  <tr>
   <td>SVM
   </td>
   <td>{'C': 10, 'gamma': 0.001, 'kernel': 'linear'}
   </td>
  </tr>
</table>


**Table 6**: Optimal hyperparameters determined by GridSearchCV across algorithms and across models. 



#### 2. **Model and Algorithm Performance Comparison**

<table>
  <tr>
   <td>
<strong>Algorithm</strong>
   </td>
   <td><strong>Model</strong>
   </td>
   <td><strong>Precision</strong>
   </td>
   <td><strong>Recall</strong>
   </td>
   <td><strong>F1</strong>
   </td>
   <td><strong>Accuracy</strong>
   </td>
  </tr>
  <tr>
   <td>Random Forest
   </td>
   <td>0
   </td>
   <td>0.54
   </td>
   <td>0.46
   </td>
   <td>0.44
   </td>
   <td>0.51
   </td>
  </tr>
  <tr>
   <td>Random Forest
   </td>
   <td>1
   </td>
   <td>0.59
   </td>
   <td>0.56
   </td>
   <td>0.55
   </td>
   <td>0.56
   </td>
  </tr>
  <tr>
   <td>Random Forest
   </td>
   <td>2
   </td>
   <td>0.53
   </td>
   <td>0.51
   </td>
   <td>0.49
   </td>
   <td>0.51
   </td>
  </tr>
  <tr>
   <td>Random Forest
   </td>
   <td>3
   </td>
   <td>0.53
   </td>
   <td>0.52
   </td>
   <td>0.50
   </td>
   <td>0.52
   </td>
  </tr>
  <tr>
   <td>Logistic Regression
   </td>
   <td>0
   </td>
   <td>0.49
   </td>
   <td>0.46
   </td>
   <td>0.43
   </td>
   <td>0.51
   </td>
  </tr>
  <tr>
   <td>Logistic Regression
   </td>
   <td>1
   </td>
   <td>0.54
   </td>
   <td>0.54
   </td>
   <td>0.54
   </td>
   <td>0.54
   </td>
  </tr>
  <tr>
   <td>Logistic Regression
   </td>
   <td>2
   </td>
   <td>0.51
   </td>
   <td>0.52
   </td>
   <td>0.51
   </td>
   <td>0.52
   </td>
  </tr>
  <tr>
   <td>Logistic Regression
   </td>
   <td>3
   </td>
   <td>0.54
   </td>
   <td>0.53
   </td>
   <td>0.53
   </td>
   <td>0.53
   </td>
  </tr>
  <tr>
   <td>Naive Bayes
   </td>
   <td>0
   </td>
   <td>0.45
   </td>
   <td>0.45
   </td>
   <td>0.44
   </td>
   <td>0.50
   </td>
  </tr>
  <tr>
   <td>Naive Bayes
   </td>
   <td><strong>1</strong>
   </td>
   <td><strong>0.59</strong>
   </td>
   <td><strong>0.59</strong>
   </td>
   <td><strong>0.59</strong>
   </td>
   <td><strong>0.59</strong>
   </td>
  </tr>
  <tr>
   <td>Naive Bayes
   </td>
   <td>2
   </td>
   <td>0.53
   </td>
   <td>0.54
   </td>
   <td>0.52
   </td>
   <td>0.54
   </td>
  </tr>
  <tr>
   <td>Naive Bayes
   </td>
   <td>3
   </td>
   <td>0.58
   </td>
   <td>0.57
   </td>
   <td>0.57
   </td>
   <td>0.57
   </td>
  </tr>
  <tr>
   <td>SVM
   </td>
   <td>0
   </td>
   <td>0.45
   </td>
   <td>0.47
   </td>
   <td>0.43
   </td>
   <td>0.53
   </td>
  </tr>
  <tr>
   <td>SVM
   </td>
   <td>1
   </td>
   <td>0.53
   </td>
   <td>0.53
   </td>
   <td>0.53
   </td>
   <td>0.53
   </td>
  </tr>
  <tr>
   <td>SVM
   </td>
   <td>2
   </td>
   <td>0.53
   </td>
   <td>0.54
   </td>
   <td>0.53
   </td>
   <td>0.54
   </td>
  </tr>
  <tr>
   <td>SVM
   </td>
   <td>3
   </td>
   <td>0.54
   </td>
   <td>0.53
   </td>
   <td>0.53
   </td>
   <td>0.53
   </td>
  </tr>
</table>


**Table 7**: Comparison of the different algorithms across different models. The optimal model is Naive Bayes in conjunction with Model 2. 

Based on Table 7, it is clear that Model 1 Naive Bayes outperforms the other models and algorithms, with Model 3 Naive Bayes as the second-best performing, and Model 1 Random Forest as the third-best performing. Unfortunately, the results, including the scores of the optimal model, are poor, and I would not recommend the use of any of these models for eliciting a data point for decision-making. However, the comparisons of the models show a promising sign that further work with the models, including the use of more data points, the use of different algorithms such as Bi-LSTM (which is designed for sequential data like text), and a thorough review of the data collection and annotation process, may improve the performance of the models. With this in mind, I performed an error analysis that may inform future work on the models.

#### 3. **Error Analysis**

I further analyzed the optimal model’s (Naive Bayes Model 1) strengths and weaknesses using confusion matrices. Because there are three classes, there are 9 different classification states that a data point can be identified. I assign the following nomenclature to these states.

* TP0: correctly identified _future _posts
* TP1: correctly identified _past_ posts
* TP2: correctly identified _present _posts
* E01: _future_ posts misclassified as _past _posts
* E02: _future _posts misclassified as _present _posts
* E10: _past _posts misclassified as _future _posts
* E12: _past _posts misclassified as _present _posts
* E20: _present_ posts misclassified as _future _posts
* E21: _present _posts misclassified as _past _posts

See report for the error analyses.

##  VI. Conclusion

To conclude, while the performances of the models were poor and are not recommendable for use in any system, the comparison of the different algorithms using different models (features) shows a sign of a gradual increase in performance when all features are used with Naive Bayes. Because of the class independence assumption of Naive Bayes, it performs well with relatively small datasets as it can quickly learn high-dimensional features. I then performed an error analysis on the optimal model (Naive Bayes Model 1) to see which data points the model struggled with the least and most. It was expected that the model will perform well in identifying _present_ posts compared to _future_ posts because there were more _present _than _future _posts in the training data. Based on the error analysis performed, most of the errors may stem from two aspects: the multi-sentence structure of the posts and the presence of verb tenses that may be associated with _past_, _present_, and _future_. Multi-sentences can complicate the identification of the post’s TP as there are more words to account for, including a mixture of different tenses that can further complicate the prediction. For future work, I aim to use more data points as I believe this may be one of the limitations of the current approach. With more data points, I can leverage deep learning techniques like Bi-LSTM, which has context management for long-range dependencies of sequential data. Furthermore, further exploration of the error analysis, exploratory data analysis, and a review of the data annotation and collection may help the performance of the current approach.



## VII. References
1. Substance Abuse and Mental Health Services Administration. (2020). Key substance use and mental health indicators in the United States: Results from the 2019 National Survey on Drug Use and Health. (HHS Publication No. PEP20-07-01-001(NSDUH Series H-55). https://www.samhsa.gov/data/
2. D. Jha, S. R. La Marca and R. Singh, "Identifying and Characterizing Opioid Addiction States Using Social Media Posts," 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2021, pp. 913-918, doi: 10.1109/BIBM52615.2021.9669628.
3. Kamila, S., Hasanuzzaman, M., Ekbal, A., & Bhattacharyya, P. (2019). Resolution of grammatical tense into actual time, and its application in Time Perspective study in the tweet space. _PLOS ONE_, _14_(2), e0211872. https://doi.org/10.1371/journal.pone.0211872
4. Sarker, A., Gonzalez-Hernandez, G., & Perrone, J. (2019). Towards automating location-specific opioid toxic surveillance from Twitter via data science methods. _Studies in health technology and informatics_, _264_, 333. [https://doi.org/10.3233/SHTI190238](https://doi.org/10.3233/SHTI190238)
5. Geeks For Geeks. (2021, October 7). Python | Sentiment Analysis using VADER. GeeksforGeeks. Retrieved December 17, 2022, from [https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/](https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/)