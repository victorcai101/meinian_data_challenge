# Meinian Data Challenge

We participated in the Meinian health data challenge hosted by [Tianchi](https://tianchi.aliyun.com/competition/introduction.htm?raceId=231654). This challenge is about predicting the risk of hypertension and hyperlipidemia, which is the cause of many diseases including stroke and coronary heart disease. Meinian is one of China's largest physical examination and medical services companies. With enough physical examination data, this challenge intends to see if other indicators can help us make predictions on hypertension and hyperlipidemia. 

We (Wei, Jiarui, Zhengyang, and me) participated in the first round of the competition, ranking [21st](https://tianchi.aliyun.com/competition/rankingList.htm?spm=5176.11165320.0.0.3e4a2df51As3pH&season=0&raceId=231654&pageIndex=2) over 3152 teams (top 1%). 

## Preprocess of the Data

To alleviate data sparsity, we ditched the features with more than 95% values missing. For string typed features, if there are limited numbers of possible values or the most frequent values, we set them as categorical features; and if the texts contain many numbers, we call them quasi-numerical features; and we call all the rest long-text features. 

**Quasi-numerical features**: to extract numbers out from string features, we also need regular expression and case-by-case examination. There are multiple reasons for numerial values. For example, false use of fullwidth numbers, containing descriptive text like "greater than", multiple numbers joined with space, numbers coming with units, etc. 

**Categorical features**: the feature extraction of such features is basically based on regular expression, as there are not many variations of the values. For the infrequent and incomprehensible values, we set them as `nan`. We started out to one-hot code the categories by its frequency, but turned out using the same techniques as long-text features as it is simpler and performing pretty well. 

**Long-text features**: for strings that are relatively long and contain multiple information, we simply call it long-text features and do not make any operations. 

**Segmentation**: we segmentate the strings with [jieba](https://github.com/fxsjy/jieba), a Chinese text segmentation module for Python. We also wrote an user dictionary for common medical terms in the data. **Vectorization**: td-idf is the method adopted. **Clustering**: to reduce the dimension of the vectorized text, we used  k-means clustering. 

##  Text Feature Extraction

The text features are mainly descriptions written by physicians about certain examinations on patients. For one group of examinations, there are certain patterns in the description which can be extracted as categorical features as well as numerical features, but there may exist some scenarios where categorical features and numerical features could not capture the proper meaning. As as result, we still need a way to represent the whole text.

**Vectorization**: word embedding is a common way to represent the unfixed-length sentences using fixed-length vectors. The most prevailing method is to use Word2Vec to represent each word in the sentence as a vector under the entire corpus and using the average of those vectors to represent the sentence. Instead of using Word2Vec which gives all the words in a sentence the same weight, we use TF-IDF, a method that represent the sentence with the words' importance by using the multiplication of the words' term frequency and inverse document frequency. For each set of examination descriptions, we choose the TF-IDF values of the 50 words with the highest term frequency as the vector.

**Vector Clustering**:  directly using the vector of 50 dimensions would lead to dimensional curse, since we have 162 set of descriptions. Instead, we use k-means clustering to reduce the dimension. After careful selection with 3-fold cross validation, we choose the cluster numbers as 5 and 10 for short text and long text respectively. Now we finish the text feature extraction process.

After adding the categorical feature for those text descriptions, our grade boost from 0.06216 to 0.2904.



## Model

For this regression problem, we choose to use LGBM. One great advantage for this model is that it can handle categorical data directly without the dummy process, which greatly reduce the feature dimension. Although we trained 5 models for the 5 different target respectively, we shared the same fine-tuned hyperparameters by minimizing the average logged mean-square-error of the five targets.

After getting the optimized hyperparameters, we further boosted our result by Bagging(Bootstrap Aggregating ). We randomly generate 100 seeds to use in train_test_split process and get 100 different results. We use  the average of the 100 results as our final results, getting our final grade of 0.2819.
