

<!-- PROJECT LOGO -->
  <h2 align="center">IMDB Reviews Classification</h2>

  <p align="center">


<!-- ABOUT THE PROJECT -->
### About The Project
  
This script is for IMDB Reviews Classification using Bag of Words (BoW) and Random Forest methods.

The dataset is available at Kaggle and contains 50.000 reviews.



### Getting Started

I am using Anaconda (Python 3.8) and the following packages:
- pandas
- numpy
- matplotlib
- scikit learn
- nltk



### Code Steps
#### Preprocessing

During the preprocessing stage the database was prepared to be used in model training. 
* special characters and links have been removed;
* words have been turned into tokens;
* words with little meaning for understanding the text, called stopwords, have been removed;
* the stemming process was applied, using the Porter Stemming algorithm, to reduce the inflected words to a root.

#### Bag of Words (BoW) - TF-IDF

The text data was represented using the bag-of-words model. This way, the presence or absence of each word will be used as a feature for the classification model. 
The TF-IDF is used to reflect the importance of each word based on its frequency in all reviews. This importance is a number between 0 and 1, this number being inversely proportional to the frequency, since the higher its frequency, the less important it will be, as it has less relevance in differentiating classifications.

#### Model Training

The classification was trained using the Random Forest model. This model consists of a group of Decision Trees, where each node is a word and the final result for a classification is the most frequent among the results of each Decision Tree.

#### Model Training

The results were analyzed usin the confusion matrix, accuracy, AUC score and ROC Curve. From these parameters it is possible to understand the relationship between true positives, false positives, true negatives and false negatives.




### Conclusions

The results of this modeling demonstrate that it is a reliable way to classify texts, which can be used for a closer relationship with customers, allowing companies' strategies to be directed from the customers' opinion in an efficient way.




<!-- CONTRIBUTING -->
### Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.




<!-- LICENSE -->
### License

Distributed under the MIT License. See `LICENSE` for more information.

