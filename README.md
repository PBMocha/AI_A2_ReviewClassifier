# Review Rating Classifier

Classifies the user reviews of a show on Imdb. The project scrapes user reviews of a particular show on Imdb and utilizes a Naive bayes Classifier implementation to determine if a specific review is positive or negative. 

### Libraries Used

- Matplotlib
- Pandas
- Numpy


### Classes and Functions

    ImdbService(): A class that is responsible for scraping and storing the show's information and user reviews into dataframes.

    .from_csv(): get review data from csv file

    .from_web(): get review data from website (IMDB)


    Classifier(reviews_df): responsible for building the vocabulary and evaluating the model. Takes a dataframe of reviews as input

    .build_vocabulary(smooth): builds the vocabulary and partitions model into training and test sets. Smooth value is used to modify the smoothing value when build_vocabulary calculates probabilities. Returns the training set (vocab), test set and the frequency of words in positive and negative reviews

    .evaluate(train, test): evaluates training set against the test set. Returns a pandas dataframe that contains prediction results 

### Running the program

The project is split into different tasks

    Run 'python main.py' to run the standard classifier

    Run 'python smooth.py' to execute task 2.1, to analysis the accuracy and smoothing relationship

    Run 'python word-length.py' to execute task 2.3, to analyse the word length and accuracy relationship