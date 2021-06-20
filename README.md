# Review Rating Classifier

Classifies the user reviews of a show on Imdb. The project scrapes user reviews of a particular show on Imdb and utilizes a Naive bayes Classifier implementation to determine if a specific review is positive or negative. 

### Libraries Used

- Matplotlib
- Pandas
- Numpy


### Classes and Functions

    ImdbService(): A class that is responsible for scraping and storing the show's information and user reviews into dataframes

    Classifier(): responsible for building the vocabulary and evaluating the model

    .build_vocabulary(smooth): builds the vocabulary and partitions model into training and test sets. Smooth value is used to modify the smoothing value when build_vocabulary calculates probabilities.

    .evaluate(train, test): evaluates training set against the test set. Returns a pandas dataframe that contains prediction results 

### Running the program

Split into different tasks