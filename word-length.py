# -------------------------------------------------------
# Assignment 2
# Written by Joshua Parial-Bolusan (40063663) Jeffrey Lam(40090989)
# For COMP 472 Section AA – Summer 2021
# --------------------------------------------------------

from classifier import * 
import matplotlib.pyplot as plt
import numpy as np
from service.imdb_service import ImdbService

#Scrape and store information
def length():

    imdb = ImdbService.from_web()

    model = Classifier(imdb.reviews_df)

    length_values = np.array([2, 4, 9])
    accuracies = []
    words_left = []

    indexes_to_drop=[]
    train_model, pos_total, neg_total, test_set = model.build_vocabulary()

    #print(train_model)
    for length in length_values:
        #print(length)
        print(train_model.shape[0])
        train_model = model.modify_length(train_model, length)

        results = model.evaluate(train_model, test_set, pos_total, neg_total)

        correct_results = len(results[results["prediction"] == True])
        accuracy = (correct_results / len(results["prediction"]))*100
        print(f"length: {length}\tAccuracy: {accuracy}")
        accuracies.append(accuracy)
        words_left.append(train_model.shape[0])

        model.model_to_file(train_model, "length-model.txt")
        model.results_to_file(results, "length-result.txt")

    plt.title("Length Classifier Performance")
    plt.xlabel("Number of Words Left")
    plt.ylabel("Correctness")
    plt.plot(words_left, accuracies)
    plt.show()


length()