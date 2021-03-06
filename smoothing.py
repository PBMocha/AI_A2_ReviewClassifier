# -------------------------------------------------------
# Assignment 2
# Written by Joshua Parial-Bolusan (40063663) Jeffrey Lam(40090989)
# For COMP 472 Section AA – Summer 2021
# --------------------------------------------------------

from classifier import * 
from service.imdb_service import ImdbService
import matplotlib.pyplot as plt
import numpy as np

#Scrape and store information
def smoothing():

    imdb = ImdbService.from_web()

    model = Classifier(imdb.reviews_df)

    smooth_values = np.arange(1, 2.01, 0.2)
    accuracies = []

    train_model, pos_total, neg_total, test_set = model.build_vocabulary()
    for s_val in smooth_values:
        s_val=round(s_val,1)

        train_model = model.modify_smooth(train_model, s_val)

        results = model.evaluate(train_model, test_set, pos_total, neg_total)

        correct_results = len(results[results["prediction"] == True])
        accuracy = (correct_results / len(results["prediction"]))*100
        print(f"smoothing: {s_val}\tAccuracy: {accuracy}")
        accuracies.append(accuracy)

        if s_val == 1:
            model.results_to_file(results, "result.txt")
        if s_val == 1.6:
            model.model_to_file(train_model, "smooth-model.txt")
            model.results_to_file(results, "smooth-result.txt")

    plt.title("Smoothing Classifier Performance")
    plt.xlabel("Smoothing Value")
    plt.ylabel("Accuracy")
    plt.plot(smooth_values, accuracies)
    plt.show()

smoothing()