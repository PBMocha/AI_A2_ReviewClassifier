# -------------------------------------------------------
# Assignment 2
# Written by Joshua Parial-Bolusan (40063663) Jeffrey Lam(40090989)
# For COMP 472 Section (your lab section) – Summer 2021
# --------------------------------------------------------

from classifier import * 
from service.imdb_service import ImdbService

#Scrape Site
imdb = ImdbService.from_web()
imdb.toCsv()

#Uncomment bottom to read from csv
#imdb = ImdbService.from_csv("data.csv")


model = Classifier(imdb.reviews_df)

train_model, pos_total, neg_total, test_set = model.build_vocabulary()

model.model_to_file(train_model, 'model.txt')
results = model.evaluate(train_model, test_set, pos_total, neg_total)

correct_results = len(results[results["prediction"] == True])
accuracy = (correct_results / len(results["prediction"]))*100
model.results_to_file(results, 'result.txt')
print(f"Accuracy: {accuracy}")