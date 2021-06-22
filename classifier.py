from typing import Dict
from service import imdb_service
from service.imdb_service import *
from math import log10
import re
import numpy as np


class Classifier:

    def __init__(self):
        self.imdb = ImdbService()
        
    def build_vocabulary(self, smooth=1):

        reviews_df = self.imdb.reviews_df
        positive_df = reviews_df[reviews_df["rating"] == "positive"]
        negative_df = reviews_df[reviews_df["rating"] == "negative"]

        pos_off = int(len(positive_df)/2)
        neg_offset = int(len(negative_df)/2)

        train_df = (positive_df.iloc[:pos_off]).append(negative_df.iloc[:neg_offset])
        train_pos = train_df[train_df["rating"] == "positive"]
        train_neg = train_df[train_df["rating"] == "negative"]

        test_df = (positive_df.iloc[pos_off:]).append(negative_df.iloc[neg_offset:])

        frequencies = {}

        stop_word_file = open("stopword.txt", "r")
        stop_words = [word.strip() for word in stop_word_file.readlines()]

        stop_word_file.close()

        model = {
            "word": [],
            "positive": [],
            "positive_prob": [],
            "negative": [],
            "negative_prob": []
        }

        #Count Frequencies
        for index, row in train_df.iterrows():
            for word in row["review"].split(" "):

                if word not in stop_words:
                    frequencies.setdefault(word, {"positive": 0, "negative": 0})
                    frequencies[word][row["rating"]] += 1

                

        words_in_pos = 0
        words_in_neg = 0
        vocab_size = len(frequencies.keys())

        for key in frequencies.keys():
            words_in_pos += frequencies[key]["positive"]
            words_in_neg += frequencies[key]["negative"]

        for word in frequencies.keys():

            pos_freq = frequencies[word]["positive"]
            neg_freq = frequencies[word]["negative"]

            model["word"].append(word)
            model["positive"].append(pos_freq)
            model["positive_prob"].append((pos_freq + smooth) / (vocab_size + words_in_pos))

            model["negative"].append(neg_freq)
            model["negative_prob"].append((neg_freq + smooth) / (vocab_size + words_in_neg))

        return pd.DataFrame(model), len(train_pos), len(train_neg), test_df

    def modify_smooth(self, train_model: pd.DataFrame, smooth=1):
        
        vocab_size = train_model.shape[0]
        pos_count = train_model["positive"].sum()
        neg_count = train_model["negative"].sum()

        train_model["positive_prob"] = train_model.apply(lambda row: ((row.positive+ smooth) / (vocab_size + pos_count)), axis=1)
        train_model["negative_prob"] = train_model.apply(lambda row: ((row.negative + smooth) / (vocab_size + neg_count)), axis=1)
        
        return train_model

    def modify_length(self, train_model: pd.DataFrame, length):
        for index, word in enumerate(train_model["word"]):
            if (len(word) <= length) & (length <= 4):
                print(train_model.loc[[index]])
                train_model = train_model.drop(index, axis=0)
                

            elif (len(word) >= length) & (length >= 9):
                print(train_model.loc[[index]])
                train_model = train_model.drop(index, axis =0)
                

        return train_model

    def model_to_file(self, model: pd.DataFrame, file: str):

        model_file = open(file, "w", encoding="utf-8")

        for index, row in model.iterrows():

            word = row["word"]
            pos_f = row["positive"]
            pos_prob = row["positive_prob"]
            neg_f = row["negative"]
            neg_prob = row["negative_prob"]

            model_file.write(f"No.{index} {word}\n{pos_f}, {pos_prob}, {neg_f}, {neg_prob}\n")
        
        model_file.close()

    def results_to_file(self, results: pd.DataFrame, file):
        results_file = open(file, "w", encoding="utf-8")

        for index, row in results.iterrows():

            title = row["title"]
            pos_result = row["positive"]
            neg_result = row["negative"]
            result = row["result"]
            actual_result = row["actual_result"]
            outcome = "right" if row["prediction"] else "wrong"

            results_file.write(f"No.{index} {title}\n{pos_result}, {neg_result}, {result}, {actual_result}, {outcome}\n")
        
        correct_results = len(results[results["prediction"] == True])
        accuracy = (correct_results / len(results["prediction"]))*100
        results_file.write(f"The prediction accuracy is: {accuracy}%")
        
        results_file.close()

    def evaluate(self, train_model: pd.DataFrame, test_set: pd.DataFrame, pos_total, neg_total):

        results = {
            "title": [],
            "positive": [],
            "negative": [],
            "result": [],
            "actual_result": [],
            "prediction": [],
            }

        for index, row in test_set.iterrows():
            review = row["review"]

            pos_prob = pos_total / (pos_total+neg_total)
            neg_prob = neg_total / (pos_total+neg_total)

            for word in review.split(" "):

                if word in train_model["word"].values:
                    
                    _word = train_model[train_model["word"] == word]

                    word_pos = _word["positive_prob"].values[0]
                    word_neg = _word["negative_prob"].values[0]

                    pos_prob += log10(word_pos)
                    neg_prob += log10(word_neg)

            train_result = "positive" if pos_prob > neg_prob else "negative"

            results["title"].append(row["title"])
            results["positive"].append(pos_prob)
            results["negative"].append(neg_prob)
            results["result"].append(train_result)
            results["actual_result"].append(row["rating"])
            results["prediction"].append(train_result == row["rating"])

        return pd.DataFrame(results)

#print(build_vocabulary())