from classifier import * 
import matplotlib.pyplot as plt
import numpy as np

#Scrape and store information
def smoothing():
    model = Classifier()

    smooth_values = np.arange(1, 2.0, 0.2)
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

        if s_val == 1.6:
            model.model_to_file(train_model, "smooth-model.txt")
            model.results_to_file(results, "smooth-result.txt")

    plt.plot(smooth_values, accuracies)
    plt.show()

smoothing()