from classifier import * 
import matplotlib.pyplot as plt
import numpy as np

#Scrape and store information
def length():
    model = Classifier()

    length_values = np.array([2, 4, 9])
    accuracies = []

    train_model, pos_total, neg_total, test_set = model.build_vocabulary()

    #print(train_model)
    for length in length_values:
        
        train_model = model.modify_length(train_model, length)

    print(train_model)
        #results = model.evaluate(train_model, test_set, pos_total, neg_total)

      
        #model.model_to_file(train_model, "length-model.txt")
        #model.results_to_file(results, "length-result.txt")

    #plt.plot(length_values, accuracies)
    #plt.show()

length()