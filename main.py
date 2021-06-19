from classifier import * 

model = Classifier()

train_model, pos_total, neg_total, test_set = model.build_vocabulary()

model.model_to_file(train_model)
results = model.evaluate(train_model, test_set, pos_total, neg_total)

correct_results = len(results[results["prediction"] == True])
accuracy = (correct_results / len(results["prediction"]))*100
print(f"Accuracy: {accuracy}")