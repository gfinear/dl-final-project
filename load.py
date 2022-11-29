import pickle

with open('data/summary_data.p', 'rb') as data_file:
    data_dict = pickle.load(data_file)

print(len(data_dict["train_summaries"]))
print(len(data_dict["test_summaries"]))
print(len(data_dict["word2idx"]))