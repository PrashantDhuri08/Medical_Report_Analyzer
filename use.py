import pickle

with open('./models/model_pklg', 'rb') as f:
    data = pickle.load(f)

pred= data.predict([[1,9,15,33,44]])

print(pred)