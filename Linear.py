from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sklearn
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


path = f'D:\DL_Data\LinearRegression'
file = f'student-mat.csv'

data = pd.read_csv(path+'\\'+file, sep=';')

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = 'G3'

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# predictions = 

# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])

best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))
    
    # If the current model has a better score than one we've already trained then save it
    if acc > best:
        best = acc
        with open("stu.pickle", "wb") as f:
            pickle.dump(linear, f)

print('Best result is', best)

# with open("stu.pickle", 'rb') as f:
#     linear = pickle.load(f)

# plot = "failures"
# plt.scatter(data[plot], data["G3"])
# plt.legend(loc=4)
# plt.xlabel(plot)
# plt.ylabel("Final Grade")
# plt.show()