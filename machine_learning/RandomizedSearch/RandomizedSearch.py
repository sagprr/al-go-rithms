# importing the dependencies
import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


dataset = sklearn.datasets.load_breast_cancer()

print(dataset)

data_frame = pd.DataFrame(dataset.data,columns = dataset.feature_names)
data_frame.head()
data_frame['label'] = dataset.target
data_frame.head()

data_frame.shape

data_frame.isnull().sum()

data_frame.drop(columns='label',axis=1)

data_frame['label'].value_counts()

#1 --> Benign
#0 --> Malignant

x = data_frame.drop(columns='label',axis=1)
y = data_frame['label']


print(x)
print(y)


x = np.asarray(x)
y = np.asarray(y)


model = SVC()

parameters = {
              'kernel':['linear','poly','rbf','sigmoid'],
              'C':[1, 5, 10, 20]
}


classifier = RandomizedSearchCV(model, parameters, cv=5)


classifier.fit(x,y)
classifier.cv_results_

# best parameters
best_parameters = classifier.best_params_
print(best_parameters)

# higest accuracy
highest_accuracy = classifier.best_score_
print(highest_accuracy)


# loading the results to pandas dataframe
result = pd.DataFrame(classifier.cv_results_)


result.head()



randomized_search_result = result[['param_C','param_kernel','mean_test_score']]


