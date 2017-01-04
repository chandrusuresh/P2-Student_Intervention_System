import ReadData as RD

data = RD.readData()
RD.exploreData(data)
X_all, y_all = RD.prepareData(data)

print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
print list(X_all.columns)[32]

# TODO: Import any additional functionality you may need here

# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

#del X_all["failures"]

# TODO: Shuffle and split the dataset into the number of training and testing points above
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_all,y_all,stratify=y_all,test_size=0.24, random_state=1)


# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

import Classifiers as CLFRS
#CLFRS.fitData(X_train,y_train,X_test,y_test,"GaussianNB")
#CLFRS.fitData(X_train,y_train,X_test,y_test,"DecisionTree")
#CLFRS.fitData(X_train,y_train,X_test,y_test,"KNN")
#CLFRS.fitData(X_train,y_train,X_test,y_test,"SGD")
#CLFRS.fitData(X_train,y_train,X_test,y_test,"SVM")
#CLFRS.fitData(X_train,y_train,X_test,y_test,"LogisticRegression")
CLFRS.fitData(X_train,y_train,X_test,y_test,"AdaBoost")


