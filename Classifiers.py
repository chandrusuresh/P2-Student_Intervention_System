# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import f1_score
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt

def fitData(X_train,y_train,X_test,y_test,classifier):
    param = [""]
    paramName = ""
    if classifier == "GaussianNB":
        clf1 = [GaussianNB()]
    elif classifier == "DecisionTree":
        clf1 = []
        clf1.append(DecisionTreeClassifier(random_state = 42))
    elif classifier == "KNN":
        clf1 = []
        clf1.append(KNeighborsClassifier())
    elif classifier == "SGD":
        clf1 = [SGDClassifier(random_state = 42)]
    elif classifier == "SVM":
        clf1 = []
        clf1.append(SVC(random_state = 42))
    elif classifier == "LogisticRegression":
        clf1 = [LogisticRegression()]
    else:
        clf1 = []
        param1 = [DecisionTreeClassifier(random_state = 42),GaussianNB(),LogisticRegression()]#,SGDClassifier()]
        param = ["DT","NB","LogReg"]#,"SGD"]
        paramName = ""
        for i in param1:
            clf1.append(AdaBoostClassifier(i))

    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for i in range(len(param)):
        clf = clf1[i]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if classifier == "DecisionTree":
            with open("student-intervention_DT_maxDepth_" + str(param[i]) + ".dot",'w') as f:
                f = tree.export_graphviz(clf,out_file=f,feature_names=X_train.columns,class_names=["yes","no"])
        elif classifier == "SVM":
            print clf.support_vectors_
            #ax3.plot(clf.support_vectors_, "s-", label="%s" % (classifier+"_"+paramName+"_"+param[i], ))
        else:
            pass

        F1 = f1_score(y_test, y_pred, pos_label='yes')
        print classifier + ": " + paramName + " = " + str(param[i]) + " : F1 score = " + str(F1)

        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

        if type(param[i]) == type(str()):
            print classifier+"_"+paramName+"_"+param[i]
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (classifier+"_"+paramName+"_"+param[i], ))
            ax2.hist(prob_pos, range=(0, 1), bins=10, label=classifier+"_"+paramName+"_"+param[i], histtype="step", lw=2)
        else:
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (classifier+"_"+paramName+"_"+str(param[i]), ))
            ax2.hist(prob_pos, range=(0, 1), bins=10, label=classifier+"_"+paramName+"_"+str(param[i]), histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.show()
