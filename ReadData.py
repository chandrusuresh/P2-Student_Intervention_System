# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score


def readData():
    # Read student data
    student_data = pd.read_csv("student-data.csv")
    print "Student data read successfully!"
    return student_data


def exploreData(student_data):
    # TODO: Calculate number of students
    n_students = student_data.shape[0]

    # TODO: Calculate number of features
    n_features = student_data.shape[1]

    # TODO: Calculate passing students
    n_passed = len(student_data[student_data["passed"] == "yes"].index.tolist())

    # TODO: Calculate failing students
    n_failed = n_students - n_passed # Assuming all students that didnt pass have failed

    # TODO: Calculate graduation rate
    grad_rate = n_passed/(n_students+0.0)*100.0

    # Print the results
    print "Total number of students: {}".format(n_students)
    print "Number of features: {}".format(n_features)
    print "Number of students who passed: {}".format(n_passed)
    print "Number of students who failed: {}".format(n_failed)
    print "Graduation rate of the class: {:.2f}%".format(grad_rate)

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)
    
    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        
        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

def prepareData(student_data):
    # Extract feature columns
    feature_cols = list(student_data.columns[:-1])

    # Extract target column 'passed'
    target_col = student_data.columns[-1]

    # Show the list of columns
    print "Feature columns:\n{}".format(feature_cols)
    print "\nTarget column: {}".format(target_col)

    # Separate the data into feature data and target data (X_all and y_all, respectively)
    X_all = student_data[feature_cols]
    y_all = student_data[target_col]

    # Show the feature information by printing the first five rows
    print "\nFeature values:"
    print X_all.head()

    X_all = preprocess_features(X_all)
    return X_all, y_all

