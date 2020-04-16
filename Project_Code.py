# Import Library

"""
A set of libraries that implement machine learning in Python.  
Below is the list: 
-   scikit-learn
-   pandas
-   matplotlib
-   seaborn
-   graphviz

Note that some libraries may not be used, but definitely 
sufficient for running the below codes.  
"""

## data visualization and utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import graphviz
%matplotlib inline

## classifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import Lasso

## evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import precision_score

## data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

## Regression Report
def regression_report(y_test, y_pred):
    
    space_tab = 3
    
    if len(y_test) != len(y_pred):
        return 'length of true labels and predicted labels are not equal.'
    
    report = []
    
    # labels
    labels = set(y_test)
    for label in labels:
        test = [1 if each==label else 0 for each in y_test]
        pred = [1 if each==label else 0 for each in y_pred]
        report.append([
            label, 
            round(r2_score(test,pred), 3), 
            round(mean_squared_error(test, pred),3), 
            round(explained_variance_score(test,pred),3), 
            y_test.count(label)
        ])
    
    # macro
    macro = [
        'macro avg', 
        round(sum([row[1] for row in report]) / len(labels),3), 
        round(sum([row[2] for row in report]) / len(labels),3), 
        round(sum([row[3] for row in report]) / len(labels),3), 
        sum([row[4] for row in report])
    ]           
    
    # micro    
    diff = [1 if y_test[i]==y_pred[i] else 0 for i in range(len(y_test))]
    same = [1] * len(y_test)
    micro = [
        'micro avg', 
        round(r2_score(diff,same),3), 
        round(mean_squared_error(diff,same), 3),
        round(explained_variance_score(diff,same), 3),
        len(y_test)
    ]
    
    #formatting
    space = ['    ', '    ', '    ', '    ', '    ']    
    header = ['    ', 'r2_score', 'mean_squared_error', 'explained_variance_score', 'support']
    
    # add all the things    
    report.insert(0,space)
    report.insert(0,header)
    report.append(space)
    report.append(micro)
    report.append(macro)        
    
    result = ''
    
    col = []
    for i in range(len(report[0])):
        col.append(max([len(str(row[i])) for row in report])+space_tab)
    
    for row in report:
        for i in range(len(row)):
            result += str(row[i]).rjust(col[i], ' ')
        result += '\n'
    return result   

# Import Dataset
"""
Import all versions of dataset for different tests.  
"""

loan_raw = pd.read_csv('kiva_loans.csv')
loan_std = pd.read_csv('kiva_loans_standardized.csv')
loan_cod = pd.read_csv('kiva_loans_dummied.csv')

# Model Selection
## Standardized One-hot encoded Dataset

selected_features = list(loan_std.columns)
selected_features.remove('repayment_interval_irregular')
selected_features.remove('repayment_interval_monthly')
selected_features.remove('repayment_interval_weekly')
selected_features.remove('repayment_interval_bullet')

""""
This is the datset.  
X is the dataset without the labels, while y are the labels
X[i]'s label is y[i] for i in range(len(y))
"""
y = loan_raw['repayment_interval']
X = loan_std[selected_features]

### Perceptron (Linear Regression)

model = Perceptron(tol=100)
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

### Logistic Regression

model = LogisticRegression(solver='lbfgs', multi_class='auto')
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

### Decisino Tree
model = DTC()
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

### Random Forest
model = RandomForestClassifier(n_estimators = 10)
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

## Standardized One-hot encoded Dataset

## LDA converted dataset
lda = LDA()
X_lda = lda.fit_transform(X,y)
X_lda = pd.DataFrame(X_lda)

""""
This is the datset.  
X_lda is the dataset without the labels after transformed with LDA, while y are the labels
X_lda[i]'s label is y[i] for i in range(len(y))
y is the same as before.  
"""

### Perceptron
model = Perceptron(tol = 100)
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X_lda):
    
    X_train, X_test = X_lda.iloc[train_index], X_lda.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

### Logistic Regression
model = LogisticRegression(solver='lbfgs', multi_class='auto')
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X_lda):
    
    X_train, X_test = X_lda.iloc[train_index], X_lda.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

### Decision Tree
model = DTC()
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X_lda):
    
    X_train, X_test = X_lda.iloc[train_index], X_lda.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

### Random Forest
model = RandomForestClassifier(n_estimators = 10)
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X_lda):
    
    X_train, X_test = X_lda.iloc[train_index], X_lda.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

### K Nearest Neighbors
model = KNN(n_neighbors=10)
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X_lda):
    
    X_train, X_test = X_lda.iloc[train_index], X_lda.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

### Support Vector Machine

model = SVC(gamma='auto')
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X_lda):
    
#     print('Start Test Iteration ',i)
    X_train, X_test = X_lda.iloc[train_index[:10000]], X_lda.iloc[test_index[:1000]]
    y_train, y_test = y.iloc[train_index[:10000]], y.iloc[test_index[:1000]]

#     print('Start Fitting Iteration ',i)
    model.fit(X_train,y_train)    
#     print('Start Prediction Iteration ',i)
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)
    
    print('Finish Test Iteration ',i)
    i += 1

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

# Advanced Training

## Best Subset

score = []
j = 1
for j in range(1,11):
    
    X_subset = SelectKBest(f_classif, k=j*27).fit_transform(X, y)
    X_subset = pd.DataFrame(X_subset)
    
    dtree = DTC()

    ALL_TRUE_LABEL = []
    ALL_PRED_LABEL = []
    kf = KFold(n_splits=10)
    i = 0

    for train_index, test_index in kf.split(X_subset):

        X_train, X_test = X_subset.iloc[train_index], X_subset.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        dtree.fit(X_train,y_train)
        ALL_PRED_LABEL.extend(dtree.predict(X_test))
        ALL_TRUE_LABEL.extend(y_test)
        
        print('Finish Test Iteration ',i)
        i += 1
    score.append(precision_score(ALL_TRUE_LABEL, ALL_PRED_LABEL, average = 'macro'))
    print('Finish Subset Iteration ',j)    

k_subset = [i * 27 for i in range(1,11)]
plt.plot(k_subset, score)
plt.xlabel('Precision(macro)')
plt.ylabel('Number of Selected Columns')
plt.title('Best Subset Selection')
plt.show()


## Tuning Parameters

### Finding the best value for minimum split
score = []
impurity = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0]
j = 1
for j in range(len(impurity)):
            
    dtree = DTC(max_depth = 46000, min_impurity_decrease = impurity[j])

    ALL_TRUE_LABEL = []
    ALL_PRED_LABEL = []
    kf = KFold(n_splits=10)
    i = 0

    for train_index, test_index in kf.split(X):    

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        dtree.fit(X_train,y_train)
        ALL_PRED_LABEL.extend(dtree.predict(X_test))
        ALL_TRUE_LABEL.extend(y_test)

        # Screen Output for tracking the progress, sometimes I wait too long......
        print('Finish Test Iteration ',i)
        i += 1
    score.append(precision_score(ALL_TRUE_LABEL, ALL_PRED_LABEL, average = 'macro'))
    print('Finish Depth Iteration ',j)
    print(classification_report(ALL_TRUE_LABEL,ALL_PRED_LABEL))
    print(confusion_matrix(ALL_TRUE_LABEL,ALL_PRED_LABEL))

min_split = ['1e-1', '1e-2', '1e-3', '1e-4', '1e-5', '1e-6', '1e-7', '1e-8', '1e-9', '0']

plt.plot(min_split, score)
plt.xlabel('Precision(macro)')
plt.ylabel('Minimum Split')
plt.grid()
plt.title('Precision(macro) vs Min Split')
plt.show()

### Finding the best value for max depth, with min_impurity_decrease set to be 1e-5 (best 
### min split)

score = []
j = 1
for j in range(1,11):
            
    dtree = DTC(max_depth = j * 10000, min_impurity_decrease = 1e-5)

    ALL_TRUE_LABEL = []
    ALL_PRED_LABEL = []
    kf = KFold(n_splits=10)
    i = 0

    for train_index, test_index in kf.split(X):    

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        dtree.fit(X_train,y_train)
        ALL_PRED_LABEL.extend(dtree.predict(X_test))
        ALL_TRUE_LABEL.extend(y_test)
        
        print('Finish Test Iteration ',i)
        i += 1
    score.append(precision_score(ALL_TRUE_LABEL, ALL_PRED_LABEL, average = 'macro'))
    print('Finish Subset Iteration ',j)

max_depth = [i * 10000 for i in range(1,11)]
plt.plot(max_depth, score)
plt.xlabel('Precision(macro)')
plt.ylabel('Maximum Depth')
plt.title('Precision(macro) vs Max Depth with min split=1e-5')
plt.show()

### Further Search on depth after fixing the best depth to be in range(55000,65000)

score = []
j = 1
for j in range(55,65):
            
    dtree = DTC(max_depth = j * 1000)

    ALL_TRUE_LABEL = []
    ALL_PRED_LABEL = []
    kf = KFold(n_splits=10)
    i = 0

    for train_index, test_index in kf.split(X):    

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        dtree.fit(X_train,y_train)
        ALL_PRED_LABEL.extend(dtree.predict(X_test))
        ALL_TRUE_LABEL.extend(y_test)

        # Screen Output for tracking the progress, sometimes I wait too long......
        print('Finish Test Iteration ',i)
        i += 1
    score.append(precision_score(ALL_TRUE_LABEL, ALL_PRED_LABEL, average = 'macro'))
    print('Finish Depth Iteration ',j)
    print(classification_report(ALL_TRUE_LABEL,ALL_PRED_LABEL))
    print(confusion_matrix(ALL_TRUE_LABEL,ALL_PRED_LABEL))

max_depth = [i * 10000 for i in range(55,65)]
plt.plot(max_depth, score)
plt.xlabel('Precision(macro)')
plt.ylabel('Maximum Depth')
plt.title('Precision(macro) vs Max Depth with min split=1e-5')
plt.show()

### Final result for Decision Tree with fine-tunned parameters

dtree = DTC(max_depth = 61000, min_impurity_decrease = 1e-5)

ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0

    for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    dtree.fit(X_train,y_train)
    ALL_PRED_LABEL.extend(dtree.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)

    # Screen Output for tracking the progress, sometimes I wait too long......
    print('Finish Test Iteration ',i)
    i += 1    

print(classification_report(ALL_TRUE_LABEL,ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL,ALL_PRED_LABEL))

## Filtering

"""
First train a decision tree for filtering all the weekly.  
Then train a decision tree for identifying all the irregular, 
monthly and bullet.  
"""
### Making special dataset
selected_features = list(loan_std.columns)
selected_features.remove('repayment_interval_irregular')
selected_features.remove('repayment_interval_monthly')
selected_features.remove('repayment_interval_bullet')

X_filter = loan_std[selected_features]

### Code for the training

dtree1 = DTC()
dtree2 = DTC()

ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X_filter):
    
    X_train, X_test = X_filter.iloc[train_index], X_filter.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    selected_columns = list(X_filter.columns)
    selected_columns.remove('repayment_interval_weekly')
    
    # make a dataset only contains `weekly` and `non-weekly` label
    dataset_weekly_train, dataset_weekly_test = X_train[selected_columns], X_test[selected_columns]
    label_weekly_train, label_weekly_test = X_train['repayment_interval_weekly'], X_test['repayment_interval_weekly']
    
    # make a dataset without `weekly` data and label
    dataset_no_weekly_train, dataset_no_weekly_test = X_train.loc[X_train['repayment_interval_weekly'] < 1], X_test.loc[X_test['repayment_interval_weekly'] < 1]
    dataset_no_weekly_train, dataset_no_weekly_test = dataset_no_weekly_train[selected_columns], dataset_no_weekly_test[selected_columns]
    label_no_weekly_train, label_no_weekly_test = y_train.loc[y_train != 'weekly'], y_test.loc[y_test != 'weekly']
        
    # train a dtree for recognizing `weekly` or not
    dtree1.fit(dataset_weekly_train,label_weekly_train)
        
    # train a dtree for recognizing `irregular`,`monthly` and `bullet`        
    dtree2.fit(dataset_no_weekly_train,label_no_weekly_train)
    
    # start prediction
    y_pred_1 = dtree1.predict(dataset_weekly_test)  
    y_pred_2 = dtree2.predict(dataset_weekly_test)
    
    # Merge Prediction Result
    y_pred = []
    for j in range(len(test_index)):
        if y_pred_1[j]:
            y_pred.append('weekly')
        else:
            y_pred.append(y_pred_2[j])

    ALL_PRED_LABEL.extend(y_pred)
    ALL_TRUE_LABEL.extend(y_test)
    
    # Screen Output for tracking the progress, sometimes I wait too long......
    print('Finish Test Iteration ',i)
    i += 1
#     break
    
print(classification_report(ALL_TRUE_LABEL,ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL,ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL,ALL_PRED_LABEL))

# Final Product

model = DTC(min_impurity_decrease=1e-5, max_depth = 61000)
ALL_TRUE_LABEL = []
ALL_PRED_LABEL = []
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train,y_train)    
    
    ALL_PRED_LABEL.extend(model.predict(X_test))
    ALL_TRUE_LABEL.extend(y_test)

    # Screen Output for tracking the progress, sometimes I wait too long......
    print('Finish Test Iteration ',i)
    i += 1
#     break

print(classification_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(regression_report(ALL_TRUE_LABEL, ALL_PRED_LABEL))
print(confusion_matrix(ALL_TRUE_LABEL, ALL_PRED_LABEL))

# Printing the final product (Decision Tree)

## train a decision tree with one-hot encoded and standardized data

dtree = DTC(max_depth = 61000, min_impurity_decrease = 1e-5)
dtree.fit(X,y)
from graphviz import Source
from sklearn.tree import export_graphviz
dotfile = open("dtree.dot", 'w')
export_graphviz(dtree, out_file = dotfile, feature_names = X.columns)
dotfile.close()

# train a decision tree with one-hot encoded data only, thus easier to 
# read the decision tree.  

selected_features = list(loan_cod.columns)
selected_features.remove('repayment_interval_irregular')
selected_features.remove('repayment_interval_monthly')
selected_features.remove('repayment_interval_weekly')
selected_features.remove('repayment_interval_bullet')

X_encoded = loan_encoded[selected_features]
dtree = DTC(max_depth = 61000, min_impurity_decrease = 1e-5)
dtree.fit(X_encoded,y)

dotfile = open("dtree2.dot", 'w')
export_graphviz(dtree, out_file = dotfile, feature_names = X.columns)
dotfile.close()
