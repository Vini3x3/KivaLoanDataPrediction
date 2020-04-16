# Project Programming Tests

## Dataset

`kiva_loans_standardized.csv` 

`kiva_loans.csv` 

## Tests

### 1. Brief Exploration

- Use k-fold validation for k=10
- Stick to default parameters

### 1A. No feature Extraction

Test data: use all data from `kiva_loans_standardized.csv`  except  `repayment_interval_bullet`, `repayment_interval_irregular`, `repayment_interval_weekly`, `repayment_interval_monthly`

Test label: use `repayment_interval` from `kiva_loans.csv`

Model under test: 

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest

==Count: 4 `summary`==

==What do we want: show Decision Tree is the best model==

### 1B. Use LDA

Test data: use all data from `kiva_loans_standardized.csv`  except  `repayment_interval_bullet`, `repayment_interval_irregular`, `repayment_interval_weekly`, `repayment_interval_monthly`.  Then LDA it, use default parameters

Test label: use `repayment_interval` from `kiva_loans.csv`

Model under test: 

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- K Nearest Neighbor
- Support Vector Machine

==Count: 6 `summary`==

==What do we want: show Decision Tree is the best model using LDA==

## 2. Advanced Training

### 2A. Use Feature Extraction ( LDA )

Test data: use all data from `kiva_loans_standardized.csv`  except  `repayment_interval_bullet`, `repayment_interval_irregular`, `repayment_interval_weekly`, `repayment_interval_monthly`.  Then LDA it, use default parameters

Test label: use `repayment_interval` from `kiva_loans.csv`

Model under test: Decision Tree 

==**Note: If Decision Tree is not the best model in the Brief Exploration, then use that model.  **==

==**Note: If you already run that in Brief Exploration, the no need to run again.  **==

==What do we want: LDA does not work well==

==Count: 1 `summary`==

### 2B. Use Feature Extraction (Best Subset)

Test data: use all data from `kiva_loans_standardized.csv`  except  `repayment_interval_bullet`, `repayment_interval_irregular`, `repayment_interval_weekly`, `repayment_interval_monthly`.  Then LDA it, use default parameters

Test label: use `repayment_interval` from `kiva_loans.csv`

Model under test: Decision Tree 

Target: 

```
For i in range(1,11): 
	Select i * 27 best columns from dataset.  i.e. 27, 27 * 2, 27 * 3, 27 * 4, ... 27 * 10.  
	k-fold for k=10: 
		Fit and train the model
		Predict
		Then record the R-square value, p-value, t-value, whatever it needs, for each test.  
	end k-fold
	record the average of each indicators in the k-fold
Endfor
Plot a graph n-subset vs indicator.  e.g. number of subset vs R-square value
```

==If there is a peak, then choose the peak and output the corresponding columns.  If there is no peak, then simpply choose all columns==

==**Note: If Decision Tree is not the best model in the Brief Exploration, then use that model.  **==

==Count: at least 1 graph==

==What do we want: select the columns that is the best for accurate prediction==

  ### 2C. Use Filtering

**Step 1: Build a Decision Tree that can identify whether it is a `weekly` or not**

Test data: use all data from `kiva_loans_standardized.csv`  except  `repayment_interval_bullet`, `repayment_interval_irregular`, `repayment_interval_weekly`, `repayment_interval_monthly`.  

Test label: use `repayment_interval_weekly` from `kiva_loans_standardized.csv`

Model under test: Decision Tree 

==**Note: If Decision Tree is not the best model in the Brief Exploration, then use that model.  **==

**Step 2: Build a Decision Tree that can identify whether it is a `bullet`, `monthly` or `irregular`**

Test data: use all data from `kiva_loans_standardized.csv`  except  `repayment_interval_bullet`, `repayment_interval_irregular`, `repayment_interval_weekly`, `repayment_interval_monthly`. then drop all row that its label is `weekly`.  

Test label: use `repayment_interval_weekly` from `kiva_loans_standardized.csv`, then drop all row that its label is `weekly`.  

**Step 3: Build a model that first filter out all the `weekly` and then if remains identify it is `monthly`, `bullet` or `irregular`**

Test data: use all data from `kiva_loans_standardized.csv`  except  `repayment_interval_bullet`, `repayment_interval_irregular`, `repayment_interval_weekly`, `repayment_interval_monthly`.  

Test label: use `repayment_interval` from `kiva_loans.csv`

Model under test: Decision Tree 

```
dtree1 # the decision tree for filter out weekly
dtree2 # the decision tree that only can identify it is monthly, bullet or irregular
X, y # X = dataset, y = label in string
all_predict, all_true # array for storing the predictions and true labels.  
for k-fold: 
	X_train, X_test, y_train, y_test # made up by the k-fold
	
	# train the dtree1
	y_train_weekly = ifelse(y_train = 1 if 'weekly' else 0)
	dtree1.fit(X_train, y_train_weekly)
	
	# make a weekly_free dataset
	weekly_indices # row number of the row that its label is weekly
	y_train_no_weekly = y_train.drop(weekly_indices)
	X_train_no_weekly = X.train.drop(weekly_indices)
	
	dtree2.fit(X_train_no_weekly, y_train_weekly)
	
	# prediction model
	prediction = []
	prediction = dtree1.predict(X_train)
	for i in nrow(prediction):
		if prediction[i] == 1:
			prediction[i] = 'weekly'
		else: 
			prediction[i] = dtree2.predict(X_test[i])
	
	# now you have prediction and y_test.  
	all_predict = prediction
	all_true = y_test
	
# summary()
# by using all_predict and all_true for an average performance.  
```

==**Note: If Decision Tree is not the best model in the Brief Exploration, then use that model.  **==

==Count: 1 summary==

==What do we want: Filtering fails, and we are so creative==

### 2C. Tune Parameters

Tune the parameters.  If any in R.  

The parameters in Python sklearn is `max_depth` and `min_impurity_decrease`.  If there are no such thing in R, just forget it and test other parameters in R decision tree.  

Print some graph for varying the parameter, say, maximum depth vs R-square error.  

Print a summary for the best parameters to show the improvement

==Count: at least 1 summary, at least 1 graph==

==What do we want: the best parameters==

## Graph

Print the decision tree.  

1. First graph

Test data: use all data from `kiva_loans_standardized.csv`  except  `repayment_interval_bullet`, `repayment_interval_irregular`, `repayment_interval_weekly`, `repayment_interval_monthly`

Test label: use `repayment_interval` from `kiva_loans.csv`

Possible function call:   rpart.plot() 

2. Second graph

Test data: use all data from `kiva_loans_dummied.csv`  except  `repayment_interval_bullet`, `repayment_interval_irregular`, `repayment_interval_weekly`, `repayment_interval_monthly`

Test label: use `repayment_interval` from `kiva_loans.csv`

Possible function call:   rpart.plot() 

==What do we want: a non-standardized dataset, like term_in_months =25, not 0.11 in normal distribution, which is difficult to read==

3. if possible, plot some graph to show how the data is divided.  But I am not optimistic because there are so many columns / dimensions and difficult to visualize.  

## Reference

 https://www.guru99.com/r-decision-trees.html 