import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('your.csv')

# Ensure all columns are numeric
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Handle missing values (e.g., by filling them with the mean of the column)
data = data.fillna(data.mean())

# Define independent and dependent variables
X = data[['your', 'independent', 'variables']]
y = data['your','dependent', 'variables']

# Add constant term for the intercept
X = sm.add_constant(X)

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Get the summary of the logistic regression
summary = result.summary()
print(summary)

# Get the Wald test for coefficients
wald_test = result.wald_test()
print("Wald Test:\n", wald_test)

# Get the odds ratios (exp(b)) for the coefficients
odds_ratios = np.exp(result.params)
print("Odds Ratios:\n", odds_ratios)

# Get the significance of the coefficients
significance = result.pvalues
print("Significance (p-values):\n", significance)

# Additional metrics
# Predicted probabilities
predicted_probs = result.predict(X)

# Classification based on threshold 0.5
predictions = [1 if x >= 0.5 else 0 for x in predicted_probs]

# Confusion matrix
confusion_matrix = pd.crosstab(y, predictions, rownames=['Actual'], colnames=['Predicted'])
print("Confusion Matrix:\n", confusion_matrix)

# Model accuracy
accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / confusion_matrix.sum().sum()
print("Accuracy:\n", accuracy)
