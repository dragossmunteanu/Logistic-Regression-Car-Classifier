from ucimlrepo import fetch_ucirepo 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 
  
# metadata 
print(car_evaluation.metadata) 
  
# variable information 
print(car_evaluation.variables) 


# Fetch the dataset
dataset = fetch_ucirepo(id=19)

# Data processing
X = dataset.data.features
y = dataset.data.targets

# Perform any necessary data cleaning and processing here

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model - Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)  # Modify the n_estimators parameter

# Train the model
rf_classifier.fit(X_train, y_train)

# Evaluate the model
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1_score = f1_score(y_test, rf_predictions)

# Compare with Logistic Regression/Softmax Regression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
lr_predictions = lr_classifier.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)

# Perform one-hot encoding on the categorical features
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Train the model
rf_classifier.fit(X_train_encoded, y_train)

# Evaluate the model
rf_predictions = rf_classifier.predict(X_test_encoded)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)

# Perform label encoding on the categorical features
label_encoder = LabelEncoder()
X_train_encoded = X_train.apply(label_encoder.fit_transform)
X_test_encoded = X_test.apply(label_encoder.transform)

# Train the model
rf_classifier.fit(X_train_encoded, y_train)

# Evaluate the model
rf_predictions = rf_classifier.predict(X_test_encoded)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)

# Calculate F1-score for Logistic Regression
rf_f1_score = f1_score(y_test, rf_predictions)
lr_f1_score = f1_score(y_test, lr_predictions)

# Print the performance metrics
print("Random Forest Classifier:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1-score:", rf_f1_score)

print("\nLogistic Regression:")
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("F1-score:", lr_f1_score)


