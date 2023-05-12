
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv('diabetes.csv')

diabetes.head()

from sklearn.model_selection import train_test_split

X=diabetes.drop('Outcome',axis=1)
Y=diabetes['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=50)

k_values = list(range(1, 31)) # List of k values to try
scores = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    # Perform cross-validation with 5 folds
    cv_scores = cross_val_score(model, X_train, Y_train, cv=5)
    # Take the average accuracy score across all folds
    avg_score = np.mean(cv_scores)
    scores.append(avg_score)

# Plot the accuracy scores for different k values
plt.plot(k_values, scores)
plt.xlabel('Number of neighbors (k)')
plt.ylabel('Cross-validated Accuracy')
plt.title('KNN Cross-validation for Diabetes Classification')
plt.show()

# Find the best value of k with the highest accuracy score
best_k = k_values[np.argmax(scores)]
print("Best value of k: ", best_k)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=6)

model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score

score=accuracy_score(Y_test,Y_pred)

score