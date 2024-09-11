from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gradient_boosting import GradientBoostingClassifier


iris = load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_iterations = 100
learning_rate = 0.01
max_depth = 10
min_samples_split = 30


gb_classifier = GradientBoostingClassifier(max_depth=max_depth, min_leaf_samples=min_samples_split,
                                           criterion='entropy', num_iterations=num_iterations,
                                           lr=learning_rate)
gb_classifier.fit(X_train, y_train)
y_pred = gb_classifier.predict(X_test)

csfr = classification_report(y_test, y_pred)
print(csfr)
