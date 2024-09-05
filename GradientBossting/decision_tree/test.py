from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from decision_tree_classifier import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


max_depth = None
min_samples_split = 30


tree_classifier = DecisionTreeClassifier(max_depth=max_depth, min_leaf_samples=min_samples_split,
                                         criterion='entropy')
tree_classifier.fit(X_train, y_train)
y_pred = tree_classifier.predict(X_test)

csfr = classification_report(y_test, y_pred)
print(csfr)