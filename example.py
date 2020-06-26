from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from ppa import calculate_feature_importances

n_feature_importance_runs = 5
train_test_ratio = 0.2
corr_threshold = 0.4

data = load_breast_cancer()
X, y = data.data, data.target
df = pd.DataFrame(data=X, columns=data.feature_names)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(df, y, df.index, test_size=train_test_ratio,
                                                                         random_state=5)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
baseline = clf.score(X_test, y_test)
print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))
df_corr = df.corr('spearman')

calculate_feature_importances(df, df_corr, y, n_feature_importance_runs, corr_threshold, baseline, clf,
                              [idx_train, idx_test], accuracy_score, train_test_ratio)

