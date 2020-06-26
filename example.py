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


#
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# ax1.barh(tree_indices,
#          clf.feature_importances_[tree_importance_sorted_idx], height=0.7)
# ax1.set_yticklabels(data.feature_names[tree_importance_sorted_idx])
# ax1.set_yticks(tree_indices)
# ax1.set_ylim((0, len(clf.feature_importances_)))
# ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
#             labels=data.feature_names[perm_sorted_idx])
# fig.tight_layout()
# plt.show()
#
# ##############################################################################
# # Handling Multicollinear Features
# # --------------------------------
# # When features are collinear, permutating one feature will have little
# # effect on the models performance because it can get the same information
# # from a correlated feature. One way to handle multicollinear features is by
# # performing hierarchical clustering on the Spearman rank-order correlations,
# # picking a threshold, and keeping a single feature from each cluster. First,
# # we plot a heatmap of the correlated features:
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# corr = spearmanr(X).correlation
# corr_linkage = hierarchy.ward(corr)
# dendro = hierarchy.dendrogram(corr_linkage, labels=data.feature_names, ax=ax1,
#                               leaf_rotation=90)
# dendro_idx = np.arange(0, len(dendro['ivl']))
#
# ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
# ax2.set_xticks(dendro_idx)
# ax2.set_yticks(dendro_idx)
# ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
# ax2.set_yticklabels(dendro['ivl'])
# fig.tight_layout()
# plt.show()
#
# ##############################################################################
# # Next, we manually pick a threshold by visual inspection of the dendrogram
# # to group our features into clusters and choose a feature from each cluster to
# # keep, select those features from our dataset, and train a new random forest.
# # The test accuracy of the new random forest did not change much compared to
# # the random forest trained on the complete dataset.
# cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
# cluster_id_to_feature_ids = defaultdict(list)
# for idx, cluster_id in enumerate(cluster_ids):
#     cluster_id_to_feature_ids[cluster_id].append(idx)
# selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
#
# X_train_sel = X_train[:, selected_features]
# X_test_sel = X_test[:, selected_features]
#
# clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
# clf_sel.fit(X_train_sel, y_train)
# print("Accuracy on test data with features removed: {:.2f}".format(
#       clf_sel.score(X_test_sel, y_test)))
