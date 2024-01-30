import pandas

from sklearn import tree



dataset = pandas.read_csv("temperature_data.csv")



dataset = pandas.get_dummies(dataset)



dataset = dataset.sample(frac = 1).reset_index()




target = dataset["actual"].values


data = dataset.drop("actual", axis =1)
data = dataset.drop("level_0", axis = 1)

data = dataset.drop(["actual", "level_0"], axis =1 )

feature_list = data.columns

data = data.values

print(feature_list)
print(target)
print(data)


machine = tree.DecisionTreeClassifier(criterion = "gini", max_depth = 10)

machine.fit(data, target)

feature_importances_raw = machine.feature_importances_

print(feature_importances_raw)

print(feature_list)

feature_zip = zip(feature_list, feature_importances_raw)

print(feature_zip)

[(feature, round)]
