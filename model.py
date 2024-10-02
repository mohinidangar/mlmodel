import pandas as pd
#to read csv file
from sklearn.preprocessing import  StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("iris.csv")

print(df.head()) # first top 5 rows

#select independent and dependent variable
X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
Y = df["species"]

#split the dataset into train and test
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3,random_state=50)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Instantiate the model
classifier = RandomForestClassifier()

#fit the model
classifier.fit(X_train,Y_train)

#make pickle file of our model
pickle.dump(classifier,open("model.pkl","wb"))