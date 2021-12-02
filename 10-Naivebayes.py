import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#reading the dataset
ds = pd.read_csv("/home/ubuntu/Downloads/germany.csv")

#dataset visualization
sns.FacetGrid(ds,hue='Outcome',size=9).map(plt.scatter,'Pregnancies','Glucose').add_legend()


# In[10]:


X=ds.iloc[:,0:8]
y=ds['Outcome']
sc = StandardScaler()
#scalling the dataset
X = sc.fit_transform(X)
#splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)


# In[11]:


#training the model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#testing the model
y_pred = classifier.predict(X_test)


# In[13]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True)
#printing the accuracy
acc=accuracy_score(y_test, y_pred)
print ("Accuracy : ",acc )
#printing the classification report
target_names=y.unique()
target_names = list(map(str, target_names))
print(target_names)
report=classification_report(y_test, y_pred, target_names=target_names)
print(report)
