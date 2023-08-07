import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from PIL import Image

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Set the title
st.title("My little exercise")

#Importing image library
image = Image.open('dc.png')
st.image(image)

#Adding a subtitle
st.subheader("Creating a webpage for a data science analysis")

#sliders
dataset = st.sidebar.selectbox("Datasets",('Breast Cancer','Wine','Diabetes'))

model = st.sidebar.selectbox("Algorithm",('KNN','SVM'))
# st.write('Your selected model is:', model)

#Loading the datasets
def load_datasets(name):
	data = None
	if name == "Breast Cancer":
		data = datasets.load_breast_cancer()
	elif name == "Wine":
		data = datasets.load_wine()
	else:
		data = datasets.load_diabetes()

	x = data.data
	y = data.target

	return data,x,y


data,x,y = load_datasets(dataset)

df_x = pd.DataFrame(x, columns=data.feature_names)  # Convert x into a DataFrame
st.dataframe(df_x)
st.write("The shape of x is:", df_x.shape)

st.write("Unique target variables",len(np.unique(y)))

#Plotting a box plot
plt.figure()
sns.boxplot(data = x)
st.pyplot(plt)

#BUILDING AKGORITHM PARAMETERS
def add_parameter(name_of_clf):
	param = {}
	if name_of_clf =='KNN':
		k = st.sidebar.slider('K',1,15)
		param['K']=k
	else:
		c = st.sidebar.slider('C',0.01,15.0)
		gamma = st.sidebar.slider('Gamma',0.01,15.0)
		param['C'] = c
		param['Gamma'] = gamma

	return param

params = add_parameter(model)

#BUILDING THE ALGORITHM
def classifier(name_of_clf, params):
	clf = None
	if name_of_clf == 'SVM':
		clf = SVC(C=params['C'],gamma=params['Gamma'])
	else:
		clf = KNeighborsClassifier(n_neighbors=params['K'])

	return clf 

algorithm = classifier(model,params)

#CREATING A RANDOM STATE
random_state =st.sidebar.slider("Random State",0,100)

#SPLITTING THE DATASET
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=random_state)

algorithm.fit(x_train,y_train)
y_pred = algorithm.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Your", model,"accuracy is:", accuracy)







