import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

#Set the title

st.title('Data Science Using Streamlit')

image = Image.open("dc.png")
st.image(image)

#Set the Subtitle
st.write("""
	# A simple Data App with Streamlit
	""")

st.write("""
	# let's explore different classifiers and datasets
	""")

dataset_name = st.sidebar.selectbox('select Dataset', ('Breast Cancer','Diabetes','Wine'))

classifier_name = st.sidebar.selectbox('Select Classifier',('SVM','KNN'))



#Create a function that takes in the desired dataset name
def get_dataset(name):
	data = None
	if name == 'Diabetes':
		data = datasets.load_diabetes()
	elif name == 'Wine':
		data = datasets.load_wine()
	else:
		data = datasets.load_breast_cancer()
	
	#Assigning the data and target value to x and y
	x = data.data
	y = data.target

	return x,y




#Calling the function
x,y = get_dataset(dataset_name)
st.dataframe(x)
st.write('Shape of the dataset is:', x.shape)
st.write('Unique target variables:',len(np.unique(y)))


#Plotting a box plot
plt.figure()
sns.boxplot(data = x, orient='h')
st.pyplot(plt)

plt.figure()
plt.hist(x)
st.pyplot(plt)

## Plotting a Pairplot
# # Convert numpy array 'x' into a pandas DataFrame
# df = pd.DataFrame(x)

# plt.figure()
# # Create the pair plot using Seaborn
# sns.pairplot(df)

# # Display the pair plot using st.pyplot() with the explicit figure argument
# st.pyplot(plt)






# BUILDING AN ALGORITHM
def add_parameter(name_of_clf):
    params = dict()
    if name_of_clf == 'SVM':
        c = st.sidebar.slider('C', 0.01, 15.0)
        gamma = st.sidebar.slider('gamma', 0.01, 15.0)
        params['C'] = c
        params['gamma'] = gamma
    elif name_of_clf == 'KNN':
        k = st.sidebar.slider('K', 1, 15)
        params['k'] = k
    return params

params = add_parameter(classifier_name)





# Accessing the classifier
def get_classifier(name_of_clf, params):
    clf = None
    if name_of_clf == 'SVM':
        clf = SVC(C=params['C'], gamma=params['gamma'])
    elif name_of_clf == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    return clf

clf = get_classifier(classifier_name, params)

#Creating a random state sidebar
random_state = st.sidebar.slider('random_state',0,100)


#Splitting the dataset
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=random_state)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

st.write(y_pred)

accuracy = accuracy_score(y_test,y_pred)
st.write('classifier name:',classifier_name)
st.write("The accuracy for your model is:", accuracy)