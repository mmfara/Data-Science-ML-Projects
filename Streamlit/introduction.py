import streamlit as st

#set title
st.title("Our First Streamlit App")

#to make use of image in your webpage, image needs to be called from PIL library
from PIL import Image

st.subheader("This is a subheader")

#assign the image and open with the fill image and then run with streamlit "st"
image = Image.open("dc.png")
st.image(image, use_column_width=True)

#This helps to write a text in our webpage
st.write("I am writing a text")

#This helps to include a markdown
st.markdown("This is a markdown cell")

#This helps to print out a message on  running a program
st.success("Congratulation, you run the app successfully")

#This help to give an information
st.info("This is an information for you")

#This helps to give warning message
st.warning("Be cautious with this app")

#This is to show error or warning
st.error("Oops, you run into an error, you need to rerun your app again or install the app again")

#This gives help information on functions
st.help(range)

import numpy as np
import pandas as pd


data = np.random.rand(10, 20)
st.dataframe(data)

st.text("----"*100)

#df = pd.DataFrame(np.random.rand(10, 20), columns=('col %d' % i for i in range(20)))
df = pd.DataFrame(np.random.rand(10, 20), columns=['col {}'.format(i) for i in range(20)])

#The style.highlight_max helps to show the maximun values in each columns
st.dataframe(df.style.highlight_max(axis=1))




st.text("----"*100)

#Display Chart

chart_data = pd.DataFrame(np.random.randn(24, 3), columns=['a', 'b', 'c'])
st.dataframe(chart_data)

#This create a line chart
st.line_chart(chart_data)  

st.text("----"*100)

#This create an area chart
st.area_chart(chart_data)

#This create an bar chart
st.bar_chart(chart_data)

st.text("----"*100)

import matplotlib.pyplot as plt

#plotting a chart with matplotlib
fig, ax = plt.subplots()
arr = np.random.normal(1, 1, size=100)
plt.hist(arr, bins=20)
st.pyplot(fig)

st.text("----"*100)

import plotly
import plotly.figure_factory as ff

#dding distplot

x1 = np.random.randn(200)-2
x2 = np.random.randn(200)
x3 = np.random.randn(200)-2

hist_data = [x1,x2,x3]
group_labels = ['Group1', 'Group2', 'Group3']

fig = ff.create_distplot(hist_data, group_labels,bin_size = [.2,.25,.5])
st.plotly_chart(fig, use_container_width=True)

st.text("----"*100)

df = pd.DataFrame(np.random.randn(100,2)/[50,50]+[37.76,-122.4], columns = ["lat","lon"])
st.map(df)

st.text("----"*100)


#creating buttons

st.text("----"*100)
if st.button("say hello"):
	st.write("Hello is here")
else: st.write("Why are you here")

st.text("----"*100)

#This help to ask question with adding options and giving a specific output
genre = st.radio("What is your favourite genre (A radio button)?", ('Comedy','Drama','Documentary'))
if genre == 'Comedy':
	st.write("Oh, You like comedy!")
elif genre == 'Drama':
	st.write("Yea, drama is good")
else:
	st.write("I see !!!")


st.text("----"*100)

#select button

option = st.selectbox("How was your night(A select button)?", ('Fantastic','Awesome','So-so'))
st.write("You said your night was:", option)

st.text("----"*100)
st.text("----"*100)

option = st.multiselect("How was your night(A multiselect button)?", ('Fantastic','Awesome','So-so'))
st.write("You said your night was:", option)

st.text("----"*100)
st.text("----"*100)

# A slider

# A slider helps to give range of a digit and where the slider should be initially placed 
age = st.slider("How old are you?",0,100,18)
st.write("Your age is :", age)

values = st.slider("Select a range of values?",0,200,(15,80))
st.write("You selected a range between :", values)

number = st.number_input('Input a number')
st.write("The number you inputed is:", number)

st.text("----"*100)
st.text("----"*100)

#File Uploader

#This helps users to upload file to the webpage
upload_file = st.file_uploader("Choose a csv file", type='csv')

if upload_file is not None:
	data = pd.read_csv(upload_file)
	st.write(data)
	st.success("successfully Uploaded")
else:
	st.markdown("Please upload a csv file")

st.text("----"*100)
st.text("----"*100)

#color picker

#This helps to select or pick different colors
# color = st.color_picker("Pick your preferred color", "#00F900")
# st.write("This is your color:", color)

st.text("----"*100)
st.text("----"*100)

#Side Bar

#This add a side bar to our webpage where users can select buttons or options from
add_sidebar = st.sidebar.selectbox("What is your favourite course?", ('A course from TDS on building Data Web APP', 'Others', 'I am not sure'))
st.write("Your favorite course is:", add_sidebar)

#If we want to add a button to the side bar, all we need to do is add sidebar at the front of any button we want.
#For example:
color = st.sidebar.color_picker("Pick your preferred color", "#00F900")
st.write("This is your color:", color)

st.text("----"*100)
st.text("----"*100)

#Progress Bar

#This helps create a progress bar which is a graphical user interface element that indicates the progress of a task or an operation
import time

my_bar = st.progress(0)
for percent_complete in range(101):
    time.sleep(0.1)
    my_bar.progress(percent_complete)



#Spinner

with st.spinner('wait for it...'):
	time.sleep(5)
st.success('Successful')

st.text("----"*100)
st.text("----"*100)

st.balloons()