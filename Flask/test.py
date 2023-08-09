'''
The render_template help to link html files created in a folder name "template".
Also, instead of importing Flask and the render_template, we can use the * to import everything
'''

from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/consent')
def consent():
    return "<h1>This is my first Flask App, and I hope to do something better next time</h1>"

@app.route('/about')
def about():
    return "We are new to Flask, and we are trying things out"



'''
Despite the import of the render_template, in which the template is the folder in which the index.html file is store,
we need to link directly to the html file in the template folder by using the @route('/index')
'''
@app.route('/index/<int:num>')
def table(num):
	return render_template('index.html',n=num) #'n' is the variable in the index html

if __name__ == '__main__':
    app.run(debug=True)
