from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    user_name = "John"
    return render_template('greeting_template.html', name=user_name)

if __name__ == '__main__':
    app.run(debug=True)
