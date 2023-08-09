from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    products = ['Product A', 'Product B', 'Product C']
    return render_template('products_template.html', products=products)

if __name__ == '__main__':
    app.run(debug=True)
