from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text-generator')
def generate_text():
    return render_template('text-generator.html')

@app.route('/text-generator-by-character', methods=['GET', 'POST'])
def generate_character_text():
    return render_template('text-generator-by-character.html')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
