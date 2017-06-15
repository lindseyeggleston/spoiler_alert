from flask import Flask, request, render_template, url_for
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generic')
def generate_text():
    return render_template('generic.html')

@app.route('/character', methods=['GET', 'POST'])
def generate_character_text():
    return render_template('character.html')




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
