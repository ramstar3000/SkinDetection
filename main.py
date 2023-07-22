from flask import Flask, render_template, request, session, redirect, url_for
import os
from cnn import cnnn
import requests

# Create flask application and add encryption value
app = Flask(__name__)
app.secret_key = 'asdvsdfbstndfbzzdf'


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/main', methods=['POST', 'GET'])
def Detection():
    if request.method == 'POST':
        print('Check1')

        if request.form.get('Submit') == 'submit':
            f = request.files['image']
            filename = 'temp' + f.filename
            path = os.path.join('temp', filename)
            f.save(path)  # Add to session data the filename. so can be accessed, also delte
            print(f)
            x = cnnn()
            print('Thru')
            _ = x.main(path)  # Needs to be editted
            print(_)

            return redirect(url_for('index'))
        return render_template('Detection.html')
    return render_template('Detection.html')


if __name__ == "__main__":
    app.secret_key = 'asdvsdfbstndfbzzdf'
    app.run(debug=True)
