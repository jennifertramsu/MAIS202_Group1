import os
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)   # Tells the instance where it is located
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')         # Tells the instance what URL should trigger the function
# def hello():            # The decorate converts the return value into an HTTP response
#     return 'Hello, World!'

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTTP methods tell the server what to do with the data
# POST methods sends data to the server, connected to the HTML form through the method attribute
@app.route('/', methods = ['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return render_template('index.html', msg='No file selected, try again!')
        if allowed_file(f.filename):
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(path)
            return render_template('upload.html', name = filename)
        