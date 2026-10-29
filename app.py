from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
import numpy as np
from keras.models import load_model
from huggingface_hub import hf_hub_download
from PIL import Image
from models import db, User
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os

#app = Flask(__name__) 
app = Flask(__name__, static_folder= 'static')
bootstrap = Bootstrap(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

#variaveis
IMG_AL = 150
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img_analizadas')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Carregando o modelo Keras (Local ou via Hugging Face Hub)
MODEL_FILENAME = 'tcc.h5'

if os.path.exists(MODEL_FILENAME):
    caminho_modelo = MODEL_FILENAME
else:
    print("Baixando o modelo tcc.h5 do Hugging Face...")
    caminho_modelo = hf_hub_download(
        repo_id="jonasherminio/resnet152v2-skin-cancer-ham10000",
        filename=MODEL_FILENAME
    )

model = load_model(caminho_modelo)

# Mapeamento das classes
tipos_cancer = {
    0: "Queratoses actínicas (Pré-cancerosas)",
    1: "Carcinoma de células basais (Maligno)",
    2: "Dermatofibroma (Benigno)",
    3: "Lesões benignas semelhantes a queratose (Benigno)",
    4: "Melanoma (Maligno)",
    5: "Nevos melanocíticos (Benigno)",
    6: "Lesões Vasculares (Benigno)"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    image = image.resize((IMG_AL, IMG_AL))
    image = np.array(image)/255  # Normaliza os valores de pixel para o intervalo [0, 1]
    image = image.reshape(IMG_AL,IMG_AL,3)
    return image

def predict(image):
    # Faça as previsões utilizando o modelo
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))[0]
    return prediction

@app.before_request
def create_tables():
    if not os.path.exists('users.db'):
        db.create_all()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            return redirect('http://127.0.0.1:5000/')
        return 'Invalid credentials'
    return render_template('pages/examples/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('pages/examples/register.html')

@app.route('/')
def index():
    return render_template('pages/forms/general.html')

@app.route('/img_analizadas/<path:filename>')
def send_analyzed_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        if request.method == 'POST':
            # Obter a imagem enviada pelo usuário
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath)
            # Fazer a previsão
            prediction = predict(image)
            # Formatar os resultados
            results = []
            # Variavel para exibir a imagem
            img_up = filename
            for i, prob in enumerate(prediction): 
                results.append({
                    'class': tipos_cancer[i],
                    'probability': f'{prob:.2%}',
                    'raw_probability': prob  # Adicionar a probabilidade bruta para ordenação

                })
            results = sorted(results, key=lambda x: x['raw_probability'], reverse=True)
            for result in results:
              del result['raw_probability']

            return render_template('pages/forms/general.html', results=results, img_up=img_up)
    else:
        return 'File not allowed'

if __name__ == '__main__':
    app.run(debug=True)
