# 🔬 Ferramenta Web para Auxílio ao Diagnóstico do Câncer de Pele com Deep Learning

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Keras](https://img.shields.io/badge/Keras-2.15.0-red)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Hub%20Integrated-yellow)
![Springer](https://img.shields.io/badge/Springer-Published%20Paper-success)

Este repositório contém o código-fonte da **Interface Web (Flask + Autenticação de Usuários)** integrada ao modelo de Deep Learning (`ResNet152V2`) para auxílio na análise e classificação de lesões dermatoscópicas de câncer de pele.

* **Autor:** Jonas Alves dos Santos Herminio
* **Instituição:** Centro Universitário Christus (Unichristus)
* **Modelo no Hugging Face:** [🤗 jonasherminio/resnet152v2-skin-cancer-ham10000](https://huggingface.co/jonasherminio/resnet152v2-skin-cancer-ham10000)

---

## 📄 Publicação Científica Internacional (Springer)

Esta pesquisa foi publicada internacionalmente pela editora **Springer** sob o título:
> **"Adoption of Deep Learning to Assist the Diagnosis of Skin Cancer"**

Para maiores informações metodológicas, análises estatísticas e detalhes científicos do estudo, acesse o capítulo oficial:
🔗 **[Leia o artigo completo na Springer](https://link.springer.com/chapter/10.1007/978-3-032-04725-0_26)**

---

## 💻 Sobre a Aplicação Web

O diagnóstico precoce do câncer de pele é determinante para o sucesso do tratamento. Esta plataforma web foi construída com foco em usabilidade, rapidez e baixa curva de aprendizado para profissionais da saúde:

* **Sistema de Autenticação:** Controle de registro e login de usuários com banco de dados SQLite (`Flask-SQLAlchemy`) e criptografia de senhas (`Werkzeug`).
* **Fluxo Direto de Classificação:** Interface intuitiva que permite o upload imediato de imagens dermatoscópicas (`.png`, `.jpg`, `.jpeg`).
* **Alta Velocidade de Resposta:** O backend pré-processa a imagem e realiza a inferência na rede neural encapsulada (`tcc.h5`), retornando as probabilidades ordenadas em **1,17 segundos em média**.

---

## 🧠 Sobre o Modelo de Inteligência Artificial (ResNet152V2)

O motor de classificação utiliza **Transfer Learning** a partir da arquitetura **ResNet152V2** (acrescida de camadas densas de 128 e 7 neurônios com ativação `softmax`), treinada por 20 épocas (otimizador SGD com *learning rate* de `0.001`, *Dropout*, *Early Stopping* e *Reduce LR*) sobre o dataset **HAM10000** (*Harvard Dataverse* — 10.015 imagens dermatoscópicas).

| Rede Avaliada | Acurácia (Teste) | Loss |
| :--- | :---: | :---: |
| **ResNet152V2 (Implementada)** | **99,01%** | **0.02** |
| Xception | 99,00% | 0.05 |
| EfficientNet-B7 | 98,00% | 0.07 |

### Classes Identificadas pelo Sistema:
1. **Nevos melanocíticos** *(Benigno)*
2. **Queratoses actínicas / Doença de Bowen** *(Pré-cancerosa)*
3. **Carcinoma de células basais** *(Maligno)*
4. **Dermatofibroma** *(Benigno)*
5. **Lesões benignas semelhantes a queratose** *(Benigno)*
6. **Melanoma** *(Maligno)*
7. **Lesões vasculares** *(Benigno)*

---

## 🛠️ Tecnologias Utilizadas

* **Backend:** Python, Flask, Flask-SQLAlchemy, Werkzeug
* **Frontend:** HTML5, CSS3, Bootstrap (Flask-Bootstrap)
* **Deep Learning & MLOps:** TensorFlow (`2.15.0`), Keras (`2.15.0`), Hugging Face Hub (`huggingface-hub`), Scikit-learn, Pillow (PIL), NumPy

---

## 🤗 Integração Automática do Modelo (`tcc.h5`) via Hugging Face

Devido ao tamanho do modelo treinado (**~209 MB**) ultrapassar o limite do GitHub (100 MB), os pesos da rede neural estão hospedados no **Hugging Face**.

* **Download Automático (Zero-Config):** O sistema conta com integração automática via `huggingface_hub`: ao executar o `app.py` pela primeira vez, o código baixa automaticamente o modelo `tcc.h5` do Hugging Face caso ele não esteja presente na pasta local.
* **Download Manual (Opcional):** Caso prefira baixar o arquivo manualmente antes de iniciar, **[📥 clique aqui para baixar o `tcc.h5` (209 MB)](https://huggingface.co/jonasherminio/resnet152v2-skin-cancer-ham10000/resolve/main/tcc.h5?download=true)** e salve-o na raiz do projeto.

---

## 🚀 Como Executar o Projeto Localmente

### Por que utilizar um Ambiente Virtual (`.venv`)?
Este projeto requer versões específicas do **TensorFlow (`2.15.0`)** e do **Keras (`2.15.0`)** para garantir compatibilidade com o arquivo `tcc.h5`. O uso do `.venv` cria um ambiente isolado exclusivo para o projeto, evitando conflitos com pacotes globais e protegendo o Python nativo do sistema operacional (especialmente no macOS e Linux).

### Passo a Passo de Instalação:

#### 1. Clone este repositório
```bash
git clone [https://github.com/jonasherminio/resnet152v2-skin-cancer-ham10000.git](https://github.com/jonasherminio/resnet152v2-skin-cancer-ham10000.git)
cd resnet152v2-skin-cancer-ham10000
```

#### 2. Crie e ative o ambiente virtual (`.venv`)

**No macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**No Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```
*(Confirme se o prefixo `(.venv)` apareceu no início da linha do seu terminal).*

#### 3. Instale todas as dependências via `requirements.txt`
O arquivo `requirements.txt` instala automaticamente o Flask, o TensorFlow/Keras e o integrador do Hugging Face:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Inicie a aplicação
```bash
python app.py
```
*(Na primeira execução, aguarde o download automático do modelo `tcc.h5` via Hugging Face Hub).*

#### 5. Acesse no navegador
Abra o navegador em:
```text
[http://127.0.0.1:5000](http://127.0.0.1:5000)
```

---

## ⚠️ Limitações e Aviso Legal (Disclaimer)

A ferramenta proposta apresenta potencial para auxiliar na triagem e no diagnóstico precoce do câncer de pele, contribuindo para a redução de erros humanos e otimização de recursos na saúde. Contudo, destacam-se limitações importantes:

1. **Generalização Populacional:** É necessária a ampliação contínua de datasets para abranger diferentes perfis demográficos e fototipos cutâneos além dos presentes no HAM10000.
2. **Validação Clínica:** A validação presencial por médicos dermatologistas é indispensável para atestar a confiabilidade em cenários reais.
3. **Regulamentação Sanitária:** Sistemas médicos baseados em IA exigem aprovação formal por órgãos regulatórios, como a **ANVISA**, antes de qualquer aplicação clínica.

> **Aviso Importante:** Este software foi desenvolvido estritamente para **fins acadêmicos, científicos e de portfólio**. Os resultados gerados pela plataforma **não configuram diagnóstico médico** e jamais substituem a avaliação presencial de um dermatologista.
