# 🔮 ChromaClairvoyant

> **Predicting the future (value) of images through the power of AI vision**

ChromaClairvoyant is an advanced image captioning system powered by deep learning. It combines CNN (Convolutional Neural Networks) and RNN (Recurrent Neural Networks) to generate descriptive captions for images, with a focus on extracting text and understanding visual content.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Django](https://img.shields.io/badge/Django-4.0+-green.svg)](https://www.djangoproject.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## ✨ Features

- **🧠 CNN-RNN Architecture**: Inception v3 encoder + LSTM decoder for robust image understanding
- **📝 Text Extraction**: Specialized for OCR and text-in-image captioning
- **🌐 Web Interface**: Django-powered web app for easy image upload and prediction
- **⚡ CLI Support**: Command-line interface for batch processing and automation
- **🎨 Modern UI**: Clean, responsive interface with animations
- **🔄 Real-time Processing**: Instant caption generation from uploaded images

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Input Image    │
│   (299x299)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Inception v3    │  ◄── Pre-trained CNN Encoder
│   Encoder       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LSTM Decoder   │  ◄── Sequence Generation
│  (256 hidden)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Caption Output  │
│  "text here"    │
└─────────────────┘
```

**Model Components:**
- **Encoder**: Inception v3 (pre-trained on ImageNet) → 256-dim embeddings
- **Decoder**: LSTM with 256 hidden units → vocabulary predictions
- **Vocabulary**: Built from dataset annotations with frequency threshold

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ankushsingh003/ChromaClairvoyant.git
   cd ChromaClairvoyant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the database**
   ```bash
   python manage.py migrate
   ```

4. **Download the dataset** (optional, for training)
   - Place your image dataset in `7GB_IMAGE_TO_TEXT_DATASET/`
   - Ensure `img.csv` and `annot.csv` are present

---

## 💻 Usage

### Web Application

Start the Django development server:

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` and upload an image to get instant captions!

### Command Line Interface

Generate captions from the terminal:

```bash
python src/predict.py --image_path "path/to/your/image.jpg"
```

**Optional arguments:**
- `--checkpoint_path`: Path to trained model checkpoint (default: `models/my_checkpoint.pth.tar`)

**Example:**
```bash
python src/predict.py --image_path "test_images/sample.jpg" --checkpoint_path "models/best_model.pth.tar"
```

### Training Your Own Model

```bash
python src/train.py
```

**Training configuration** (edit in `src/train.py`):
- `embed_size`: 256
- `hidden_size`: 256
- `num_layers`: 1
- `learning_rate`: 3e-4
- `num_epochs`: 100

Model checkpoints are saved to `models/` directory.

---

## 📁 Project Structure

```
ChromaClairvoyant/
├── src/
│   ├── model.py          # CNN-RNN architecture
│   ├── dataset.py        # Data loading & vocabulary
│   ├── train.py          # Training script
│   └── predict.py        # CLI prediction tool
├── core/
│   ├── views.py          # Django views
│   └── urls.py           # URL routing
├── templates/
│   └── index.html        # Web UI
├── static/
│   ├── css/              # Stylesheets
│   └── js/               # JavaScript
├── web_project/
│   └── settings.py       # Django settings
├── manage.py             # Django management
├── requirements.txt      # Dependencies
└── README.md            # This file
```

---

## 🎯 Model Details

### Encoder (CNN)
- **Architecture**: Inception v3
- **Pre-training**: ImageNet
- **Output**: 256-dimensional feature vectors
- **Modifications**: Replaced final FC layer, added dropout (0.5)

### Decoder (RNN)
- **Architecture**: LSTM
- **Hidden size**: 256
- **Layers**: 1
- **Embedding**: 256-dimensional word embeddings
- **Dropout**: 0.5

### Vocabulary
- **Special tokens**: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- **Frequency threshold**: 5 (words appearing < 5 times are mapped to `<UNK>`)
- **Tokenization**: Simple whitespace + lowercase

---

## 📊 Dataset

The model is designed to work with the **TextOCR dataset** or similar image-to-text datasets.

**Expected format:**
- `img.csv`: Image metadata (id, file_name)
- `annot.csv`: Annotations (image_id, utf8_string)
- Images in `train_val_images/train_images/`

---

## 🚀 Deployment

### Deploy to Render (Recommended)

Render is the recommended platform for deploying ChromaClairvoyant due to better support for Django applications with ML models.

**Steps:**

1. **Push your code to GitHub** (already done!)

2. **Create a new Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure the service**
   - **Name**: chromaclairvoyant
   - **Environment**: Python 3
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn web_project.wsgi:application`

4. **Set Environment Variables**
   - `SECRET_KEY`: Generate a secure key
   - `DEBUG`: `False`
   - `ALLOWED_HOSTS`: `.onrender.com`
   - `PYTHON_VERSION`: `3.11.0`

5. **Deploy!** Render will automatically build and deploy your app.

**Access your app**: `https://chromaclairvoyant.onrender.com`

### Deploy to Vercel (Limited Support)

> [!WARNING]
> **Vercel has limitations** for this project:
> - 50MB deployment size limit
> - 10-second execution timeout
> - Not ideal for ML models (~100MB PyTorch weights)
> 
> **Use Render instead for best results.**

If you still want to try Vercel:

1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel`
3. Follow the prompts

### Environment Variables

Create a `.env` file (use `.env.example` as template):

```bash
SECRET_KEY=your-super-secret-key-here
DEBUG=False
ALLOWED_HOSTS=.onrender.com,yourdomain.com
```

**Generate a secure SECRET_KEY:**
```python
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### Troubleshooting Deployment

**Issue: "Application failed to start"**
- Check logs for errors
- Ensure all dependencies are in `requirements.txt`
- Verify `build.sh` has execute permissions

**Issue: "Static files not loading"**
- Run `python manage.py collectstatic`
- Check `STATIC_ROOT` and `STATICFILES_STORAGE` settings

**Issue: "Model too large"**
- Consider model quantization
- Use external storage (S3, Google Cloud Storage)
- Load models on-demand instead of at startup

---

## 🛠️ Technologies Used

- **Backend**: Django 4.0+
- **Deep Learning**: PyTorch 2.0+
- **Computer Vision**: torchvision, PIL
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: pandas, numpy

---

## 🔮 Future Enhancements

- [ ] Attention mechanism for improved caption quality
- [ ] Transformer-based decoder (BERT/GPT integration)
- [ ] Multi-language support
- [ ] Beam search for better caption generation
- [ ] REST API for integration
- [ ] Docker containerization
- [ ] Model quantization for mobile deployment

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Ankush Singh**

- GitHub: [@ankushsingh003](https://github.com/ankushsingh003)
- Project Link: [https://github.com/ankushsingh003/ChromaClairvoyant](https://github.com/ankushsingh003/ChromaClairvoyant)

---

## 🙏 Acknowledgments

- Inception v3 architecture from [Google Research](https://arxiv.org/abs/1512.00567)
- Inspired by "Show and Tell: A Neural Image Caption Generator" paper
- TextOCR dataset for training data

---

<div align="center">
  <strong>⭐ Star this repo if you find it useful! ⭐</strong>
</div>
