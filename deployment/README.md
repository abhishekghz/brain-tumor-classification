# Brain Tumor MRI Classifier - Streamlit App

This Streamlit app provides a web interface for brain tumor classification from MRI images using a ResNet50 deep learning model.

## Features

- **Predict**: Upload an MRI image and get tumor classification with confidence score
- **Admin Panel**: Review uploaded images, label them, and retrain the model
- **Continual Learning**: Model improves over time as new labeled data is added

## Deployment on Streamlit Community Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository, branch (main), and main file path: `deployment/gui_app.py`
6. Click "Deploy"

### Setting Admin Password

In Streamlit Cloud settings, add a secret:
```toml
admin_password = "your-secure-password"
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set admin password
export ADMIN_PASSWORD="your-password"

# Run the app
streamlit run deployment/gui_app.py
```

## Model

The app uses a ResNet50 model trained on brain MRI images to classify four tumor types:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

Accuracy: ~93% on validation data
