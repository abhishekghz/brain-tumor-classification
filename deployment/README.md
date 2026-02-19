# Brain Tumor MRI Classifier - Streamlit App

This Streamlit app provides a web interface for brain tumor classification from MRI images using the trained model artifact (ViT or ResNet50).

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
deploy_model_filename = "best_model_resnet.pth"
model_cloud_url = "https://huggingface.co/<user>/<repo>/resolve/main/best_model_resnet.pth"
# Optional for private Hugging Face repos
hf_token = "hf_xxx"
```

The app first checks local checkpoints in this order:
- `models/$DEPLOY_MODEL_FILENAME` (default: `best_model_resnet.pth`)
- configured model from `src/config.py`
- `models/best_model_resnet.pth`
- `models/best_model_vit.pth`
- `models/best_model.pth`

If none are present locally, it downloads from `model_cloud_url`.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set admin password
export ADMIN_PASSWORD="your-password"
export DEPLOY_MODEL_FILENAME="best_model_resnet.pth"
export MODEL_CLOUD_URL="https://huggingface.co/<user>/<repo>/resolve/main/best_model_resnet.pth"
# Optional for private repos
export HF_TOKEN="hf_xxx"

# Run the app
streamlit run deployment/gui_app.py
```

## Model

The app uses the first available model from the local fallback list above (or downloads from cloud), and supports runtime override via `DEPLOY_MODEL_FILENAME`.

Current default deployment model is `best_model_resnet.pth`.

The classifier predicts four tumor types:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor
