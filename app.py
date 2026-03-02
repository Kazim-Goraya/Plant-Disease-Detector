import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ── Disease info database ─────────────────────
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'cure': 'Apply fungicides containing captan or myclobutanil. Remove infected leaves.',
        'severity': 'Moderate'
    },
    'Apple___Black_rot': {
        'cure': 'Prune infected branches. Apply copper-based fungicide.',
        'severity': 'High'
    },
    'Apple___Cedar_apple_rust': {
        'cure': 'Apply fungicide in spring. Remove nearby juniper trees if possible.',
        'severity': 'Moderate'
    },
    'Apple___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Blueberry___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'cure': 'Apply sulfur-based fungicide. Improve air circulation.',
        'severity': 'Moderate'
    },
    'Cherry_(including_sour)___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'cure': 'Use resistant varieties. Apply strobilurin fungicides.',
        'severity': 'High'
    },
    'Corn_(maize)___Common_rust_': {
        'cure': 'Apply fungicides early. Use rust-resistant hybrids.',
        'severity': 'Moderate'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'cure': 'Apply fungicide at tasseling stage. Use resistant varieties.',
        'severity': 'High'
    },
    'Corn_(maize)___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Grape___Black_rot': {
        'cure': 'Apply mancozeb or myclobutanil. Remove mummified fruit.',
        'severity': 'High'
    },
    'Grape___Esca_(Black_Measles)': {
        'cure': 'Prune infected wood. Apply wound sealants after pruning.',
        'severity': 'High'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'cure': 'Apply copper fungicide. Ensure good drainage.',
        'severity': 'Moderate'
    },
    'Grape___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'cure': 'No cure available. Remove infected trees to prevent spread.',
        'severity': 'Critical'
    },
    'Peach___Bacterial_spot': {
        'cure': 'Apply copper sprays during dormancy. Use resistant varieties.',
        'severity': 'High'
    },
    'Peach___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Pepper,_bell___Bacterial_spot': {
        'cure': 'Apply copper bactericide. Avoid overhead irrigation.',
        'severity': 'Moderate'
    },
    'Pepper,_bell___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Potato___Early_blight': {
        'cure': 'Apply chlorothalonil fungicide. Remove infected leaves promptly.',
        'severity': 'Moderate'
    },
    'Potato___Late_blight': {
        'cure': 'Apply metalaxyl fungicide immediately. Destroy infected plants.',
        'severity': 'Critical'
    },
    'Potato___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Raspberry___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Soybean___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Squash___Powdery_mildew': {
        'cure': 'Apply potassium bicarbonate or neem oil spray.',
        'severity': 'Moderate'
    },
    'Strawberry___Leaf_scorch': {
        'cure': 'Apply captan fungicide. Remove infected leaves.',
        'severity': 'Moderate'
    },
    'Strawberry___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
    'Tomato___Bacterial_spot': {
        'cure': 'Apply copper bactericide. Use disease-free seeds.',
        'severity': 'High'
    },
    'Tomato___Early_blight': {
        'cure': 'Apply chlorothalonil or mancozeb. Rotate crops annually.',
        'severity': 'Moderate'
    },
    'Tomato___Late_blight': {
        'cure': 'Apply metalaxyl or chlorothalonil immediately. Remove infected plants.',
        'severity': 'Critical'
    },
    'Tomato___Leaf_Mold': {
        'cure': 'Improve ventilation. Apply copper fungicide.',
        'severity': 'Moderate'
    },
    'Tomato___Septoria_leaf_spot': {
        'cure': 'Apply fungicide containing chlorothalonil. Remove lower leaves.',
        'severity': 'Moderate'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'cure': 'Apply miticide or neem oil. Increase humidity around plants.',
        'severity': 'Moderate'
    },
    'Tomato___Target_Spot': {
        'cure': 'Apply chlorothalonil fungicide. Avoid overhead watering.',
        'severity': 'Moderate'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'cure': 'No cure. Control whitefly vectors. Remove infected plants.',
        'severity': 'Critical'
    },
    'Tomato___Tomato_mosaic_virus': {
        'cure': 'No cure. Remove infected plants. Disinfect tools regularly.',
        'severity': 'High'
    },
    'Tomato___healthy': {
        'cure': 'No treatment needed. Plant is healthy!',
        'severity': 'None'
    },
}

SEVERITY_COLOR = {
    'None':     '🟢',
    'Moderate': '🟡',
    'High':     '🟠',
    'Critical': '🔴',
}

# ── Load model (cached) ───────────────────────
@st.cache_resource
def load_model():
    MODEL_PATH = r'C:\Users\goray\Desktop\FYP\PDS\plant_disease_model.pth'
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes  = len(class_to_idx)

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, idx_to_class, device

# ── Transform ─────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── UI ────────────────────────────────────────
st.title('🌿 Plant Disease Detector')
st.markdown('**Final Year Project** | ResNet50 | 38 Classes | 90.87% Accuracy')
st.markdown('---')

# Load model
with st.spinner('Loading AI model...'):
    model, idx_to_class, device = load_model()
st.success('✅ Model loaded and ready!')

# Upload image
st.markdown('### 📷 Upload a Plant Leaf Image')
uploaded = st.file_uploader(
    'Supported formats: JPG, PNG, JPEG',
    type=['jpg', 'jpeg', 'png']
)

if uploaded:
    img = Image.open(uploaded).convert('RGB')

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption='Uploaded Image', use_container_width=True)

    # Predict
    with st.spinner('Analyzing...'):
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)[0]
            top5   = probs.topk(5)

    top_label = idx_to_class[top5.indices[0].item()]
    top_prob  = top5.values[0].item() * 100
    info      = DISEASE_INFO.get(top_label, {'cure': 'Consult an expert.', 'severity': 'Unknown'})
    severity  = info['severity']
    sev_icon  = SEVERITY_COLOR.get(severity, '⚪')

    with col2:
        st.markdown('### 🔍 Detection Result')
        st.markdown(f'**Disease:** `{top_label.replace("_", " ")}`')
        st.markdown(f'**Confidence:** `{top_prob:.1f}%`')
        st.markdown(f'**Severity:** {sev_icon} `{severity}`')
        st.progress(int(top_prob))

    st.markdown('---')

    # Treatment
    st.markdown('### 💊 Recommended Treatment')
    if severity == 'None':
        st.success(info['cure'])
    elif severity == 'Critical':
        st.error(f"⚠️ CRITICAL: {info['cure']}")
    elif severity == 'High':
        st.warning(info['cure'])
    else:
        st.info(info['cure'])

    # Top 5 chart
    st.markdown('### 📊 Top 5 Predictions')
    labels = [idx_to_class[i.item()].replace('___', '\n').replace('_', ' ')
              for i in top5.indices]
    values = [v.item() * 100 for v in top5.values]
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(5)]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel('Confidence (%)')
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info('👆 Upload a leaf image to get started')
    st.markdown('''
    **Supported Plants:**
    Apple, Blueberry, Cherry, Corn, Grape, Orange,
    Peach, Pepper, Potato, Raspberry, Soybean,
    Squash, Strawberry, Tomato
    ''')

st.markdown('---')
st.markdown('*Built with PyTorch + Streamlit | FYP 2025*')
