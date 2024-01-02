import torch
import torchvision.transforms.functional as F 
import streamlit as st
from PIL import Image, ImageDraw 
from fovea_model import Net


def load_model():
    model = torch.load('./fovea_model.pt', map_location=torch.device('cpu'))
    model.eval()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)

    return model

def transform_image(image, target_size=(256, 256)):
    image = F.resize(image, target_size)
    image = F.to_tensor(image)

    return image
    
def model_inference(model, image):
    device = next(model.parameters()).device
    pred = model(image.unsqueeze(0).to(device))[0].cpu()
    return pred

def rescale_label(label, target_size):
    scaled_label = [ai*bi for ai, bi in zip(label, target_size)]
    return scaled_label


def draw_boundary_box(image, label, w_h=(25, 25)):
    label = rescale_label(label, image.shape[1:])
    image = F.to_pil_image(image)

    w, h = w_h
    cx, cy = label
    draw = ImageDraw.Draw(image)
    draw.rectangle(((cx-w, cy-h), (cx+w, cy+h)), outline='green', width=2)

    return image
    
    
def main():
    st.title('Fovea Localization ...')
    uploaded_file = st.file_uploader('Upload Eye Image .', type=('png', 'jpg', 'jpeg'))

    if uploaded_file is not None: 
        image = Image.open(uploaded_file)
    
        t_image = transform_image(image)
        pred_label = model_inference(model, t_image)
        pred_image = draw_boundary_box(t_image, pred_label)
        
        st.image([F.to_pil_image(t_image), pred_image], caption=['Uploaded Image', 'Fovea Location'], width=256)

        

if __name__ == '__main__':
    model = load_model()
    main()
