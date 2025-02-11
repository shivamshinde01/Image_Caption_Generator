import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageFilter
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Set up Google Gemini API Key
genai.configure(api_key="AIzaSyBTR3MmV7TiGPznME9B5CODfOkyVb2_sa0")

# Load BLIP model for image description
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Streamlit UI
st.set_page_config(page_title="AI Caption Generator", layout="wide")
st.title("🖼️ AI-Powered Image Caption Generator")
st.write("Upload an image, enter a prompt, apply filters, enhance resolution, and get AI-generated captions for different platforms.")

# Image Upload
uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

# User Prompt Input
user_prompt = st.text_area("✍️ Enter a short description or theme for the image (Optional)")

# Platform Selection with Dropdown
st.subheader("📱 Select Platform for Caption Generation")
platform = st.selectbox("Choose a platform:", ["Instagram", "Facebook", "Twitter", "LinkedIn"], index=0)

# Layout for responsiveness
col1, col2 = st.columns(2)

with col1:
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.sidebar.header("🎨 Image Editing Options")
    filter_option = st.sidebar.selectbox("🖌 Apply Filter", ["None", "Black & White", "HDR", "Sepia", "Blur", "Sharpen"])
    enhance_option = st.sidebar.selectbox("🔍 Enhance Image Quality", ["None", "AI Resolution Upscale"])
    brightness = st.sidebar.slider("🌞 Adjust Brightness", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("🎭 Adjust Contrast", 0.5, 2.0, 1.0)
    sharpness = st.sidebar.slider("🔎 Adjust Sharpness", 0.5, 2.0, 1.0)

# Function to Apply Filters
def apply_filters(image, filter_option):
    if filter_option == "Black & White":
        image = image.convert("L")
    elif filter_option == "HDR":
        image = image.filter(ImageFilter.DETAIL)
    elif filter_option == "Sepia":
        sepia = ImageEnhance.Color(image).enhance(0.3)
        image = sepia.convert("RGB")
    elif filter_option == "Blur":
        image = image.filter(ImageFilter.GaussianBlur(2))
    elif filter_option == "Sharpen":
        image = image.filter(ImageFilter.SHARPEN)
    return image

# Function to Generate Image Description using BLIP
def generate_image_description(image):
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

if uploaded_image:
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)
    image = apply_filters(image, filter_option)
    
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    st.sidebar.download_button("📥 Download Edited Image", img_byte_arr, file_name="edited_image.png", mime="image/png")
    
    image_description = generate_image_description(image)
    
    if platform:
        caption_prompt = f"""Generate exactly 2 short captions (under 10 words) and 2 long captions (1-2 sentences) for the following image:
        
        Image Description: '{image_description}'
        Platform: {platform}
        User Input: {user_prompt if user_prompt else "No additional context"}

        ### Output Format ###
        Short Captions:
        1. [First short caption]
        2. [Second short caption]

        Long Captions:
        1. [First long caption]
        2. [Second long caption]
        """

        ai_response = genai.GenerativeModel("gemini-pro").generate_content(caption_prompt)
        
        if ai_response and ai_response.text:
            captions = ai_response.text.strip().split("\n")
            
            short_captions = [line[3:].strip() for line in captions if line.startswith("1.") or line.startswith("2.")][:2]
            long_captions = [line[3:].strip() for line in captions if line.startswith("1.") or line.startswith("2.")][2:]

            short_caption_1 = short_captions[0] if len(short_captions) > 0 else "N/A"
            short_caption_2 = short_captions[1] if len(short_captions) > 1 else "N/A"
            long_caption_1 = long_captions[0] if len(long_captions) > 0 else "N/A"
            long_caption_2 = long_captions[1] if len(long_captions) > 1 else "N/A"

            st.subheader("📝 AI-Generated Captions")
            st.write("**Short Captions:**")
            st.write(f"1. {short_caption_1}")
            st.write(f"2. {short_caption_2}")
            st.write("**Long Captions:**")
            st.write(f"1. {long_caption_1}")
            st.write(f"2. {long_caption_2}")
    
    with col2:
        st.image(image, caption="Processed Image", use_column_width=True)
