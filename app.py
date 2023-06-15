import requests
from io import BytesIO
import streamlit as st
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
import random
import pandas as pd
from datetime import datetime
import json

@st.cache_resource
def load_model():
    euler_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    model_id = "darkstorm2150/Protogen_x3.4_Official_Release"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler=euler_scheduler, safety_checker=None)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe

def main():
    st.title('Image Generation from Text')
    pipe = load_model()
    
    h_px = st.slider("Height", min_value=16, max_value=2000, value=1024, step=8)
    w_px = st.slider("Width", min_value=16, max_value=2000, value=768, step=8)
    steps = st.slider("Steps", min_value=1, max_value=100, value=25, step=1)
    cfg = st.slider("Guidance", min_value=1.0, max_value=30.0, value=7.5, step=0.5)

    st.header('Prompt Generator for Stable Diffusion')

    base_prompt = st.text_area('Enter the base prompt:', 'An upscaled photorealistc close up portrait of an (18-year-old:3.5) anime version of ({person}:3.4) {action} (((alone))) in front of a ({background_scene}:3.0) wearing a ({outfit}:3.3), ((sharp focus)), 8k')
    
    neg = st.text_area('Enter the negative Prompt:', 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art')

    # Define lists of options for each dropdown
    persons = ['Elisha Cuthbert']
    outfits = ['sweater']
    background_scenes = ['mountain']
    shots = ['close up']

    expressions = ['smiling', 'frowning', 'laughing', 'crying', 'screaming', 'yawning', 'grinning', 'blinking', 'talking', 'gazing']
    pose = ['standing', 'sitting', 'kneeling', 'lying', 'jumping', 'leaning', 'laying down', 'posing', 'looking away', 'stretching']
    
    if 'person' not in st.session_state:
        st.session_state.person = random.choice(persons)
    person = st.text_area('Person:', st.session_state.person)
    
    if st.button('Generate Prompt'):
        def generate_prompt(base_prompt, shot, person, facial_expression, action, outfit, background_scene):
            return base_prompt.format(shot=shot, person=person, facial_expression=facial_expression, action=action, outfit=outfit, background_scene=background_scene)
        
        with st.spinner('Generating Image...'):
            n_imgs = 1
            for i in range(n_imgs):           
                facial_expression = random.choice(expressions)
                action = random.choice(pose)
                outfit = random.choice(outfits)
                background_scene = random.choice(background_scenes)
                shot = random.choice(shots)
                prompt_string = generate_prompt(base_prompt, shot, person, facial_expression, action, outfit, background_scene)

                # Display info
                # directory_path = "/users/nateh/downloads/images/"
                # filename = f'{directory_path}{person}_{h_px}x{w_px}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                st.write(prompt_string)
                # st.write(filename)

                # Generate Image
                image = pipe(prompt_string, negative_prompt=neg, height=h_px, width=w_px, num_inference_steps=steps, guidance_scale=cfg).images[0]
            
                # Display image in Streamlit
                st.image(image)

                # Save Images
                # image.save(filename)

                # r = requests.post(
                # "https://api.deepai.org/api/torch-srgan", 
                # files={'image': open(f'{filename}', 'rb'),},
                # headers={'api-key': 'fa90ea95-5041-4735-bb71-e5a4fc20d475'})
                # url = (r.json())['output_url']
                # response = requests.get(url)
                # upscaled = Image.open(BytesIO(response.content)).convert("RGB")
                # st.image(upscaled)
                # filename2 = f'{directory_path}{person}_{h_px}x{w_px}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                # upscaled.save(filename2)

if __name__ == '__main__':
    main()
