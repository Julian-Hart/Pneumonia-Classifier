import base64
import json
import io
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import runpod

model = models.resnet18(weighs=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load('src/pneumonia_classifier.pth', map_location=torch.device('cpu'), weights_only=False))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['NORMAL', 'PNEUMONIA']

def validate_input(job_input):
    if job_input is None:
        return None, "Please provide an image input."
    
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON input."
        
    image_data = job_input.get('image')
    if image_data is None:
        return None, "Please provide an image input."
    
    if not isinstance(image_data, str):
        return None, "Invalid image input. Expected a base64-encoded string."
    
    return {
        'image': image_data
    }, None

def handler(job):
   
    job_input = job['input']

    validated_input, error_message = validate_input(job_input)

    if error_message:
        return {
            'error': error_message
            }
    
    image_base64 = validated_input['image']
    
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        predicted_class = class_names[predicted.item()]

        return {"prediction": predicted_class}
    
    except base64.binascii.Error:
        return {
            'error': 'Invalid base64-encoded string.'
        }
    except IOError:
        return {
            'error': 'Invalid image format.'
        }
    
    except Exception as e:
        return {
            'error': str(e)
        }
    
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

