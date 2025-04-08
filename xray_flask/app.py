import base64
import time
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        file = request.files['image']
        image_data = base64.b64encode(file.read()).decode('utf-8')

        headers = {
            "Authorization": open('API_KEY', 'r').read(),
        }

        res = requests.post('https://api.runpod.ai/v2/mfcx7jx0ou7xl2/run',
                           json = {'input': {'image': image_data}},
                           headers=headers)

        res_id = res.json()['id']

        for _ in range(10):
            status_res = requests.get(f'https://api.runpod.ai/v2/mfcx7jx0ou7xl2/status/{res_id}', headers=headers)
            status = status_res.json()

            if status.get('status').lower() == 'completed':
                prediction = status['output']['prediction']
                break

            time.sleep(2)
    
        return render_template('index.html', original_image=f'data:image/jpeg;base64,{image_data}', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)