# Email Reply Classifier

## Setup Instructions

1. Create virtual environment:
   ```
   python -m venv venv
   ```

2. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the API:
   ```
   python api.py
   ```

5. Open your browser and go to: `http://localhost:8000/docs`

6. Click on "POST /predict", then "Try it out"

7. Enter email text like: `{"text": "Looking forward to the demo!"}`

8. Click "Execute" to get prediction (positive/negative/neutral) and confidence score

## Other ways to test:

Using Python (run `python` first, then enter these commands):
```python
import requests
response = requests.post("http://localhost:8000/predict", 
                        json={"text": "Looking forward to the demo!"})
print(response.json())
```

Or create a file `test_api.py` with the above code and run: `python test_api.py`

