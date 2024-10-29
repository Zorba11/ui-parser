import requests
import json

def test_api():
    # Make sure this URL matches exactly what Modal provides in your dashboard
    # It should look something like: https://{username}--{app-name}-{function-name}.modal.run
    url = "https://zorba11--ui-coordinates-finder-fastapi-app.modal.run/process"
    
    headers = {
        'Accept': 'application/json',
    }
    
    try:
        files = {
            'file': ('screen-1.png', open('/Users/zorba11/Desktop/screen-1.png', 'rb'), 'image/png')
        }
        
        # Make the request
        response = requests.post(
            url,
            files=files,
            headers=headers
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Content: {response.content.decode()}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api()