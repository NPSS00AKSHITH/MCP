import requests

def test_remote_access():
    url = "http://localhost:8000/health"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Successfully connected to Docker container!")
            print("Response:", response.json())
        else:
            print(f"Failed to connect. Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error connecting: {e}")

if __name__ == "__main__":
    test_remote_access()
