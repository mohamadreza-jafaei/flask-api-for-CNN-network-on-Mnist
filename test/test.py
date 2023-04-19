import requests
session = requests.Session()
session.trust_env = False
resp = session.post("http://localhost:5000/predict", files={'image': open('two.png', 'rb')})

print(resp.text)