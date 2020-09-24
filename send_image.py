import requests

url = 'http://127.0.0.1:5000/'
file = {'image': open('C:/Users/daoha/Downloads/bien_xe1.jpg', 'rb')}
headers = {"Content-Type": "multipart/form-data"}

response = requests.request("POST", url, files=file, headers=headers)

print(response.text)

""" curl -i -X POST -H "Content-Type: multipart/form-data" -F "file=@/Users/daoha/Downloads/bien_xe1.jpg" http://localhost:5000/ """
