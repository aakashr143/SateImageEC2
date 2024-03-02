import urllib.request

print("Starting")
urllib.request.urlretrieve("https://satellite-image-classification.s3.eu-central-1.amazonaws.com/model.pth", f"model.pth")
print("Downloaded")