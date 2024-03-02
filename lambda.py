import json
import urllib.request
import urllib.parse

def lambda_handler(event, context):
    
    try:
        url = "http://18.158.59.37/predict"
        print(event['queryStringParameters']['image'])
        data = {'image_url': event['queryStringParameters']['image']}
        data = json.dumps(data)
        data = data.encode('utf-8')
        
        req = urllib.request.Request(url, data, headers={'Content-Type': 'application/json'})
        response = urllib.request.urlopen(req)
        response_body = response.read().decode('utf-8')
        # event['queryStringParameters']['image']
        return {
            'statusCode': 200,
            'body': json.dumps(response_body)
        }
    except:
        return {
            'statusCode': 400,
            'body': json.dumps("Error")
        }
