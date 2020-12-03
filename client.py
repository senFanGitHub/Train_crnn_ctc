import urllib.request ## 只import urllib 会报错
import json
import base64
import cv2
import time

class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        """
        只要检查到了是bytes类型的数据就把它转为str类型
        :param obj:
        :return:
        """
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)
    
    
class RecClient(object):
    def __init__(self, url):
        self.url = url

    def getBase64(self, filename):
        with open(filename, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()) ## for python3 need decode()
        return encoded_string

    def request(self, api_func, request_data):
        url_func = "%s/api/%s" % (self.url, api_func)
        print(url_func)
        data=json.dumps(request_data,cls=MyEncoder)
        
        req = urllib.request.Request(url=url_func, data=data.encode('UTF8'), headers={'content-type': 'application/json'})
  
        res = urllib.request.urlopen(req)
  

        
        return res.read()

    def recognize(self, filename,key='key'):
        base64Image = self.getBase64(filename)
        json_data = {"image": base64Image,"key":key}

        api_result = self.request("recognize", json_data)

        return json.loads(api_result)

    
if __name__ == '__main__':
    from argparse import ArgumentParser
       
    parser = ArgumentParser()
    parser.add_argument("-i", "--img", action="store", dest="img", default='demo.jpg', 
        help="test img.", required=False)
    parser.add_argument("-d", "--host", action="store", dest="host", default='0.0.0.0:51134', 
        help="test img.", required=False)
    
    parser.print_help()
    args = parser.parse_args()
     
    host =f"http://{args.host}"    
    recClient = RecClient(host)

    A=time.time()
    for i in range(50):
        print(recClient.recognize(args.img))
    B=time.time()
    print(f"time:{(B-A)/50}")

    
