from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS, cross_origin

# creates instanceof flask
app = Flask(__name__)
cors = CORS(app)

# this is a decorator that tells flask which url is the endpoint to run the below function
@app.route('/hello', methods=['POST']) # specifies which kind of http request is allowed for the end point
#@cross_origin(origin='localhost',headers=['Content-Type','Authorization'])
def hello():
    message = request.get_json(force=True)
    name = message['name'] # get the name from the json object
    response = {
        'greeting': 'Hello ' + name + '!'
    }
    return jsonify(response)