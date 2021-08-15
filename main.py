from flask import Flask, request, Response
from flask_restful import Api, reqparse
from flask_cors import CORS, cross_origin
from knn_new import make_recommendation

app = Flask(__name__)

api = Api(app)
cors = CORS(app)


@app.route('/', methods=['GET'])
def home():
    return \
        'recommender for webapp of Tran Oanh Toai - N17DCCN155'



@app.route('/recommender', methods=['GET'])
def getProcducts():
    productName = request.args.get('productName')
    if productName is None:
        return Response([], mimetype='application/json')
    return Response(make_recommendation(productName), mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)
