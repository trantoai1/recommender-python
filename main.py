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
        '<p>You can read <a href="docs" target="_blank">docs</a> to use</p>' \
        '<h1>Update 20/4</h1><br>' \
        '<p>Cập nhật API giữa kì, chi tiết xem <a href="docs" target="_blank">dataproperty</a></p>' \
        '<h1>Update 18/4</h1><br>' \
        '<h2>================</h2><br>' \
        '<p>-Customer thêm trường isAdmin (bool) để kiểm tra Admin </p>' \
        '<p>-Check các khóa ngoại của Food-Foodtype, Order-Customer, Order-Food </p>' \
        '<p>-Status response check khóa ngoại:{ "status": -4, "message": "Foreign key error", "data": { "field": "customerid", "value": "098502392p" } } </p>' \
        '<h2>================</h2><br>' \
        '<p>-Cập nhật api_key: ở header, api_key=123456, nếu không có api_key thì status reponse =401</p>' \
        '<p>-ORDER: Thêm trường confirm(bool) là trạng thái đặt hàng </p>' \
        '<p>-Mẫu JSON ORDER: {"customerid":"0985023919","confirm":false,"ordertime":"2021-03-12T19:25:43","totalprice":500000.0,"listfood":[{"foodid":"12","amount":1},{"foodid":"13","amount":3}]}</p>' \
        '<h1>Update 17/4</h1><br>' \
        '<p>-Method GET: {\"status\": 0,\"message\": \"Success\",\"data\":{danh sách hoặc 1 object}}</p>' \
        '<p>-Method GET: {\"status\": -1,\"message\": \"Không tìm thấy dữ liệu\"}</p>' \
        '<p>-Method POST: { "status": -2, "message": "Missing agr", "data": { "field": " tên cột dữ liệu lỗi" } }</p>' \
        '<p>-Method POST: { "status": -3, "message": "Validation fail (lỗi xác thực dữ liệu)", "data": { "field": " tên cột dữ liệu lỗi" } }</p>' \
        '<h1>Update 16/4</h1><br>' \
        '<p>-Method POST order: giảm bớt trường _id(xử lý trên API)</p>' \
        ''


@app.route('/recommender', methods=['GET'])
def getProcducts():
    productName = request.args.get('productName')
    if productName is None:
        return Response([], mimetype='application/json')
    return Response(make_recommendation(productName), mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)
