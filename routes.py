from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from src.ml.predict import make_prediction
from src.api.errors import handle_error

api_bp = Blueprint('api', __name__)
api = Api(api_bp)

class Predict(Resource):
    def post(self):
        try:
            data = request.get_json()
            if not data or 'text' not in data:
                return handle_error('Invalid input', 400)

            text_input = data['text']
            prediction, probability = make_prediction(text_input)

            return jsonify({
                'prediction': prediction,
                'probability': probability
            })
        except Exception as e:
            return handle_error(str(e), 500)

api.add_resource(Predict, '/predict')