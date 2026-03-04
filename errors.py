from flask import jsonify

class APIError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.message = message

    def to_dict(self):
        return {'error': self.message}

def handle_api_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

def handle_not_found_error(error):
    response = jsonify({'error': 'Resource not found'})
    response.status_code = 404
    return response

def handle_internal_server_error(error):
    response = jsonify({'error': 'Internal server error'})
    response.status_code = 500
    return response