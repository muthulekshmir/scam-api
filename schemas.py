from marshmallow import Schema, fields, validate

class PredictRequestSchema(Schema):
    text = fields.String(required=True, validate=validate.Length(min=1))

class PredictResponseSchema(Schema):
    prediction = fields.String(required=True)
    probability = fields.Float(required=True)