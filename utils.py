def log_error(error_message):
    # Function to log error messages
    print(f"ERROR: {error_message}")

def format_response(prediction, probability):
    # Function to format the response
    return {
        "prediction": prediction,
        "probability": probability
    }