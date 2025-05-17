from flask import Flask, jsonify
from flask_cors import CORS
import io
import sys
from tema1 import run_all

app = Flask(__name__)
CORS(app)

@app.route('/tema1-output')
def get_tema1_output():
    buffer = io.StringIO()
    sys.stdout = buffer

    run_all()  # Calls all exercises

    sys.stdout = sys.__stdout__
    output = buffer.getvalue()
    return jsonify({"output": output})
if __name__ == "__main__":
    app.run(debug=True)