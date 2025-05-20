from flask import Flask, jsonify, request
from flask_cors import CORS
import io
import sys

import numpy as np
from tema1 import run_all
from tema2 import example_custom_diagonals
from tema3 import tema3
from tema4 import main

app = Flask(__name__)
CORS(app)

@app.route('/tema1-output')
def get_tema1_output():
    buffer = io.StringIO()
    sys.stdout = buffer

    run_all()  

    sys.stdout = sys.__stdout__
    output = buffer.getvalue()
    return jsonify({"output": output})

@app.route('/tema2-output', methods = ['POST'])
def get_tema2_output():
    data = request.get_json()
    tip = data.get('tip')
    n = data.get('n')
    epsilon = data.get('epsilon')
    A = np.array(data['A']) if data.get('A') is not None else None 
    dU = np.array(data['dU']) if data.get('dU') is not None else None 
    b = np.array(data['b']) if data.get('b') is not None else None 

    buffer = io.StringIO()
    sys.stdout = buffer 
    
    example_custom_diagonals(tip=tip, n=n, epsilon=epsilon, A=A, dU=dU, b=b)

    sys.stdout = sys.__stdout__ 
    output = buffer.getvalue()
    return jsonify({"output": output})

@app.route('/tema3-output', methods=['POST'])
def get_tema3_output():
    buffer = io.StringIO()
    sys.stdout = buffer

    tema3()

    sys.stdout = sys.__stdout__
    output = buffer.getvalue()
    return jsonify({"output": output})

@app.route('/tema4-output', methods=['POST'])
def get_tema4_output():
    buffer = io.StringIO()
    sys.stdout = buffer 

    main()

    sys.stdout = sys.__stdout__
    output = buffer.getvalue()
    return jsonify({"output": output})

if __name__ == "__main__":
    app.run(debug=True)