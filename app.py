from flask import Flask, render_template, request, jsonify
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'IOC.NS').strip().upper()
        
        # Import quick analysis function
        from web_analysis import run_quick_analysis
        
        # Run quick analysis (no LSTM training, faster)
        results = run_quick_analysis(ticker)
        
        return jsonify(results)
        
    except Exception as e:
        import traceback
        error_msg = f"Error analyzing {ticker}: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 