from openai import OpenAI
import os
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-uGNIeQaOCBkeqsuZfrN691FmRym1KqxUhjSPWDfjZOMXerro"),
    base_url="https://api.chatanywhere.tech/v1"
)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return jsonify({
            'status': 'success',
            'response': response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000) 