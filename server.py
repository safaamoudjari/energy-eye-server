from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Energy Eye Secure Server is running "

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "Server works fine"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
Add Flask server
