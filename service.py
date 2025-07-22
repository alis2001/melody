from flask import Flask, request, jsonify
from report_generation import genera_report_medico

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_report():
    data = request.get_json()
    if not data or "conversation" not in data:
        return jsonify({"error": "Missing conversation data"}), 400

    try:
        conversation = data["conversation"]
        report = genera_report_medico(conversation)
        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010)
