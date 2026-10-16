import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from kayro_core import Asistente

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)

bot = Asistente(os.path.dirname(os.path.abspath(__file__)))


@app.get("/")
def inicio():
    return jsonify({"estado": "ok", "servicio": "KAYRO IA"})


@app.route("/chat", methods=["GET", "POST"])
def chat():
    datos = request.get_json(silent=True) or {}
    mensaje = datos.get("mensaje") or datos.get("message") or request.args.get("q", "")
    return jsonify(bot.responder(str(mensaje)[:500]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
