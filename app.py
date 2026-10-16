import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from kayro_core import Asistente

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)

bot = Asistente(os.path.dirname(os.path.abspath(__file__)))


def _sesion(datos):
    # Cada usuario necesita su propio id para que el bot recuerde SU conversación
    s = (datos.get("sesion") or datos.get("usuario") or datos.get("user")
         or request.args.get("sesion") or request.args.get("usuario"))
    if not s:
        s = (request.headers.get("X-Forwarded-For") or request.remote_addr or "anon").split(",")[0].strip()
    return str(s)[:80]


@app.get("/")
def inicio():
    return jsonify({"estado": "ok", "servicio": "KAYRO IA"})


@app.route("/chat", methods=["GET", "POST"])
def chat():
    datos = request.get_json(silent=True) or {}
    mensaje = datos.get("mensaje") or datos.get("message") or request.args.get("q", "")
    return jsonify(bot.responder(str(mensaje)[:500], _sesion(datos)))


@app.route("/olvidar", methods=["GET", "POST"])
def olvidar():
    datos = request.get_json(silent=True) or {}
    bot.memoria.reiniciar(_sesion(datos))
    return jsonify({"estado": "ok", "respuesta": "Memoria borrada 🧹"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
