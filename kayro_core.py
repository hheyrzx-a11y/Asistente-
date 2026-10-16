import os, re, json, random, unicodedata, time, ast, math, operator, threading
from collections import OrderedDict
from datetime import datetime, timezone
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion


def normalizar(texto):
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def crear_modelo():
    union = FeatureUnion([
        ("palabras", TfidfVectorizer(preprocessor=normalizar, analyzer="word",
                                     ngram_range=(1, 2), sublinear_tf=True)),
        ("letras", TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                   ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", union),
                     ("clf", LogisticRegression(C=30, max_iter=3000,
                                                class_weight="balanced"))])


def aumentar(frase, rng):
    # Crea variantes con errores de dedo para reconocer palabras mal escritas
    salida = [frase]
    for _ in range(2):
        s = list(frase)
        if len(s) > 3:
            i = rng.randrange(len(s))
            op = rng.choice(["borrar", "duplicar", "cambiar"])
            if op == "borrar":
                del s[i]
            elif op == "duplicar":
                s.insert(i, s[i])
            elif i < len(s) - 1:
                s[i], s[i + 1] = s[i + 1], s[i]
        salida.append("".join(s))
    return salida


def entrenar(datos):
    rng = random.Random(7)
    X, y = [], []
    for nombre, info in datos["intents"].items():
        for p in info["patrones"]:
            for v in aumentar(p, rng):
                X.append(v)
                y.append(nombre)
    modelo = crear_modelo()
    modelo.fit(X, y)
    return modelo


# ───────────────────────── CALCULADORA SEGURA ─────────────────────────
class _Err(Exception):
    pass


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.FloorDiv: operator.floordiv}
_NUM = r"\d+(?:\.\d+)?"
_KW = re.compile(r"\b(?:cuanto|calcula|calcular|resuelve|resultado|operacion|suma|resta)\b")
_SEG = re.compile(r"^(?:y\s+)?(?:ahora\s+)?(?:si\s+le\s+)?(?:le\s+)?"
                  r"(mas|menos|por|entre|x|\+|-|\*|/|dividido\s+entre|dividido\s+por|multiplicado\s+por|elevado\s+a)"
                  r"\s*(" + _NUM + r")\s*$")
_MAPA = {"mas": "+", "+": "+", "menos": "-", "-": "-", "por": "*", "x": "*", "*": "*",
         "multiplicado por": "*", "entre": "/", "/": "/", "dividido entre": "/",
         "dividido por": "/", "elevado a": "**"}


def _eval(n):
    if isinstance(n, ast.Expression):
        return _eval(n.body)
    if isinstance(n, ast.Constant) and type(n.value) in (int, float):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
        v = _eval(n.operand)
        return v if isinstance(n.op, ast.UAdd) else -v
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        a, b = _eval(n.left), _eval(n.right)
        if isinstance(n.op, ast.Pow) and (abs(b) > 200 or abs(a) > 1e9):
            raise OverflowError
        return _BIN[type(n.op)](a, b)
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sqrt"
            and len(n.args) == 1 and not n.keywords):
        v = _eval(n.args[0])
        if v < 0:
            raise _Err("neg")
        return math.sqrt(v)
    raise ValueError("no permitido")


def _fmt(v):
    if isinstance(v, complex):
        raise ValueError("complejo")
    if isinstance(v, float):
        if math.isinf(v) or math.isnan(v):
            raise OverflowError
        v = round(v, 8)
        if v == int(v):
            v = int(v)
    if isinstance(v, int) and v.bit_length() > 4000:
        raise OverflowError
    s = str(v)
    if len(s) > 22:
        s = "{:.6e}".format(float(v))
    return s


def _limpiar(texto):
    t = str(texto).lower().replace("²", "**2").replace("³", "**3")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", ".")
    t = re.sub(r"[¿?¡!]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _candidatos(t):
    if re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", t):
        return []  # parece una fecha
    s = re.sub(r"(" + _NUM + r")\s*(?:por\s*ciento|porciento|%)\s*de\s*", r"\1 * 0.01 * ", t)
    s = re.sub(r"\bal\s+cuadrado\b", " ** 2 ", s)
    s = re.sub(r"\bal\s+cubo\b", " ** 3 ", s)
    s = re.sub(r"\b(?:elevado\s+a(?:\s+la)?(?:\s+potencia)?(?:\s+de)?|a\s+la\s+potencia\s+de)\b", " ** ", s)
    s = re.sub(r"\braiz\s+cuadrada\s+de\b|\braiz\s+de\b", " R ", s)
    A, D = r"(?<=[\d\)])\s*", r"\s*(?=[\d\(R])"
    s = re.sub(A + r"(?:dividido\s+(?:entre|por|en)|dividido|entre|sobre)" + D, " / ", s)
    s = re.sub(A + r"(?:multiplicado\s+por|multiplicado\s+x|veces|por|x)" + D, " * ", s)
    s = re.sub(A + r"(?:mas|suma(?:do)?\s+a)" + D, " + ", s)
    s = re.sub(A + r"menos" + D, " - ", s)
    salida = []
    for p in re.split(r"[^0-9.+\-*/()R\s]+", s):
        p = re.sub(r"\s+", " ", p).strip(" .")
        if not p or not re.search(r"[+\-*/R]", p):
            continue
        n_nums = len(re.findall(_NUM, p))
        if n_nums < (1 if "R" in p else 2):
            continue
        solo_resta = re.search(r"[+*/R]", p) is None
        if solo_resta and not ("menos" in t or " - " in t or _KW.search(t) or len(t.split()) == 1):
            continue  # cosas como "2-3 dias" o "555-1234"
        salida.append(p)
    return salida


def _resolver(expr):
    py = expr
    for _ in range(3):
        py = re.sub(r"R\s*(" + _NUM + r"|\([^()]*\)|sqrt\([^()]*\))", r"sqrt(\1)", py)
    if "R" in py or len(py) > 80:
        raise ValueError("expresion")
    return _eval(ast.parse(py, mode="eval"))


def _bonita(expr):
    e = re.sub(r"\s+", " ", expr).replace("**", "^").replace("*", "×").replace("/", "÷").replace("R", "√")
    e = re.sub(r"(" + _NUM + r") × 0\.01 ×", r"\1% de", e)
    e = re.sub(r"\((-?" + _NUM + r")\)", r"\1", e)
    return re.sub(r"√\s+", "√", e)


def calcular(texto, previo=None):
    # Devuelve ("ok", operacion, resultado_texto, valor) | ("error", mensaje) | None
    t = _limpiar(texto)
    cands = []
    if previo is not None:
        m = _SEG.match(t)
        if m:
            op = _MAPA[re.sub(r"\s+", " ", m.group(1))]
            cands.append("(" + _fmt(previo) + ") " + op + " " + m.group(2))
    if not cands:
        cands = _candidatos(t)
    for c in cands:
        try:
            v = _resolver(c)
            return ("ok", _bonita(c), _fmt(v), v)
        except ZeroDivisionError:
            return ("error", "🚫 No se puede dividir entre cero. ¡Buen intento! 😄")
        except OverflowError:
            return ("error", "😵 Ese número es demasiado grande para mí.")
        except _Err:
            return ("error", "🚫 No puedo sacar raíz cuadrada de un número negativo.")
        except (ValueError, SyntaxError, TypeError, RecursionError):
            continue
    return None


# ───────────────────────── MEMORIA POR USUARIO ─────────────────────────
class Memoria:
    # Guarda en RAM: nombre, último tema, último resultado y últimos mensajes.
    def __init__(self, max_sesiones=2000, ttl=6 * 3600):
        self.d = OrderedDict()
        self.max = max_sesiones
        self.ttl = ttl

    def _nueva(self):
        return {"nombre": None, "ultimo_intent": None, "ultima_respuesta": None,
                "resultado": None, "historial": [], "t": time.time()}

    def get(self, sid):
        ahora = time.time()
        m = self.d.get(sid)
        if m is None or ahora - m["t"] > self.ttl:
            m = self._nueva()
            self.d[sid] = m
        m["t"] = ahora
        self.d.move_to_end(sid)
        while len(self.d) > self.max:
            self.d.popitem(last=False)
        return m

    def reiniciar(self, sid):
        self.d[sid] = self._nueva()
        return self.d[sid]


_NOMBRE = re.compile(r"(?i)\b(?:me llamo|mi nombre es|me dicen|ll[aá]mame)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,20})\b")
_NO_NOMBRE = {"yo", "en", "de", "por", "que", "como", "asi", "un", "una", "el", "la", "algo",
              "tu", "mi", "ya", "y", "pero", "cuando", "donde", "porque"}


def extraer_nombre(texto):
    m = _NOMBRE.search(str(texto))
    if not m or m.group(1).lower() in _NO_NOMBRE:
        return None
    n = m.group(1)
    return n[:1].upper() + n[1:].lower()


DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
FLUJO = ["login", "parchar", "studio", "visibilidad"]
RECHAZO_DEF = "🙅‍♂️ Eso no es lo mío, pero con KAYRO STUDIO sí te ayudo: parchar tu video y subirlo a TikTok Studio."
NO_ENTENDI_DEF = "🤔 No te entendí bien. ¿Me lo dices con otras palabras?"


def _ahora(tz):
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz))
    except Exception:
        return datetime.now(timezone.utc)


def elegir(x):
    if isinstance(x, list):
        return random.choice(x) if x else ""
    return str(x)


class Asistente:
    def __init__(self, carpeta="."):
        with open(os.path.join(carpeta, "intents.json"), encoding="utf-8") as f:
            self.datos = json.load(f)
        self.umbral = float(self.datos.get("umbral", 0.30))
        self.zona = self.datos.get("zona_horaria", "America/Mexico_City")
        self.memoria = Memoria()
        self.lock = threading.Lock()
        try:
            self.modelo = joblib.load(os.path.join(carpeta, "model.joblib"))
        except Exception:
            # Si la versión de librerías no coincide, se reentrena al arrancar (tarda 1-2 s)
            self.modelo = entrenar(self.datos)

    # ---------- utilidades ----------
    def _llenar(self, texto, m):
        if "{" not in texto:
            return texto
        a = _ahora(self.zona)
        return (texto.replace("{nombre}", m["nombre"] or "amigo")
                     .replace("{hora}", "%d:%02d" % (a.hour, a.minute))
                     .replace("{fecha}", "%s %d de %s de %d" % (DIAS[a.weekday()], a.day, MESES[a.month - 1], a.year)))

    def _resp(self, intent, m):
        return self._llenar(elegir(self.datos["intents"][intent]["respuesta"]), m)

    def _salida(self, intent, conf, texto, m, contexto=True, msg=None):
        m["ultima_respuesta"] = texto
        if contexto:
            m["ultimo_intent"] = intent
        if msg is not None:
            m["historial"].append(msg)
            del m["historial"][:-10]
        return {"intent": intent, "confianza": round(float(conf), 3), "respuesta": texto,
                "nombre": m["nombre"]}

    # ---------- respuesta principal ----------
    def responder(self, mensaje, sesion="anon"):
        with self.lock:
            m = self.memoria.get(str(sesion))
            msg = str(mensaje)[:500]
            if not normalizar(msg) and not re.search(r"\d", msg):
                return self._salida("vacio", 0.0, elegir(self.datos["vacio"]), m, False)

            # 1) ¿Me dice su nombre?
            nuevo = extraer_nombre(msg)
            if nuevo:
                m["nombre"] = nuevo
                return self._salida("nombre", 1.0,
                                    "¡Mucho gusto, %s! 😊 Ya me acordé de tu nombre. ¿En qué te ayudo?" % nuevo,
                                    m, False, msg)

            # 2) ¿Es una cuenta?
            calc = calcular(msg, m["resultado"])
            if calc:
                if calc[0] == "ok":
                    m["resultado"] = calc[3]
                    txt = random.choice(["🧮 %s = %s", "🧠 %s = %s", "✅ %s = %s"]) % (calc[1], calc[2])
                    return self._salida("matematicas", 1.0, txt, m, True, msg)
                return self._salida("matematicas", 1.0, calc[1], m, True, msg)

            # 3) Modelo de intenciones
            proba = self.modelo.predict_proba([msg])[0]
            i = int(proba.argmax())
            intent = str(self.modelo.classes_[i])
            conf = float(proba[i])
            intents = self.datos["intents"]

            if intent == "fuera_tema":
                return self._salida("fuera_tema", conf, self._llenar(elegir(self.datos["rechazo"]), m), m, False, msg)
            if conf < self.umbral:
                return self._salida("no_entendi", conf, self._llenar(elegir(self.datos["no_entendi"]), m), m, False, msg)

            # 4) Intenciones con memoria
            if intent == "repetir":
                texto = m["ultima_respuesta"] or "Todavía no te he dicho nada 😅 ¿En qué te ayudo?"
                return self._salida("repetir", conf, texto, m, False, msg)

            if intent == "mi_nombre":
                if m["nombre"]:
                    texto = "Te llamas %s 😄 ¡Claro que me acuerdo!" % m["nombre"]
                else:
                    texto = "Aún no me lo has dicho 😅 Escríbeme «me llamo ...» y no se me olvida."
                return self._salida("mi_nombre", conf, texto, m, False, msg)

            if intent == "recordar":
                if m["historial"]:
                    texto = "Lo último que me dijiste antes fue: «%s» 🧠" % m["historial"][-1]
                else:
                    texto = "Todavía no hemos hablado de nada 😄 ¡Empecemos!"
                return self._salida("recordar", conf, texto, m, False, msg)

            if intent == "olvidar":
                self.memoria.reiniciar(str(sesion))
                m = self.memoria.get(str(sesion))
                return self._salida("olvidar", conf, "🧹 Listo, borré lo que hablamos. ¡Empezamos de cero!", m, False)

            if intent == "siguiente":
                flujo = [x for x in FLUJO if x in intents]
                ult = m["ultimo_intent"]
                if flujo and ult in flujo:
                    k = flujo.index(ult)
                    if k + 1 < len(flujo):
                        nxt = flujo[k + 1]
                        texto = self._resp(nxt, m)
                        m["ultimo_intent"] = nxt
                    else:
                        texto = "🎉 ¡Ese era el último paso! Si algo no te sale, dime y lo revisamos."
                        m["ultimo_intent"] = None
                elif flujo:
                    texto = "Empecemos por el principio 👇\n\n" + self._resp(flujo[0], m)
                    m["ultimo_intent"] = flujo[0]
                else:
                    texto = self.datos["no_entendi"] if isinstance(self.datos["no_entendi"], str) else NO_ENTENDI_DEF
                return self._salida("siguiente", conf, texto, m, False, msg)

            # 5) Respuesta normal del JSON
            return self._salida(intent, conf, self._resp(intent, m), m, True, msg)
