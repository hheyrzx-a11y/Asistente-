import os, re, json, random, unicodedata
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


class Asistente:
    def __init__(self, carpeta="."):
        with open(os.path.join(carpeta, "intents.json"), encoding="utf-8") as f:
            self.datos = json.load(f)
        self.umbral = float(self.datos.get("umbral", 0.40))
        try:
            self.modelo = joblib.load(os.path.join(carpeta, "model.joblib"))
        except Exception:
            # Si la versión de librerías no coincide, se reentrena al arrancar (tarda 1-2 s)
            self.modelo = entrenar(self.datos)

    def responder(self, mensaje):
        if not normalizar(mensaje):
            return {"intent": "vacio", "confianza": 0.0,
                    "respuesta": self.datos["vacio"]}
        proba = self.modelo.predict_proba([str(mensaje)])[0]
        i = int(proba.argmax())
        intent = str(self.modelo.classes_[i])
        conf = float(proba[i])
        if intent == "fuera_tema" or conf < self.umbral:
            return {"intent": "fuera_tema", "confianza": round(conf, 3),
                    "respuesta": self.datos["rechazo"]}
        return {"intent": intent, "confianza": round(conf, 3),
                "respuesta": self.datos["intents"][intent]["respuesta"]}
