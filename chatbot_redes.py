"""
Proyecto: Chatbot con IA para consultar tecnologias y estandares de redes digitales
Materia : Fundamentos de Inteligencia Artificial
Autor   : Javier Maximiliano Canul de Dios
Escuela : Universidad Tecnologica de Tabasco

DESCRIPCION
-----------
Chatbot de RECUPERACION (version sencilla del enfoque RAG sin modelo de
lenguaje generativo). Responde solo con la informacion que esta en su base de
conocimiento y reconoce cuando no sabe.

Componentes de IA predictiva:
  1. Clasificador de temas (regresion logistica sobre TF-IDF):
     predice si la consulta es de cableado estructurado, redes inalambricas,
     fibra optica, seguridad basica o fuera de alcance.
  2. Recuperador por similitud (TF-IDF + similitud de coseno):
     busca en la base de conocimiento la entrada mas parecida a la consulta.
  3. Reglas de decision con umbrales afinados con datos (conjunto "dev"):
     decide si responde, si no hay informacion suficiente o si la consulta
     esta fuera de los temas del chatbot.

Flujo de una consulta:
  texto -> normalizar -> predecir tema -> buscar en la base -> decidir ->
  mostrar respuesta + preguntas relacionadas

USO
---
    python chatbot_redes.py entrenar                (entrena, afina y evalua)
    python chatbot_redes.py preguntar "pregunta"    (una sola consulta)
    python chatbot_redes.py chat                    (conversacion por consola)
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion, Pipeline

# ======================================================================
# CONFIGURACION
# ======================================================================
try:
    CARPETA = Path(__file__).parent
except NameError:                      # Google Colab / Jupyter
    CARPETA = Path.cwd()

ARCHIVO_KB = CARPETA / "base_conocimiento.csv"
ARCHIVO_TEMAS = CARPETA / "datos_preguntas.csv"
ARCHIVO_BANCO = CARPETA / "banco_preguntas.csv"
CARPETA_SALIDAS = CARPETA / "salidas"
ARCHIVO_MODELO = CARPETA_SALIDAS / "chatbot_redes.joblib"

PESO_PREGUNTA = 0.6     # peso de la similitud con la pregunta de la base
                        # (el resto, 0.4, es la similitud con la explicacion)
BONO_TEMA = 0.15        # cuanto ayuda el tema predicho al ordenar resultados
ORDEN_DECISIONES = ["responde", "sin_info", "fuera_alcance"]

# Mensajes tomados del prompt principal del documento (secciones 6 y 7)
MSG_SIN_INFO = ("La base de conocimiento no contiene información suficiente "
                "para responder esta consulta. Se recomienda revisar o "
                "incorporar documentación relacionada con el tema.")
MSG_FUERA = ("La consulta se encuentra fuera de los temas contemplados en este "
             "chatbot. Puedes realizar preguntas sobre cableado estructurado, "
             "redes inalámbricas, fibra óptica o seguridad básica en redes.")
MSG_BIENVENIDA = ("¡Hola! Soy un asistente de consulta sobre redes digitales. "
                  "Pregúntame sobre cableado estructurado, redes inalámbricas, "
                  "fibra óptica o seguridad básica en redes.")
SALUDOS = {"hola", "buenas", "buenos dias", "buenas tardes", "buenas noches",
           "hey", "que tal"}

STOPWORDS = {
    "que", "es", "el", "la", "los", "las", "un", "una", "unos", "unas", "de",
    "del", "en", "y", "o", "a", "al", "se", "por", "para", "con", "como",
    "cual", "cuales", "mi", "mis", "tu", "su", "lo", "le", "me", "hay",
    "si", "no", "mas", "muy", "ya", "esta", "este", "esto", "son", "ser",
}


# ======================================================================
# 1. PREPROCESAMIENTO
# ======================================================================
SINONIMOS = {"cortafuegos": "firewall", "inalambrico": "inalambrica",
             "inalambricos": "inalambricas", "inalambricas": "inalambrica"}


def raiz(palabra: str) -> str:
    """Quita plurales sencillos (cables -> cable, conectores -> conector)."""
    if len(palabra) > 4 and palabra.endswith("es"):
        return palabra[:-2]
    if len(palabra) > 3 and palabra.endswith("s"):
        return palabra[:-1]
    return palabra


def normalizar(texto: str, con_raiz: bool = False) -> str:
    """Minusculas, sin acentos ni signos y sin palabras vacias.
    Con con_raiz=True tambien reduce plurales (se usa en la busqueda)."""
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace("wi-fi", "wifi").replace("wi fi", "wifi")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    palabras = [SINONIMOS.get(p, p) for p in texto.split() if p not in STOPWORDS]
    if con_raiz:
        palabras = [raiz(p) for p in palabras]
    return " ".join(palabras)


def crear_vectorizador() -> FeatureUnion:
    """TF-IDF por palabras (1-2 gramas) y por grupos de caracteres (2-5)."""
    return FeatureUnion([
        ("palabras", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                       sublinear_tf=True)),
    ])


# ======================================================================
# 2. CARGA DE DATOS
# ======================================================================
def cargar_archivos():
    for ruta in (ARCHIVO_KB, ARCHIVO_TEMAS, ARCHIVO_BANCO):
        if not ruta.exists():
            raise SystemExit(f"Falta el archivo {ruta.name} en {CARPETA}. "
                             "Subelo a la misma carpeta que este programa.")
    kb = pd.read_csv(ARCHIVO_KB).fillna("")
    temas = pd.read_csv(ARCHIVO_TEMAS).dropna()
    banco = pd.read_csv(ARCHIVO_BANCO).fillna("")
    return kb, temas, banco


# ======================================================================
# 3. CONSTRUCCION DEL CHATBOT
# ======================================================================
def construir_chatbot(kb: pd.DataFrame, temas: pd.DataFrame) -> dict:
    """Entrena el clasificador de temas y prepara el indice de busqueda."""
    # --- clasificador de temas: usa las preguntas de entrenamiento + las de la base
    entrenamiento = pd.concat([
        temas[["pregunta", "tema"]],
        kb[["pregunta", "tema"]],
    ]).drop_duplicates(subset="pregunta")
    clasificador = Pipeline([
        ("vector", crear_vectorizador()),
        ("clf", LogisticRegression(C=20, max_iter=3000)),
    ])
    clasificador.fit(entrenamiento["pregunta"].apply(normalizar),
                     entrenamiento["tema"])

    # --- indice de busqueda (con raices de palabra para unir singular y plural)
    # Cada entrada tiene su pregunta principal y varias formas alternas de
    # preguntar lo mismo (columna "alternas"); todas apuntan a la misma respuesta.
    variantes, duenos = [], []
    for i, fila in kb.iterrows():
        formas = [fila["pregunta"]] + [a.strip() for a in
                                       str(fila["alternas"]).split("|") if a.strip()]
        for forma in formas:
            variantes.append(normalizar(forma, True))
            duenos.append(i)
    texto_variantes = pd.Series(variantes)
    texto_explic = kb["explicacion"].apply(lambda t: normalizar(t, True))
    texto_preguntas = pd.Series(
        [" ".join(v for v, d in zip(variantes, duenos) if d == i)
         for i in range(len(kb))])
    documentos = texto_preguntas + " " + texto_explic

    vectorizador = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    vectorizador.fit(documentos)

    # pesos (idf) de cada palabra, para medir que tanto cubre la base la consulta
    uni = TfidfVectorizer().fit(documentos)
    idf = dict(zip(uni.get_feature_names_out(), uni.idf_))

    return {
        "kb": kb.reset_index(drop=True),
        "clasificador": clasificador,
        "vectorizador": vectorizador,
        "matriz_variantes": vectorizador.transform(texto_variantes),
        "duenos": np.array(duenos),
        "matriz_explic": vectorizador.transform(texto_explic),
        "idf": idf,
        "idf_max": float(max(idf.values())),
        "palabras_doc": [set(d.split()) for d in documentos],
        "umbral_sim": 0.30,       # se reemplazan al afinar con el conjunto dev
        "umbral_cob": 0.50,
        "umbral_fuera": 0.50,
    }


# ======================================================================
# 4. ANALISIS Y DECISION DE UNA CONSULTA
# ======================================================================
def analizar(bot: dict, consulta: str, bono_tema: float = BONO_TEMA) -> dict:
    """Calcula el tema predicho y el ranking de la base de conocimiento."""
    texto = normalizar(consulta)
    kb = bot["kb"]
    if not texto:
        return {"vacia": True}

    # 1) tema predicho
    clasificador = bot["clasificador"]
    probas = clasificador.predict_proba([texto])[0]
    clases = list(clasificador.classes_)
    tema_pred = clases[int(np.argmax(probas))]
    prob_tema = float(probas.max())

    # 2) similitud con cada entrada de la base
    texto_r = normalizar(consulta, True)
    q = bot["vectorizador"].transform([texto_r])
    sim_var = cosine_similarity(q, bot["matriz_variantes"])[0]
    sim_preg = np.zeros(len(kb))
    np.maximum.at(sim_preg, bot["duenos"], sim_var)   # mejor variante de cada entrada
    sim_expl = cosine_similarity(q, bot["matriz_explic"])[0]
    similitud = PESO_PREGUNTA * sim_preg + (1 - PESO_PREGUNTA) * sim_expl

    # 3) cobertura: que parte (ponderada) de las palabras de la consulta
    #    aparece en cada entrada. Palabras que la base no conoce cuentan
    #    como informacion que falta, con el peso maximo.
    tokens = texto_r.split()
    pesos = [bot["idf"].get(t, bot["idf_max"]) for t in tokens]
    total = sum(pesos) or 1.0
    cobertura = np.array([
        sum(w for t, w in zip(tokens, pesos) if t in palabras) / total
        for palabras in bot["palabras_doc"]])

    # 4) ranking: similitud + ayuda del tema predicho
    prob_de_cada_entrada = np.array([probas[clases.index(t)] for t in kb["tema"]])
    puntaje = similitud + bono_tema * prob_de_cada_entrada
    orden = np.argsort(puntaje)[::-1]

    return {"vacia": False, "tema": tema_pred, "prob_tema": prob_tema,
            "orden": orden.tolist(), "similitud": similitud,
            "cobertura": cobertura, "puntaje": puntaje}


def decidir(analisis: dict, umbral_sim: float, umbral_cob: float,
            umbral_fuera: float) -> str:
    """Devuelve: 'responde', 'sin_info' o 'fuera_alcance'."""
    if analisis["vacia"]:
        return "sin_info"
    if analisis["tema"] == "fuera de alcance" and analisis["prob_tema"] >= umbral_fuera:
        return "fuera_alcance"
    mejor = analisis["orden"][0]
    if analisis["similitud"][mejor] < umbral_sim or analisis["cobertura"][mejor] < umbral_cob:
        return "sin_info"
    return "responde"


# ======================================================================
# 5. RESPUESTA AL USUARIO
# ======================================================================
def responder(bot: dict, consulta: str) -> str:
    if normalizar(consulta) in SALUDOS:
        return MSG_BIENVENIDA
    a = analizar(bot, consulta)
    decision = decidir(a, bot["umbral_sim"], bot["umbral_cob"], bot["umbral_fuera"])
    if decision == "fuera_alcance":
        return MSG_FUERA
    if decision == "sin_info":
        return MSG_SIN_INFO

    kb = bot["kb"]
    mejor = a["orden"][0]
    fila = kb.iloc[mejor]
    texto = (f"[Tema: {fila['tema']} | similitud {a['similitud'][mejor]:.2f}]\n\n"
             f"{fila['explicacion']}")

    relacionadas = [i for i in a["orden"][1:4]
                    if a["similitud"][i] >= 0.5 * bot["umbral_sim"]][:2]
    if relacionadas:
        texto += "\n\nTambién podrías preguntar:"
        for i in relacionadas:
            texto += f"\n  - {kb.iloc[i]['pregunta']}"
    return texto


# ======================================================================
# 6. EVALUACION
# ======================================================================
def preparar_analisis(bot: dict, banco: pd.DataFrame, bono_tema=BONO_TEMA):
    return [analizar(bot, p, bono_tema) for p in banco["pregunta"]]


def decision_correcta(decision: str, tipo: str) -> bool:
    return {"en_base": "responde", "sin_info": "sin_info",
            "fuera_alcance": "fuera_alcance"}[tipo] == decision


def exactitud_decisiones(bot, banco, analisis, umbral_sim, umbral_cob,
                         umbral_fuera) -> float:
    """Una respuesta cuenta como correcta si la decision es la esperada y,
    cuando debia responder, la entrada recuperada es la correcta."""
    aciertos = 0
    for (_, fila), a in zip(banco.iterrows(), analisis):
        d = decidir(a, umbral_sim, umbral_cob, umbral_fuera)
        ok = decision_correcta(d, fila["tipo"])
        if ok and fila["tipo"] == "en_base":
            ok = bot["kb"].iloc[a["orden"][0]]["id"] == fila["id_esperado"]
        aciertos += ok
    return aciertos / len(banco)


def afinar_umbrales(bot, banco_dev) -> tuple:
    """Busca los umbrales que mejor funcionan en el conjunto DEV
    (el conjunto TEST no se toca para no hacer trampa)."""
    analisis = preparar_analisis(bot, banco_dev)
    mejores, mejor_acc = [], -1.0
    for us in np.arange(0.05, 0.61, 0.025):
        for uc in np.arange(0.30, 1.01, 0.05):
            for uf in np.arange(0.30, 0.96, 0.10):
                acc = exactitud_decisiones(bot, banco_dev, analisis, us, uc, uf)
                if acc > mejor_acc + 1e-9:
                    mejor_acc, mejores = acc, [(us, uc, uf)]
                elif abs(acc - mejor_acc) < 1e-9:
                    mejores.append((us, uc, uf))
    # entre los empates se toma la mediana, que suele ser mas estable
    us, uc, uf = (round(float(np.median([m[i] for m in mejores])), 3) for i in range(3))
    acc = exactitud_decisiones(bot, banco_dev, analisis, us, uc, uf)
    return us, uc, uf, acc


def top_k_retrieval(bot, banco, analisis, k) -> float:
    filas = [(f, a) for (_, f), a in zip(banco.iterrows(), analisis)
             if f["tipo"] == "en_base"]
    ok = sum(f["id_esperado"] in bot["kb"].iloc[a["orden"][:k]]["id"].values
             for f, a in filas)
    return ok / len(filas)


def evaluar(bot, banco_test):
    analisis = preparar_analisis(bot, banco_test)
    us, uc, uf = bot["umbral_sim"], bot["umbral_cob"], bot["umbral_fuera"]

    reales, predichas, filas_error = [], [], []
    for (_, fila), a in zip(banco_test.iterrows(), analisis):
        d = decidir(a, us, uc, uf)
        id_rec = bot["kb"].iloc[a["orden"][0]]["id"] if not a["vacia"] else ""
        reales.append({"en_base": "responde"}.get(fila["tipo"], fila["tipo"]))
        predichas.append(d)
        correcto = decision_correcta(d, fila["tipo"])
        if correcto and fila["tipo"] == "en_base":
            correcto = id_rec == fila["id_esperado"]
        if not correcto:
            filas_error.append({"pregunta": fila["pregunta"], "esperado":
                                fila["id_esperado"] or fila["tipo"],
                                "decision": d, "id_recuperado": id_rec,
                                "similitud": round(float(a["similitud"][a["orden"][0]]), 3),
                                "cobertura": round(float(a["cobertura"][a["orden"][0]]), 3),
                                "tema_predicho": a["tema"]})

    por_tipo = {}
    for tipo in ["en_base", "sin_info", "fuera_alcance"]:
        mask = banco_test["tipo"] == tipo
        sub_b = banco_test[mask]
        sub_a = [a for a, m in zip(analisis, mask) if m]
        por_tipo[tipo] = exactitud_decisiones(bot, sub_b, sub_a, us, uc, uf)

    metricas = {
        "exactitud_global": exactitud_decisiones(bot, banco_test, analisis, us, uc, uf),
        "exactitud_por_tipo": por_tipo,
        "top1_recuperacion": top_k_retrieval(bot, banco_test, analisis, 1),
        "top3_recuperacion": top_k_retrieval(bot, banco_test, analisis, 3),
    }
    # ablacion: que pasa si no se usa el tema predicho para ordenar
    sin_tema = preparar_analisis(bot, banco_test, bono_tema=0.0)
    metricas["top1_sin_clasificador"] = top_k_retrieval(bot, banco_test, sin_tema, 1)
    return metricas, reales, predichas, pd.DataFrame(filas_error), analisis


# ======================================================================
# 7. GRAFICAS
# ======================================================================
def graficas(bot, banco_test, metricas, reales, predichas, analisis):
    CARPETA_SALIDAS.mkdir(exist_ok=True)

    # matriz de confusion de decisiones
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        reales, predichas, labels=ORDEN_DECISIONES, ax=ax, cmap="Blues",
        colorbar=False)
    ax.set_title("Decisiones del chatbot (conjunto de prueba)")
    ax.set_xlabel("Decisión del chatbot")
    ax.set_ylabel("Decisión esperada")
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDAS / "matriz_decisiones.png", dpi=150)
    plt.close()

    # aciertos por tipo de pregunta
    nombres = {"en_base": "Preguntas en la base", "sin_info": "Sin información",
               "fuera_alcance": "Fuera de alcance"}
    valores = [metricas["exactitud_por_tipo"][t] * 100 for t in nombres]
    plt.figure(figsize=(7, 4.5))
    barras = plt.bar(list(nombres.values()), valores,
                     color=["#2b6cb0", "#dd6b20", "#38a169"])
    for b, v in zip(barras, valores):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f} %", ha="center")
    plt.ylim(0, 110)
    plt.ylabel("Aciertos (%)")
    plt.title("Aciertos por tipo de pregunta (conjunto de prueba)")
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDAS / "aciertos_por_tipo.png", dpi=150)
    plt.close()

    # distribucion de similitud
    sims = {"en_base": [], "sin_info": [], "fuera_alcance": []}
    for (_, fila), a in zip(banco_test.iterrows(), analisis):
        sims[fila["tipo"]].append(float(a["similitud"][a["orden"][0]]))
    plt.figure(figsize=(7, 4.5))
    for tipo, color in zip(sims, ["#2b6cb0", "#dd6b20", "#38a169"]):
        plt.hist(sims[tipo], bins=10, range=(0, 1), alpha=0.6,
                 label=nombres[tipo], color=color)
    plt.axvline(bot["umbral_sim"], color="black", linestyle="--",
                label=f"Umbral ({bot['umbral_sim']})")
    plt.xlabel("Similitud con la mejor entrada de la base")
    plt.ylabel("Cantidad de preguntas")
    plt.title("Similitud según el tipo de pregunta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDAS / "distribucion_similitud.png", dpi=150)
    plt.close()

    # efecto del clasificador de temas en la recuperacion
    plt.figure(figsize=(6, 4.5))
    etiquetas = ["Solo similitud", "Similitud + tema"]
    vals = [metricas["top1_sin_clasificador"] * 100,
            metricas["top1_recuperacion"] * 100]
    barras = plt.bar(etiquetas, vals, color=["#a0aec0", "#2b6cb0"])
    for b, v in zip(barras, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f} %", ha="center")
    plt.ylim(0, 110)
    plt.ylabel("Entrada correcta en 1er lugar (%)")
    plt.title("Efecto del clasificador de temas")
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDAS / "efecto_clasificador.png", dpi=150)
    plt.close()


# ======================================================================
# 8. ENTRENAR, AFINAR Y EVALUAR
# ======================================================================
def entrenar():
    CARPETA_SALIDAS.mkdir(exist_ok=True)
    kb, temas, banco = cargar_archivos()
    print(f"Base de conocimiento: {len(kb)} entradas | "
          f"Datos para el clasificador de temas: {len(temas)} preguntas")
    banco_dev = banco[banco["conjunto"] == "dev"].reset_index(drop=True)
    banco_test = banco[banco["conjunto"] == "test"].reset_index(drop=True)
    print(f"Banco de preguntas: {len(banco_dev)} para afinar (dev) y "
          f"{len(banco_test)} para evaluar (test)\n")

    bot = construir_chatbot(kb, temas)

    us, uc, uf, acc_dev = afinar_umbrales(bot, banco_dev)
    bot["umbral_sim"], bot["umbral_cob"], bot["umbral_fuera"] = us, uc, uf
    print(f"Umbrales afinados con dev -> similitud: {us} | cobertura: {uc} | "
          f"fuera de alcance: {uf} (exactitud en dev: {acc_dev*100:.1f} %)\n")

    metricas, reales, predichas, errores, analisis = evaluar(bot, banco_test)
    print("RESULTADOS EN EL CONJUNTO DE PRUEBA (no usado para afinar)")
    print(f"  Exactitud global de decisiones : {metricas['exactitud_global']*100:.1f} %")
    for tipo, v in metricas["exactitud_por_tipo"].items():
        print(f"    - {tipo:<14}: {v*100:.1f} %")
    print(f"  Entrada correcta en 1er lugar  : {metricas['top1_recuperacion']*100:.1f} %")
    print(f"  Entrada correcta en top 3      : {metricas['top3_recuperacion']*100:.1f} %")
    print(f"  Top 1 sin usar el clasificador : {metricas['top1_sin_clasificador']*100:.1f} %\n")

    if len(errores):
        print(f"Preguntas con error ({len(errores)}):")
        print(errores.to_string(index=False), "\n")
        errores.to_csv(CARPETA_SALIDAS / "errores_prueba.csv", index=False,
                       encoding="utf-8-sig")

    graficas(bot, banco_test, metricas, reales, predichas, analisis)

    joblib.dump(bot, ARCHIVO_MODELO)
    resumen = {
        "entradas_base_conocimiento": int(len(kb)),
        "preguntas_dev": int(len(banco_dev)),
        "preguntas_test": int(len(banco_test)),
        "umbral_similitud": us, "umbral_cobertura": uc,
        "umbral_fuera_alcance": uf,
        "exactitud_dev": round(acc_dev, 4),
        "exactitud_test": round(metricas["exactitud_global"], 4),
        "exactitud_por_tipo_test": {k: round(v, 4) for k, v in
                                    metricas["exactitud_por_tipo"].items()},
        "top1_test": round(metricas["top1_recuperacion"], 4),
        "top3_test": round(metricas["top3_recuperacion"], 4),
        "top1_sin_clasificador_test": round(metricas["top1_sin_clasificador"], 4),
    }
    with open(CARPETA_SALIDAS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"Archivos guardados en: {CARPETA_SALIDAS}")
    return bot


# ======================================================================
# 9. USO DEL CHATBOT
# ======================================================================
def cargar_bot() -> dict:
    if not ARCHIVO_MODELO.exists():
        print("Todavía no hay un chatbot entrenado, entrenando uno...\n")
        return entrenar()
    return joblib.load(ARCHIVO_MODELO)


def modo_chat():
    bot = cargar_bot()
    print("\n" + MSG_BIENVENIDA)
    print("(escribe 'salir' para terminar)")
    while True:
        consulta = input("\nTú > ").strip()
        if consulta.lower() in {"salir", "exit", "q"}:
            print("Hasta luego.")
            break
        if consulta:
            print("\nChatbot >", responder(bot, consulta))


def en_notebook() -> bool:
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def main():
    if en_notebook():       # Colab / Jupyter: entrenar y abrir el chat
        entrenar()
        modo_chat()
        return

    parser = argparse.ArgumentParser(
        description="Chatbot de consulta de redes digitales (IA predictiva)")
    sub = parser.add_subparsers(dest="comando")
    sub.add_parser("entrenar", help="entrena, afina umbrales y evalua")
    p = sub.add_parser("preguntar", help="responde una sola consulta")
    p.add_argument("consulta")
    sub.add_parser("chat", help="conversacion por consola")
    args = parser.parse_args()

    if args.comando == "preguntar":
        print(responder(cargar_bot(), args.consulta))
    elif args.comando == "chat":
        modo_chat()
    else:
        entrenar()


if __name__ == "__main__":
    main()
