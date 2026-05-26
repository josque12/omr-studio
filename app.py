import streamlit as st
import json
import numpy as np
import cv2
import tensorflow as tf
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import tempfile


st.set_page_config(
    page_title="OMR — Reconocimiento de Partituras",
    page_icon="🎵",
    layout="wide"
)

st.title("OMR — Reconocimiento Óptico de Partituras")
st.markdown("Sube una imagen de partitura y el modelo detectará los símbolos musicales.")

@st.cache_resource
def load_model_and_vocab():
    base = Path(__file__).parent
    with open(base / "vocab.json", "r", encoding="utf-8") as f:
        save_data = json.load(f)
    vocab      = save_data["vocab"]
    idx2tok    = {int(k): v for k, v in save_data["idx2tok"].items()}
    VOCAB_SIZE = save_data["VOCAB_SIZE"]
    BLANK_IDX  = save_data["BLANK_IDX"]
    model      = tf.keras.models.load_model(str(base / "best_model.keras"))
    return model, vocab, idx2tok, BLANK_IDX

with st.spinner("Cargando modelo..."):
    model, vocab, idx2tok, BLANK_IDX = load_model_and_vocab()
st.success("Modelo listo ✓")

IMG_HEIGHT = 128

def preprocess_image(img_bytes, height=IMG_HEIGHT):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    new_w = max(1, int(w * height / h))
    img = cv2.resize(img, (new_w, height))
    img = img.astype(np.float32) / 255.0
    img = 1.0 - img
    return img

def predict(model, img, idx2tok, blank_idx):
    X = img[np.newaxis, :, :, np.newaxis]
    y_pred    = model(X, training=False)
    T         = y_pred.shape[1]
    input_len = np.array([T])
    decoded, log_probs = tf.keras.backend.ctc_decode(
        y_pred, input_len, greedy=True
    )
    raw = decoded[0].numpy()[0]
    tokens = [
        idx2tok[t] for t in raw
        if t >= 0 and t != blank_idx and t in idx2tok
    ]
    confidence = float(-log_probs.numpy()[0][0])
    return tokens, confidence

def tokens_to_musicxml(tokens, title="OMR Result"):
    DURATION_MAP = {
        "double_whole":  ("breve",   8.0),
        "whole":         ("whole",   4.0),
        "half":          ("half",    2.0),
        "quarter":       ("quarter", 1.0),
        "eighth":        ("eighth",  0.5),
        "sixteenth":     ("16th",    0.25),
        "thirty_second": ("32nd",    0.125),
        "sixty_fourth":  ("64th",    0.0625),
    }
    ALTER_PREFIX = {
        "Bb":("B",-1),"Ab":("A",-1),"Eb":("E",-1),"Db":("D",-1),
        "Gb":("G",-1),"Cb":("C",-1),"Fb":("F",-1),
        "B#":("B", 1),"C#":("C", 1),"D#":("D", 1),"E#":("E", 1),
        "F#":("F", 1),"G#":("G", 1),"A#":("A", 1),
        "A":("A",0),"B":("B",0),"C":("C",0),"D":("D",0),
        "E":("E",0),"F":("F",0),"G":("G",0),
    }
    KEY_FIFTHS = {
        "CM":0,"GM":1,"DM":2,"AM":3,"EM":4,"BM":5,
        "FM":-1,"BbM":-2,"EbM":-3,"AbM":-4,"DbM":-5,"GbM":-6,
    }
    CLEF_MAP = {
        "G1":("G","1"),"G2":("G","2"),
        "F3":("F","3"),"F4":("F","4"),
        "C1":("C","1"),"C2":("C","2"),
        "C3":("C","3"),"C4":("C","4"),"C5":("C","5"),
    }

    def parse_pitch(p):
        for prefix in sorted(ALTER_PREFIX.keys(), key=len, reverse=True):
            if p.startswith(prefix):
                step, alter = ALTER_PREFIX[prefix]
                rest = p[len(prefix):]
                octave = int(rest) if rest.isdigit() else 4
                return step, alter, octave
        return None, None, None

    def parse_duration(s):
        fermata = s.endswith("_fermata")
        s = s.replace("_fermata", "")
        dots = 0
        if s.endswith(".."): dots, s = 2, s[:-2]
        elif s.endswith("."): dots, s = 1, s[:-1]
        if s not in DURATION_MAP: return None, None, None, None
        dur_type, base = DURATION_MAP[s]
        beats = base
        add = base / 2
        for _ in range(dots):
            beats += add; add /= 2
        return dur_type, beats, dots, fermata

    def parse_note(tok, prefix="note-"):
        body = tok[len(prefix):]
        if "_" not in body: return None
        pitch_str, dur_str = body.split("_", 1)
        step, alter, octave = parse_pitch(pitch_str)
        if step is None: return None
        dur_type, beats, dots, fermata = parse_duration(dur_str)
        if dur_type is None: return None
        return {"step":step,"alter":alter,"octave":octave,
                "type":dur_type,"beats":beats,"dots":dots,"fermata":fermata}

    def parse_rest(tok):
        dur_type, beats, dots, fermata = parse_duration(tok[5:])
        if dur_type is None: return None
        return {"type":dur_type,"beats":beats,"dots":dots,"fermata":fermata}

    DIVISIONS = 16
    beats_per = 4.0
    beat_type = 4
    fifths    = 0
    clef_sign = "G"
    clef_line = "2"
    measures  = []
    current   = []
    accum     = 0.0

    def flush():
        nonlocal accum
        if current:
            measures.append(list(current))
            current.clear()
            accum = 0.0

    for tok in tokens:
        tok = tok.strip()
        if tok.startswith("clef-"):
            code = tok[5:]
            if code in CLEF_MAP: clef_sign, clef_line = CLEF_MAP[code]
        elif tok.startswith("keySignature-"):
            fifths = KEY_FIFTHS.get(tok[13:], 0)
        elif tok.startswith("timeSignature-"):
            sig = tok[14:]
            if sig == "C": beats_per, beat_type = 4.0, 4
            elif sig == "C/": beats_per, beat_type = 2.0, 2
            else:
                try:
                    b, bt = sig.split("/")
                    beats_per, beat_type = float(b), int(bt)
                except: pass
        elif tok.startswith("multirest-"):
            flush()
            try: n = int(tok[10:])
            except: n = 1
            measures.append([("multirest", {"count": n})])
            accum = 0.0
        elif tok == "barline":
            flush()
        elif tok == "tie":
            for i in range(len(current)-1, -1, -1):
                if current[i][0] == "note":
                    current[i][1]["tie_stop"] = True; break
        elif tok.startswith("note-"):
            p = parse_note(tok, "note-")
            if p:
                current.append(("note", p)); accum += p["beats"]
                if accum >= beats_per - 0.001: flush()
        elif tok.startswith("gracenote-"):
            p = parse_note(tok, "gracenote-")
            if p: current.append(("grace", p))
        elif tok.startswith("rest-"):
            p = parse_rest(tok)
            if p:
                current.append(("rest", p)); accum += p["beats"]
                if accum >= beats_per - 0.001: flush()

    if current: flush()
    if not measures: measures = [[]]

    #  Generar XML como string 
    L = []
    L.append('<?xml version="1.0" encoding="UTF-8"?>')
    L.append('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"')
    L.append('  "http://www.musicxml.org/dtds/partwise.dtd">')
    L.append('<score-partwise version="3.1">')
    L.append(f'  <work><work-title>{title}</work-title></work>')
    L.append('  <identification><encoding>')
    L.append('    <software>OMR CRNN Model</software>')
    L.append('  </encoding></identification>')
    L.append('  <part-list>')
    L.append('    <score-part id="P1"><part-name>Music</part-name></score-part>')
    L.append('  </part-list>')
    L.append('  <part id="P1">')

    real_num = 1
    for m_idx, events in enumerate(measures):
        is_mr = len(events) == 1 and events[0][0] == "multirest"

        if is_mr:
            n = events[0][1]["count"]
            dur = int(beats_per * DIVISIONS)
            L.append(f'    <measure number="{real_num}" implicit="yes">')
            L.append(f'      <attributes>')
            if m_idx == 0:
                L.append(f'        <divisions>{DIVISIONS}</divisions>')
                L.append(f'        <key><fifths>{fifths}</fifths></key>')
                L.append(f'        <time><beats>{int(beats_per)}</beats><beat-type>{beat_type}</beat-type></time>')
                L.append(f'        <clef><sign>{clef_sign}</sign><line>{clef_line}</line></clef>')
            L.append(f'        <measure-style><multiple-rest>{n}</multiple-rest></measure-style>')
            L.append(f'      </attributes>')
            L.append(f'      <note><rest measure="yes"/><duration>{dur}</duration><type>whole</type></note>')
            L.append(f'    </measure>')
            continue

        L.append(f'    <measure number="{real_num}">')
        real_num += 1

        if m_idx == 0:
            L.append(f'      <attributes>')
            L.append(f'        <divisions>{DIVISIONS}</divisions>')
            L.append(f'        <key><fifths>{fifths}</fifths></key>')
            L.append(f'        <time><beats>{int(beats_per)}</beats><beat-type>{beat_type}</beat-type></time>')
            L.append(f'        <clef><sign>{clef_sign}</sign><line>{clef_line}</line></clef>')
            L.append(f'      </attributes>')

        for kind, data in events:
            L.append('      <note>')
            if kind == "grace": L.append('        <grace/>')
            if kind in ("note","grace"):
                L.append('        <pitch>')
                L.append(f'          <step>{data["step"]}</step>')
                if data["alter"] != 0:
                    L.append(f'          <alter>{data["alter"]}</alter>')
                L.append(f'          <octave>{data["octave"]}</octave>')
                L.append('        </pitch>')
            else:
                L.append('        <rest/>')
            if kind != "grace":
                L.append(f'        <duration>{max(1,int(round(data["beats"]*DIVISIONS)))}</duration>')
            L.append(f'        <type>{data["type"]}</type>')
            for _ in range(data.get("dots",0)):
                L.append('        <dot/>')
            if data.get("fermata") or data.get("tie_stop"):
                L.append('        <notations>')
                if data.get("fermata"): L.append('          <fermata/>')
                if data.get("tie_stop"): L.append('          <tied type="stop"/>')
                L.append('        </notations>')
            if data.get("tie_stop"): L.append('        <tie type="stop"/>')
            L.append('      </note>')

        L.append('    </measure>')

    L.append('  </part>')
    L.append('</score-partwise>')
    return "\n".join(L)

#  Interfaz principal 
st.divider()

uploaded = st.file_uploader(
    "📂 Sube tu imagen de partitura",
    type=["png", "jpg", "jpeg"],
    help="Imagen en escala de grises, fondo blanco, notas negras"
)

if uploaded:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Imagen subida")
        st.image(uploaded, use_column_width=True)

    img_bytes = uploaded.read()
    img_proc  = preprocess_image(img_bytes)

    if img_proc is None:
        st.error("No se pudo leer la imagen.")
        st.stop()

    with st.spinner("Analizando partitura..."):
        tokens, confidence = predict(model, img_proc, idx2tok, BLANK_IDX)

    with col2:
        st.subheader("Resultado")
        st.metric("Tokens detectados", len(tokens))
        st.metric("Log-probabilidad", f"{confidence:.4f}")

        st.subheader("Secuencia de tokens")
        st.code(" ".join(tokens), language=None)

    st.divider()
    st.subheader("Tokens individuales")
    cols = st.columns(8)
    for i, tok in enumerate(tokens):
        cols[i % 8].markdown(
            f'<div style="background:#1e3a5f;color:#7eb8f7;padding:4px 6px;'
            f'border-radius:6px;font-size:11px;margin:2px;text-align:center">'
            f'{tok}</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.subheader("⬇Descargar MusicXML")

    title_input = st.text_input("Título de la partitura", value="Mi Partitura OMR")

    xml_str = tokens_to_musicxml(tokens, title=title_input)

    st.download_button(
        label="Descargar .musicxml (MuseScore)",
        data=xml_str.encode("utf-8"),
        file_name="partitura_omr.musicxml",
        mime="application/vnd.recordare.musicxml+xml"
    )

    with st.expander("Ver XML generado"):
        st.code(xml_str, language="xml")