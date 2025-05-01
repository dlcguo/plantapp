import cv2
import datetime
import google.generativeai as genai
import io
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
import streamlit as st
import sqlite3

from plantapp.config import cfg
from plantapp.utils.predict import load_model, predict

# This file is the main app for the plantapp project

FEED_DIR   = pathlib.Path("feedback")
FEED_DIR.mkdir(exist_ok=True)
DB_PATH    = FEED_DIR / "reports.db"

def _init_db():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS misreports (
                   id            INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts_utc        TEXT,
                   plant_pred    TEXT,
                   disease_pred  TEXT,
                   plant_conf    REAL,
                   disease_conf  REAL,
                   user_note     TEXT,
                   image_png     BLOB
               )"""
        )
_init_db()

def save_feedback_sql(img, plant_pred, disease_pred,
                      plant_conf, disease_conf, user_note):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO misreports
            (ts_utc, plant_pred, disease_pred, plant_conf,
             disease_conf, user_note, image_png)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                datetime.datetime.utcnow().isoformat(timespec="seconds"),
                plant_pred,
                disease_pred,
                float(plant_conf) if plant_conf is not None else None,
                float(disease_conf) if disease_conf is not None else None,
                user_note,
                png_bytes,
            ),
        )
    st.cache_data.clear()   

st.set_page_config(page_title="Plant-Disease Classifier", layout="wide")
st.markdown(
    """
    <style>
        .stApp { 
            background:#F5E9D8; 
        }

        .block-container  { 
            padding-top:2rem; 
        }

        img { 
            border-radius:10px; display:block; margin:auto; 
        }

        html, body, [class^="st-"], [class*=" st-"] {
            color:#000 !important;        
        }

        h1, h2, h3, h4, p, .stMarkdown, .stMarkdown p, .stMarkdown ul  {
            text-align:center; color:#000 !important;
        }

        .stMarkdown ul { 
            list-style-position:inside; 
            }

        figcaption { 
            text-align:center; color:#000 !important; 
        }

        div[data-testid="stImage"] { 
            display:flex; justify-content:center; 
        }
        
        div[data-testid="stImage"] img 
            { margin:auto;
        }   

        div[data-testid="stFileUploader"] > section {
            background:#EEE9DA;          
            border:2px dashed #555;      
            color:#000;                  
        }

        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] span {
            color:#000 !important;
        }

        div[data-testid="column"] > div {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        code, kbd, samp {
            background:#E0E0E0 !important;   
            color:#000 !important;         
            padding:2px 4px;
            border-radius:4px;
            font-size:90%;
        }

        div[data-testid="stFileUploader"] button {
            background:#4CAF50 !important;  
            color:#fff !important;         
            border:none !important;
        }

        div.stButton > button {           
        background:#4CAF50 !important; 
        color:#fff !important;       
        border:none !important;
        border-radius:4px !important;
        padding:0.4rem 1rem !important;
        }

        div.stButton > button:hover {
            background:#45a044 !important;
        }

        textarea {
            background-color:#FFF !important;
            color:#000 !important;
        }

        div[data-testid="stForm"] div.stButton {         
            display:flex;
            justify-content:center;      
        }
        div[data-testid="stForm"] div.stButton > button { 
            margin:auto;                
        }   

        .stFormSubmitButton>button {
            background:#4CAF50 !important; 
            color:#fff !important;       
            border:none !important;
            border-radius:4px !important;
            padding:0.4rem 1rem !important;
            justify-content:center;      
        }

        .stFormSubmitButton>button:hover {
            background:#45a044 !important;
            color:#fff !important;       
        }
    </style>
    """,
    unsafe_allow_html=True,
)

genai.configure(api_key=cfg.gemini_api_key or None)
@st.cache_data(show_spinner=False)
def query_gemini(disease: str) -> str:
    prompt = (f"You are an experienced agronomist. Provide detailed treatment advice for '{disease}' in paragraph form in language that a gardener hobbyist would understand. Keep it under 160 words.")
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    rsp = model.generate_content(prompt, safety_settings={"HARASSMENT":"block_none"})
    return rsp.text.strip()

@st.cache_resource
def load():
    return load_model()
model = load()

left, mid, right = st.columns([3, 2, 3])
with mid:

    st.title("Plant-Disease Classifier")

    img = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

    if img:
        pil = Image.open(img).convert("RGB")
        st.image(pil, caption="Input", use_container_width=True)

        with st.spinner("Predicting …"):
            try:
                out = predict(model, pil)
                plant_idx, dis_idx, cam, plant_prob, dis_prob = out
            except ValueError:
                plant_idx, dis_idx, cam = predict(model, pil)
                plant_prob = dis_prob = None
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        plant = cfg.plant_names[plant_idx]
        disease = cfg.disease_names[dis_idx]

        st.subheader("Prediction")
        st.write(
            f"**Plant:** {plant} (ID {plant_idx}) "
            f"{'' if plant_prob is None else f'• confidence {plant_prob:.1%}'}"
        )
        st.write(
            f"**Status:** {disease} (ID {dis_idx}) "
            f"{'' if dis_prob is None else f'• confidence {dis_prob:.1%}'}"
        )

        if not st.session_state.get("feedback_done"):
            if st.button("Report misclassification"):
                st.session_state["show_form"] = True

        # Show the form until the user clicks submit
        if st.session_state.get("show_form"):
            with st.expander("Submit correction", expanded=True):
                with st.form("feedback_form", clear_on_submit=True):
                    note = st.text_area(
                        "Describe what seems wrong (e.g. correct class, lighting, etc.) as well as the expected  characteristics",
                        height=100,
                        placeholder="Example: Should be 'Tomato – Late blight', image at dusk."
                    )
                    sent = st.form_submit_button("Submit")
                    if sent:
                        save_feedback_sql(pil, plant, disease,
                                        plant_prob, dis_prob, note)
                        st.session_state["feedback_done"] = True   
                        st.session_state["show_form"] = False   
                        st.rerun()                    

        if st.session_state.get("feedback_done"):
            st.success("Thank you for your correction, we will do our best to improve the model.")

        if disease.lower() not in {"healthy","n/a","none"} and cfg.gemini_api_key:
            with st.spinner("Getting treatment advice..."):
                advice = query_gemini(disease)
            st.subheader("Suggested Treatment (Gemini)")
            st.markdown(f"<div style='text-align:center'>{advice}</div>",
                        unsafe_allow_html=True)
        elif disease.lower() == "healthy":
            st.success("Plant looks healthy! No treatment needed.")
        else:
            st.info("No plant detected or status unavailable — please try another image.")

        st.markdown(
            "<div style='text-align:center; font-size:0.9rem; margin-top:0.8rem;'>"
            "<em>Disclaimer — Treatment guidance and plant identification are automatically generated via "
            "Google Gemini and a ResNet backbone respectively. For critical decisions always consult a qualified "
            "agronomist or plant-health professional.</em>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Areas of Interest (Grad-CAM)")
        img_np = np.array(pil.resize((cfg.img_size, cfg.img_size))) / 255.0
        heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
        st.image(0.5*img_np + 0.5*heat, caption="Grad-CAM", use_container_width=True)
        
        # For debugging purposes of feedback results
        # with sqlite3.connect("feedback/reports.db") as conn:
        #     df = pd.read_sql_query(
        #         "SELECT ts_utc, plant_pred, disease_pred, plant_conf, disease_conf, user_note "
        #         "FROM misreports ORDER BY ts_utc DESC LIMIT 20", conn
        #     )
        # st.dataframe(df, use_container_width=True)