"""
╔══════════════════════════════════════════════════════════════════════╗
║  Dashboard Streamlit — PJJ Analisis Masukan Vertikal                ║
║  Seluruh logika analisis IDENTIK dengan file PJJ asli               ║
║  (NMF, Grid Search, Regional, Visualisasi)                          ║
╚══════════════════════════════════════════════════════════════════════╝
Jalankan:
    pip install streamlit plotly wordcloud scikit-learn openpyxl
              folium streamlit-folium branca geopandas
    streamlit run dashboard_pjj_analisis.py
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import re, warnings, itertools, time, random
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# ── Library opsional (graceful fallback jika tidak terinstall) ────────────────
def _try_import(name):
    try: return __import__(name), True
    except ImportError: return None, False

_bertopic_mod, HAS_BERTOPIC = _try_import("bertopic")
_top2vec_mod,  HAS_TOP2VEC  = _try_import("top2vec")
_corex_mod,    HAS_COREX    = _try_import("corextopic")

import json as _js, base64 as _b64, zlib as _zl
import folium, branca.colormap as _bcm
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PJJ · Analisis Masukan Vertikal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&family=DM+Mono:wght@400&display=swap');
:root{
    --bg:#0b0d12; --card:#12151e; --hover:#191d2a; --border:#1d2236;
    --a1:#5b8dee; --a2:#3ecfac; --a3:#f5a623; --a4:#f06b6b; --a5:#a78bfa;
    --hi:#edf0f8; --md:#8b95b0; --lo:#434b66; --r:10px;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif !important;color:var(--hi) !important;}
.stApp{background:var(--bg) !important;}
section[data-testid="stSidebar"]{background:#080a0f !important;border-right:1px solid var(--border);}
.block-container{padding:1.4rem 2rem !important;max-width:1680px;}
#MainMenu,footer,header{visibility:hidden;}
h1,h2,h3{font-family:'Syne',sans-serif !important;font-weight:800 !important;}
.sh{font-family:'Syne',sans-serif;font-size:.95rem;font-weight:700;color:var(--hi);
    display:flex;align-items:center;gap:.55rem;
    padding:.55rem 0 .35rem;border-bottom:1px solid var(--border);margin:1.1rem 0 .7rem;}
.sh .d{width:7px;height:7px;border-radius:50%;background:var(--a1);flex-shrink:0;}
.sw{display:inline-block;background:var(--hover);border:1px solid var(--border);
    color:var(--md);border-radius:5px;font-size:.68rem;
    padding:.08rem .45rem;margin:.12rem;font-family:'DM Mono',monospace;}
.stTabs [data-baseweb="tab-list"]{gap:.4rem;background:var(--card) !important;
    border-radius:var(--r);padding:.25rem .35rem;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:'Syne',sans-serif !important;font-size:.82rem !important;
    font-weight:700 !important;color:var(--md) !important;border-radius:7px !important;
    padding:.35rem .85rem !important;background:transparent !important;border:none !important;}
.stTabs [aria-selected="true"]{color:var(--hi) !important;background:var(--hover) !important;
    border-bottom:2px solid var(--a1) !important;}
section[data-testid="stSidebar"] label{color:var(--md) !important;font-size:.8rem !important;}
section[data-testid="stSidebar"] .stButton>button{
    background:linear-gradient(135deg,var(--a1),#3d6fd4) !important;
    color:white !important;border:none !important;
    font-family:'Syne',sans-serif !important;font-weight:700 !important;
    border-radius:8px !important;transition:all .18s !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTA — identik dengan file PJJ asli
# ─────────────────────────────────────────────────────────────────────────────
COLORS = [
    '#2E86AB','#A23B72','#F18F01','#C73E1D','#44BBA4',
    '#8B5E83','#3B1F2B','#E94F37','#1A936F','#C6A84B','#5C6BC0','#D62839'
]

WARNA_PULAU = {
    'Jawa'         : '#2E86AB', 'Sumatera'     : '#F18F01',
    'Jakarta'      : '#C73E1D', 'Kalimantan'   : '#44BBA4',
    'Sulawesi'     : '#A23B72', 'Bali'         : '#E94F37',
    'Papua'        : '#1A936F', 'Nusa Tenggara': '#8B5E83',
}

PULAU_ORDER = ['Jawa','Sumatera','Jakarta','Kalimantan',
               'Sulawesi','Bali','Papua','Nusa Tenggara']

STOPWORDS_ID = set([
    'yang','dan','di','ke','dari','untuk','dengan','ini','itu','ada',
    'tidak','pada','dalam','atau','juga','sudah','agar','bisa',
    'dapat','lebih','akan','saat','sehingga','jika','apabila','ketika',
    'mohon','kami','kita','saya','user','perlu','harus','masukan',
    'tolong','atas','hal','seperti','karena','namun','tetapi','tapi',
    'dimana','bagaimana','setelah','sebelum','oleh','telah','belum',
    'pernah','sering','selalu','kembali','klik','menu','tombol',
    'tampilan','proses','data','sistem','aplikasi','coretax','ijin',
    'bapak','ibu','ka','si','ya','ok','oke','baik','sehingga',
    'terdapat','dilakukan','dibuat','dibuatkan','ditambahkan',
    'setiap','hanya','wajib','pajak','per','wp','ar','seksi',
    'kepala','nama','kasi','pasal','masa','tahun','jenis','spt','nomor',
    'saja','satu','beberapa','lagi','masih','sebaiknya','tersebut','terkait',
    'sebagai','melalui','melakukan','seharusnya','nilai','lanjut','waktu','sama',
    'sesuai','yg','bukan','cukup','semua','awal','baru','tanpa','langsung',
    'hasil','muncul','sederhana','disediakan','usul','menjadi','diberikan','masuk',
    'sangat','pilihan','salah','memudahkan','menampilkan','diisi','perubahan',
    'dimunculkan','menggunakan','membuat','mana','ulang','memilih','selesai',
])

INTERPRETASI_DEFAULT = {
    0 : "Otomasi vs Input Manual & Kesalahan",
    1 : "Penerbitan STP & Nothit",
    2 : "Manajemen Kasus & Pembatalan",
    3 : "Pembayaran, Pelaporan & Realisasi",
    4 : "Faktur & Bukti Potong",
    5 : "Dafnom STP & Keterlambatan Lapor",
    6 : "Keluhan UX — Alur Terlalu Panjang",
    7 : "Dashboard Pengawasan & Monitoring",
    8 : "Approweb & Sistem Legacy",
    9 : "SP2DK / LHP2DK / BAP2DK",
    10: "Kertas Kerja & EWS",
    11: "Alur Persetujuan",
}

# ─────────────────────────────────────────────────────────────────────────────
# GEOJSON INDONESIA (Natural Earth, embedded)
# ─────────────────────────────────────────────────────────────────────────────
import base64 as _b64, zlib as _zl, json as _js
_GEO_B64 = (
    "eNqcnUuLhslxpf9Lr0tF3i/eGmYxw4BhlsKLxtNjBG21kFsYY/zf5zwnMvNrBoytkSyrVRX1XvLN"
    "jOuJE//2w6//+qeffvibH/7bTz/++pc///S3v/z880//8OsffvnjD18//J/42T//8De//7f/R06/"
    "/dOff/nTT3/+9Q/8/t9++NNffv7xL/r9f//xX3784d+/fvjHn375p59+/fO/8rvzp3/3y8//+o++"
    "8D/88suf//cf/vjjr77273+f0/pua6w2cmuj7TK+fje+W8l7td1XL322v/+yWC85pVbGqLvljVjP"
    "aaeZ565b8vnK9Z5HqWtl/Thbbu9Vdkup9FVKvXJr15J62mu1VJEba03+NufCwxw5XavNNap+sbYf"
    "b845Zl16Ct0+9SM3Wy257NKLXqNYbtWdWlp59bX2vd4aepZZ8lpjd4vtlCt/PMtMqxwxPfJYuetR"
    "1p5x2z3qaH3kvnuZ8br7O6fScl9pZC2PX2PpPXdbKbdR2zhipZTcU9p71lRziK1Vemq56tXaPnK1"
    "6cFa3bX21Jfl9He9l5X1uquuI6cPNrI+3Nh6Zr/G0tuvzlcaeoLRj2DXd+hlr9T1NUoIapX7HGlX"
    "rfWVm0lvrRXQt9eHs1yts42U0kqtpXbkVtEa5T60uHmMkNNbjdLSmHrxeuT0vbUTlvZVzzvWpY2S"
    "hhamtzRjnXP6TiMVfYDe09QSWU5fsOpafLod6yy5vLR/WtNm3fokIbfGGFt/Pyff5AgWPU1feo+a"
    "9aIWHHzDkvRFem3ryGn7tK2PoUeuKRZmaoWm3lX7odd55HrVdpq17N5riRvPWbV+UztXe2Zcud2q"
    "DlEadY5WY5/yk6WtuvOs90X0rZueRMu3d5/e91O31dKwEWe915ta9bW6PoBuH+do6PjoiGZ9Sb3f"
    "ldOrz66tuVqPndBG19ZovW/t6XbfQ2dL21YfpOv8+L464aXuXnOarNCR21sPp48+atrD7ystwG3T"
    "YElr3Ddr43Piqv6E/W85rZDOhzbc6jv0huR04nVWu1Zs1hZ6Y2lN9TiDjZDnkats+qFtqe3evCy5"
    "a++wzbb+YR+xxllYW5ukztAa+klqW7u+p5Lu03WtactFB24cbTD0/l3fTPtAX+A+3dBH3zr9Tfs/"
    "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"
    "l+PK/9P//C+WEiYw"
)
_IDN_GEO = _js.loads(_zl.decompress(_b64.b64decode(_GEO_B64)))

_PULAU_CENTER = {
    "Jawa":(-7.34,110.18),"Sumatera":(-0.41,101.55),
    "Kalimantan":(-0.25,114.02),"Sulawesi":(-2.11,121.20),
    "Papua":(-4.11,136.87),"Nusa Tenggara":(-9.11,121.03),
    "Bali":(-8.38,115.03),"Jakarta":(-6.19,106.83),
}

def buat_peta_choropleth(pulau_dict, c0="#061224", c1="#1e4080", c2="#5b8dee", height=480):
    """Peta Folium choropleth — banyak data = gelap, sedikit = terang."""
    vals = {p: d["n"] for p, d in pulau_dict.items()}
    vmin, vmax = min(vals.values()) if vals else 0, max(vals.values()) if vals else 1

    m = folium.Map(location=[-2.5,118], zoom_start=4,
                   tiles="CartoDB dark_matter", prefer_canvas=True)
    cmap = _bcm.LinearColormap([c2, c1, c0], vmin=vmin, vmax=vmax, caption="Jumlah Masukan")
    cmap.add_to(m)

    for feat in _IDN_GEO["features"]:
        pulau = feat["properties"]["pulau"]
        val   = vals.get(pulau, 0)
        n_    = pulau_dict.get(pulau, {}).get("n", 0)
        pct_  = pulau_dict.get(pulau, {}).get("pct", 0.0)
        color = cmap(val)

        folium.GeoJson(feat,
            style_function=lambda x, c=color: {
                "fillColor":c,"color":"#7aaac8","weight":0.6,"fillOpacity":0.85,"opacity":0.4},
            highlight_function=lambda x: {"fillOpacity":0.97,"weight":2.0,"color":"#fff"},
            tooltip=folium.Tooltip(
                f"<b>{pulau}</b> · {n_:,} item ({pct_:.1f}%)",
                style="font-family:sans-serif;font-size:13px;background:rgba(10,12,18,.92);"
                      "color:#eef0f6;border:1px solid #2d3a5c;border-radius:6px;padding:4px 8px"),
        ).add_to(m)

        center = _PULAU_CENTER.get(pulau)
        if center:
            folium.Marker(center, icon=folium.DivIcon(
                html=f"<div style='font-family:sans-serif;font-weight:800;font-size:11px;"
                     f"color:#fff;text-shadow:0 0 5px rgba(0,0,0,1),0 1px 4px rgba(0,0,0,.9);"
                     f"white-space:nowrap'>{pulau}<br>"
                     f"<span style='font-weight:400;font-size:9.5px'>{n_:,}</span></div>",
                icon_size=(140,36), icon_anchor=(70,18))).add_to(m)
    return m

# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI BANTU — identik dengan file PJJ asli
# ─────────────────────────────────────────────────────────────────────────────
def pisah_masukan(teks):
    """Identik dengan fungsi pisah_masukan di file PJJ asli."""
    teks = str(teks).strip()
    if re.search(r'(?:^|\n)\s*2[\.\)]\s', teks):
        parts = re.split(r'(?:^|\n)\s*\d{1,2}[\.\)]\s+', teks)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 8]
        if parts:
            return parts
    if '>' in teks and re.search(r'(?:^|\n)\s*>', teks):
        parts = re.split(r'\n\s*>', teks)
        parts = [re.sub(r'^>', '', p).strip() for p in parts]
        parts = [p for p in parts if p and len(p) > 8]
        if len(parts) >= 2:
            return parts
    if teks.count('\n') >= 2 and len(teks) > 150:
        parts = [p.strip() for p in teks.split('\n')
                 if p.strip() and len(p.strip()) > 15]
        if len(parts) >= 2:
            return parts
    return [teks]

def bersihkan_nlp(t):
    """Identik dengan fungsi bersihkan_nlp di file PJJ asli."""
    t = str(t).lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\b\d+\b', '', t)
    return re.sub(r'\s+', ' ', t).strip()

def top_kata(components, fn, n=10):
    """Identik dengan fungsi top_kata di file PJJ asli."""
    return {
        i: [fn[j] for j in comp.argsort()[-n:][::-1]]
        for i, comp in enumerate(components)
    }

def ringkasan_kekuatan(df, N_TOPICS, NAMA_TOPIK):
    """Identik dengan fungsi ringkasan_kekuatan di file PJJ asli."""
    rows = []
    for i in range(N_TOPICS):
        sub = df[df['topik_id'] == i]['kekuatan']
        rows.append({
            'topik_id': i, 'label': f"T{i+1}: {NAMA_TOPIK[i][:25]}",
            'mean': sub.mean(), 'median': sub.median(),
            'std': sub.std(), 'min': sub.min(), 'max': sub.max(), 'n': len(sub),
        })
    return pd.DataFrame(rows)

def get_dev(deviasi, pulau, topik_idx):
    """Identik dengan fungsi get_dev di file PJJ asli."""
    tcol = f"T{topik_idx+1}"
    if pulau in deviasi.index and tcol in deviasi.columns:
        return float(deviasi.loc[pulau, tcol])
    return 0.0

def fmt_dev(v):
    """Identik dengan fungsi fmt_dev di file PJJ asli."""
    return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"

def plotly_dark():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8b95b0", size=11),
        title_font=dict(family="Syne", color="#edf0f8", size=13),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b95b0")),
        xaxis=dict(gridcolor="#1d2236", zerolinecolor="#1d2236", color="#8b95b0"),
        yaxis=dict(gridcolor="#1d2236", zerolinecolor="#1d2236", color="#8b95b0"),
        margin=dict(l=8, r=8, t=42, b=8),
    )

def sh(icon, title):
    st.markdown(f'<div class="sh"><span class="d"></span>{icon} {title}</div>',
                unsafe_allow_html=True)

def buf_img(fig):
    """Simpan matplotlib figure ke BytesIO."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

def mpl_to_st(fig, caption="", use_container_width=True):
    """Tampilkan matplotlib figure di Streamlit."""
    buf = buf_img(fig)
    st.image(buf, caption=caption, use_container_width=use_container_width)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "stopwords": set(STOPWORDS_ID),
        "done": False,
        "df": None,
        "topics": None,
        "NAMA_TOPIK": {},
        "N_TOPICS": 8,
        "run_id": 0,
        "hasil": {},
        "history": {},   # {teknik: {sil, K, topics, dist}}
        "anchor_words": {},  # untuk CorEx
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ─────────────────────────────────────────────────────────────────────────────
# KATALOG TEKNIK PEMBANDING
# ─────────────────────────────────────────────────────────────────────────────
TEKNIK_KATALOG = {
    "NMF"          : {"full":"Non-negative Matrix Factorization","vec":"TF-IDF",
                      "color":"#5b8dee","req":[],"need_k":True,
                      "desc":"Topik sparse & interpretatif. Terbaik untuk teks pendek. (Utama — tidak berubah)"},
    "LDA"          : {"full":"Latent Dirichlet Allocation","vec":"Count",
                      "color":"#3ecfac","req":[],"need_k":True,
                      "desc":"Probabilistik klasik. Soft-assignment — satu dokumen bisa masuk banyak topik. Cocok untuk dokumen panjang."},
    "LSA"          : {"full":"Latent Semantic Analysis","vec":"TF-IDF",
                      "color":"#f5a623","req":[],"need_k":True,
                      "desc":"SVD pada TF-IDF. Cepat, menangkap sinonim & relasi semantik laten."},
    "KMeans"       : {"full":"K-Means Clustering","vec":"TF-IDF",
                      "color":"#a78bfa","req":[],"need_k":True,
                      "desc":"Hard-assignment cepat. Mudah dijelaskan ke stakeholder non-teknis."},
    "GMM"          : {"full":"Gaussian Mixture Model","vec":"TF-IDF+PCA",
                      "color":"#34d399","req":[],"need_k":True,
                      "desc":"Soft-assignment probabilistik. Klaster elips, cocok data heterogen."},
    "Agglomerative": {"full":"Agglomerative Hierarchical","vec":"TF-IDF",
                      "color":"#fb923c","req":[],"need_k":True,
                      "desc":"Hierarki topik dari bawah ke atas. Berguna untuk eksplorasi struktur data."},
    "BERTopic"     : {"full":"BERTopic (Transformer+HDBSCAN)","vec":"Sentence Embedding",
                      "color":"#f472b6","req":["bertopic"],"need_k":False,
                      "desc":"State-of-the-art. Memahami makna kontekstual. Tidak perlu K."},
    "Top2Vec"      : {"full":"Top2Vec","vec":"Doc Embedding",
                      "color":"#60a5fa","req":["top2vec"],"need_k":False,
                      "desc":"Embedding dokumen langsung ke ruang topik. Otomatis temukan K."},
    "CorEx"        : {"full":"Correlation Explanation","vec":"Count Binary",
                      "color":"#facc15","req":["corextopic"],"need_k":True,
                      "desc":"Semi-supervised dengan anchor words. Cocok domain spesifik (pajak, hukum)."},
}

def cek_tersedia(teknik):
    return all(
        _try_import(r)[1]
        for r in TEKNIK_KATALOG[teknik]["req"]
    )


@st.cache_data(show_spinner=False)
def run_teknik_pembanding(_teks_t, teknik, K, max_f, min_d, ngram_t, sw_t, anchor_t):
    """Jalankan teknik pembanding. NMF dijalankan terpisah via alur utama PJJ."""
    teks  = list(_teks_t)
    sw    = list(sw_t)
    ngram = ngram_t
    anchors = {int(k): v.split() for k, v in dict(anchor_t).items() if v.strip()}

    # ── Vektorisasi ──────────────────────────────────────────────────────
    if teknik in ("LDA",):
        vec = CountVectorizer(max_features=max_f, min_df=min_d,
                              ngram_range=ngram, stop_words=sw)
        X   = vec.fit_transform(teks); fn = vec.get_feature_names_out()
    elif teknik in ("CorEx",):
        vec = CountVectorizer(max_features=max_f, min_df=min_d,
                              ngram_range=(1,1), stop_words=sw, binary=True)
        X   = vec.fit_transform(teks); fn = vec.get_feature_names_out()
    elif teknik in ("BERTopic","Top2Vec"):
        X = None; fn = None
    else:
        vec = TfidfVectorizer(max_features=max_f, min_df=min_d,
                              ngram_range=ngram, stop_words=sw)
        X   = vec.fit_transform(teks); fn = vec.get_feature_names_out()

    # ── Model ────────────────────────────────────────────────────────────
    def _sort_topik(topik_id, topics_raw, K):
        freq      = {i:(topik_id==i).sum() for i in range(K)}
        order     = sorted(range(K), key=lambda x:-freq[x])
        o2n       = {o:n for n,o in enumerate(order)}
        n2o       = {n:o for n,o in enumerate(order)}
        topik_id  = np.array([o2n[t] for t in topik_id])
        topics    = {i: topics_raw[n2o[i]] for i in range(K)}
        return topik_id, topics

    def _top(comps, fn_arr, n=10):
        return {i:[fn_arr[j] for j in comp.argsort()[-n:][::-1]]
                for i,comp in enumerate(comps)}

    def _sil(X, labels):
        try:
            return round(silhouette_score(X, labels,
                         sample_size=min(800,len(teks)), random_state=42), 4)
        except: return 0.0

    if teknik == "LDA":
        m   = LatentDirichletAllocation(n_components=K, random_state=42,
                                         max_iter=30, learning_method="online")
        W   = m.fit_transform(X)
        raw = _top(m.components_, fn)
        tid = W.argmax(axis=1); kuat = W.max(axis=1)
        tid, topics = _sort_topik(tid, raw, K)
        sil = _sil(X, tid)

    elif teknik == "LSA":
        m   = TruncatedSVD(n_components=K, random_state=42)
        W   = np.abs(m.fit_transform(X))
        raw = _top(np.abs(m.components_), fn)
        tid = W.argmax(axis=1); kuat = W.max(axis=1)
        tid, topics = _sort_topik(tid, raw, K)
        sil = _sil(X, tid)

    elif teknik == "KMeans":
        Xn  = normalize(X)
        m   = KMeans(n_clusters=K, random_state=42, n_init=15)
        lbl = m.fit_predict(Xn)
        dist= m.transform(Xn)
        W   = normalize(1/(dist+1e-9), norm="l1")
        comps = np.zeros((K, X.shape[1]))
        for ki in range(K):
            mask = lbl==ki
            if mask.sum()>0:
                comps[ki] = np.asarray(X[mask].mean(axis=0)).flatten()
        raw = _top(comps, fn)
        tid = W.argmax(axis=1); kuat = W.max(axis=1)
        tid, topics = _sort_topik(tid, raw, K)
        sil = _sil(X, tid)

    elif teknik == "GMM":
        n_pca = min(50, X.shape[1]-1, X.shape[0]-1)
        Xd    = TruncatedSVD(n_components=n_pca, random_state=42).fit_transform(X)
        m     = GaussianMixture(n_components=K, random_state=42,
                                covariance_type="full", max_iter=200)
        m.fit(Xd); W = m.predict_proba(Xd)
        comps = np.zeros((K, X.shape[1]))
        for ki in range(K):
            w   = W[:, ki]; tot = w.sum()
            if tot > 0:
                comps[ki] = np.asarray((X.T @ w)).flatten() / tot
        raw = _top(comps, fn)
        tid = W.argmax(axis=1); kuat = W.max(axis=1)
        tid, topics = _sort_topik(tid, raw, K)
        sil = _sil(X, tid)

    elif teknik == "Agglomerative":
        from sklearn.metrics import pairwise_distances
        n_pca  = min(50, X.shape[1]-1, X.shape[0]-1)
        Xd     = TruncatedSVD(n_components=n_pca, random_state=42).fit_transform(X)
        m      = AgglomerativeClustering(n_clusters=K, linkage="ward")
        lbl    = m.fit_predict(Xd)
        cents  = np.array([Xd[lbl==ki].mean(axis=0) for ki in range(K)])
        dist   = pairwise_distances(Xd, cents, metric="euclidean")
        W      = normalize(1/(dist+1e-9), norm="l1")
        comps  = np.zeros((K, X.shape[1]))
        for ki in range(K):
            mask = lbl==ki
            if mask.sum()>0:
                comps[ki] = np.asarray(X[mask].mean(axis=0)).flatten()
        raw = _top(comps, fn)
        tid = W.argmax(axis=1); kuat = W.max(axis=1)
        tid, topics = _sort_topik(tid, raw, K)
        sil = _sil(X, tid)

    elif teknik == "BERTopic":
        from bertopic import BERTopic
        m    = BERTopic(language="multilingual", nr_topics=K,
                        calculate_probabilities=True, verbose=False)
        lbl, probs = m.fit_transform(teks)
        W = probs if (probs is not None and len(probs.shape)==2) \
            else np.eye(K)[np.clip(np.array(lbl),0,K-1)]
        raw = {}
        for i in range(K):
            try: raw[i] = [w for w,_ in m.get_topic(i)[:10] if isinstance(w,str)]
            except: raw[i] = []
        tid = W.argmax(axis=1); kuat = W.max(axis=1)
        tid, topics = _sort_topik(tid, raw, K)
        sil = _sil(W, tid)

    elif teknik == "Top2Vec":
        from top2vec import Top2Vec
        m   = Top2Vec(teks, speed="fast-learn", workers=1)
        n_t = min(K, m.get_num_topics())
        raw = {i: list(m.topic_words[i][:10]) for i in range(n_t)}
        doc_top, _, _ = m.get_documents_topics(list(range(len(teks))), num_topics=1)
        lbl = np.array([t[0] if t[0]<K else 0 for t in doc_top])
        W   = np.zeros((len(teks), K))
        for ii,l in enumerate(lbl): W[ii,l] = 1.0
        tid = lbl; kuat = np.ones(len(teks))
        tid, topics = _sort_topik(tid, raw, K)
        sil = 0.0

    elif teknik == "CorEx":
        from corextopic import corextopic as ct
        Xbin = (X > 0).astype(float)
        anchor_list = [anchors.get(i,[]) for i in range(K)]
        m = ct.Corex(n_hidden=K, seed=42)
        m.fit(Xbin, words=fn, anchors=anchor_list, anchor_strength=3)
        W = np.array(m.p_y_given_x)
        if len(W.shape)==1: W = W.reshape(-1,1)
        raw = {}
        for i in range(K):
            try: raw[i] = [w for w,_ in m.get_topics(n_words=10)[i] if isinstance(w,str)]
            except: raw[i] = list(fn[:10])
        tid = W.argmax(axis=1); kuat = W.max(axis=1)
        tid, topics = _sort_topik(tid, raw, K)
        sil = _sil(W, tid)

    return tid, kuat, topics, sil


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.6rem 0 .4rem;text-align:center">
      <span style="font-family:Syne;font-size:1.2rem;font-weight:800;
          background:linear-gradient(135deg,#5b8dee,#3ecfac);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent">
          📊 PJJ Analitik
      </span>
      <div style="font-size:.65rem;color:#434b66;margin-top:.15rem">
          Analisis Masukan Vertikal · NMF + Grid Search
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # ── Upload ───────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:.66rem;text-transform:uppercase;letter-spacing:.1em;color:#5b8dee;font-weight:700;margin-bottom:.3rem">📂 Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Excel", type=["xlsx","xls"],
                                 label_visibility="collapsed")

    if uploaded:
        fb = uploaded.read()
        xl = pd.ExcelFile(BytesIO(fb))
        sheet = st.selectbox("Sheet", xl.sheet_names,
                              index=xl.sheet_names.index("Masukan Vertikal")
                              if "Masukan Vertikal" in xl.sheet_names else 0)
        scols = pd.read_excel(BytesIO(fb), sheet_name=sheet, nrows=0).columns.tolist()
        dcol  = next((c for c in scols if any(x in c.lower()
                      for x in ["permasalahan","masukan","keluhan"])), scols[0])
        col_m = st.selectbox("Kolom Teks", scols,
                              index=scols.index(dcol) if dcol in scols else 0)

        st.divider()
        st.markdown('<div style="font-size:.66rem;text-transform:uppercase;letter-spacing:.1em;color:#5b8dee;font-weight:700;margin-bottom:.3rem">⚙️ Konfigurasi Analisis</div>', unsafe_allow_html=True)

        # ── Teknik Utama ─────────────────────────────────────────────────
        TEKNIK_UTAMA_OPTS = {k: v for k, v in TEKNIK_KATALOG.items()
                             if k not in ("BERTopic","Top2Vec")}  # embedding butuh resource besar
        teknik_utama = st.selectbox(
            "Teknik Analisis Utama",
            list(TEKNIK_UTAMA_OPTS.keys()),
            index=0,
            format_func=lambda t: (
                f"{'✅' if cek_tersedia(t) else '⚠️'} {t} — {TEKNIK_KATALOG[t]['full'][:30]}"
            ),
            help="Teknik yang dipakai sebagai analisis utama. "
                 "Semua langkah lain (K optimal, Grid Search, Regional) tetap sama.",
        )
        info_utama = TEKNIK_KATALOG[teknik_utama]
        st.markdown(f"""
        <div style="background:#12151e;border-left:3px solid {info_utama['color']};
             border-radius:5px;padding:.45rem .7rem;margin:.2rem 0 .6rem;
             font-size:.73rem;color:#8b95b0;line-height:1.6">
            <b style="color:{info_utama['color']}">{teknik_utama}</b> · {info_utama['vec']}<br>
            {info_utama['desc']}
        </div>""", unsafe_allow_html=True)

        if not cek_tersedia(teknik_utama):
            st.warning(f"⚠️ pip install {' '.join(info_utama['req'])}")

        # Anchor words untuk CorEx utama
        anchor_utama = {}
        if teknik_utama == "CorEx" and cek_tersedia("CorEx"):
            st.markdown('<div style="font-size:.66rem;color:#facc15;font-weight:700;margin:.3rem 0 .1rem">⚓ Anchor Words</div>', unsafe_allow_html=True)
            k_tmp = st.session_state.get("N_TOPICS", 8)
            for i in range(min(k_tmp, 6)):
                w = st.text_input(f"T{i+1}", key=f"anc_utama_{i}",
                                   placeholder="pembayaran faktur",
                                   label_visibility="visible")
                anchor_utama[i] = w

        st.divider()

        # Grid Search — sesuai file PJJ asli (tidak berubah)
        run_grid = st.checkbox("Jalankan Grid Search TF-IDF", value=True,
                                help="Cari kombinasi max_features, min_df, ngram terbaik. "
                                     "Nonaktifkan untuk pakai parameter manual (lebih cepat).")

        if not run_grid:
            manual_max_f = st.selectbox("max_features", [100,200,300,400,500,700,1000,1500,2000], index=3)
            manual_min_d = st.selectbox("min_df", [2,3,5], index=1)
            manual_ngram = st.selectbox("ngram_range", ["(1,1)","(1,2)"], index=1)

        auto_k = st.checkbox("K optimal otomatis", value=True,
                              help="Cari K optimal via silhouette score (K=3..19)")
        if not auto_k:
            manual_k = st.slider("Jumlah Topik (K)", 3, 20,
                                  st.session_state.get("N_TOPICS", 8))
        else:
            manual_k = None
            if st.session_state.get("done"):
                st.caption(f"✅ K optimal terakhir: **{st.session_state['N_TOPICS']}** topik")

        st.caption("⚙️ Parameter lanjutan & stopwords → tab **✏️ Edit**")
        st.divider()

        # ── Teknik Pembanding ────────────────────────────────────────────
        st.markdown('<div style="font-size:.66rem;text-transform:uppercase;letter-spacing:.1em;color:#a78bfa;font-weight:700;margin-bottom:.3rem">⚖️ Teknik Pembanding</div>', unsafe_allow_html=True)
        st.caption("Jalankan teknik lain untuk dibandingkan dengan NMF. Settingan NMF tidak berubah.")

        semua_teknik = [t for t in TEKNIK_KATALOG if t != "NMF"]
        teknik_pilihan = st.multiselect(
            "Pilih teknik pembanding:",
            options=semua_teknik,
            default=[],
            format_func=lambda t: (
                f"{'✅' if cek_tersedia(t) else '⚠️'} {t} — {TEKNIK_KATALOG[t]['full'][:30]}"
            ),
            key="teknik_pembanding",
        )

        # Info teknik yang dipilih
        for t in teknik_pilihan:
            info = TEKNIK_KATALOG[t]
            avail = cek_tersedia(t)
            st.markdown(f"""
            <div style="background:#12151e;border-left:3px solid {info['color']};
                 border-radius:4px;padding:.4rem .7rem;margin:.2rem 0;
                 font-size:.72rem;color:#8b95b0">
                <b style="color:{info['color']}">{t}</b> · {info['vec']}<br>
                {info['desc']}
                {f'<br><span style="color:#f5a623">⚠️ pip install {" ".join(info["req"])}</span>' if not avail else ''}
            </div>""", unsafe_allow_html=True)

        # Anchor words untuk CorEx (tampil jika CorEx dipilih)
        if "CorEx" in teknik_pilihan and cek_tersedia("CorEx"):
            st.markdown('<div style="font-size:.66rem;color:#facc15;margin-top:.4rem;font-weight:700">⚓ Anchor Words CorEx</div>', unsafe_allow_html=True)
            k_tmp = manual_k if not auto_k else st.session_state.get("N_TOPICS", 8)
            for i in range(min(k_tmp, 6)):
                val = st.session_state["anchor_words"].get(i, "")
                new = st.text_input(f"Topik {i+1}", value=val, key=f"anc_{i}",
                                     placeholder="pembayaran faktur")
                st.session_state["anchor_words"][i] = new

        run_teknik_btn = st.button("⚖️ Jalankan Pembanding",
                                    use_container_width=True,
                                    disabled=len(teknik_pilihan)==0,
                                    key="btn_pembanding")
        st.divider()
        run_btn = st.button("▶ Jalankan Analisis", type="primary", use_container_width=True)
    else:
        run_btn = False; run_teknik_btn = False; teknik_pilihan = []; col_m = ""; fb = None

# ─────────────────────────────────────────────────────────────────────────────
# LANDING
# ─────────────────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem 1.5rem">
      <div style="font-family:Syne;font-size:2.5rem;font-weight:800;
          background:linear-gradient(135deg,#5b8dee,#3ecfac);
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.4rem">
          📊 Insight Map
      </div>
      <p style="color:#8b95b0;font-size:1rem">
          NMF Topic Modeling · Grid Search TF-IDF · Analisis Regional · Peta Indonesia
      </p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    infos = [
        ("🔍","Grid Search TF-IDF","54 kombinasi max_features × min_df × ngram, pilih yang silhouette terbaik"),
        ("🤖","NMF Topic Modeling","Dekomposisi matriks non-negatif, topik sparse & mudah diinterpretasi"),
        ("🗺️","Peta Indonesia","Choropleth interaktif distribusi masukan per pulau"),
        ("📊","Analisis Regional","Heatmap, stacked bar, bubble chart, dan temuan utama per wilayah"),
    ]
    for col, (icon,title,desc) in zip([c1,c2,c3,c4], infos):
        col.markdown(f"""
        <div style="background:#12151e;border:1px solid #1d2236;border-radius:10px;
             padding:.9rem 1rem;height:100%">
          <div style="font-size:1.5rem">{icon}</div>
          <div style="font-family:Syne;font-weight:800;color:#5b8dee;margin:.3rem 0">{title}</div>
          <div style="font-size:.78rem;color:#8b95b0;line-height:1.6">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.info("⬅️ Upload file Excel di sidebar untuk memulai.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPROCESSING — identik dengan file PJJ asli
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dan_preprocess(fb, sheet, col):
    df_raw = pd.read_excel(BytesIO(fb), sheet_name=sheet)
    COL    = col
    noise  = r'^[\s\.\-a]+$'
    df = df_raw.dropna(subset=[COL]).copy()
    df = df[~df[COL].astype(str).str.strip().str.match(noise)]
    df = df.drop_duplicates(subset=[COL]).reset_index(drop=True)

    # Split masukan — identik PJJ asli
    rows_split = []
    for idx, row in df.iterrows():
        items = pisah_masukan(row[COL])
        for j, item in enumerate(items):
            rows_split.append({
                'id_asal': idx, 'no_item': j+1, 'total_item': len(items),
                'masukan_asal': row[COL], 'masukan_item': item,
                'Kanwil': row.get('Kanwil',''), 'Pulau': row.get('Pulau',''),
            })

    df = pd.DataFrame(rows_split)
    df['masukan_nlp'] = df['masukan_item'].apply(bersihkan_nlp)
    df['panjang']     = df['masukan_item'].str.len()
    df['jml_kata']    = df['masukan_nlp'].str.split().str.len()
    df = df[df['jml_kata'] >= 5].reset_index(drop=True)
    df['Pulau'] = df['Pulau'].astype(str).str.strip().replace({'Sualwesi': 'Sulawesi'})
    return df

with st.spinner("🔄 Memuat data..."):
    df = load_dan_preprocess(fb, sheet, col_m)

# Reset hasil analisis jika sheet/kolom berubah (hindari stale data)
_data_key = f"{sheet}|{col_m}"
if st.session_state.get("_data_key") != _data_key:
    st.session_state["_data_key"] = _data_key
    st.session_state["done"]    = False
    st.session_state["hasil"]   = {}
    st.session_state["history"] = {}

PULAU_VALID = [p for p in PULAU_ORDER if p in df["Pulau"].unique()]

# KPI
k1,k2,k3,k4 = st.columns(4)
k1.metric("📝 Total Item",       f"{len(df):,}", f"{df['id_asal'].nunique():,} baris asal")
k2.metric("🏢 Kanwil",           str(df['Kanwil'].nunique()), "kanwil")
k3.metric("🏝️ Pulau",            str(df['Pulau'].nunique()), "pulau")
k4.metric("📖 Rata-rata Kata",   f"{df['jml_kata'].mean():.1f}", "kata/item")

if not st.session_state.get("done"):
    st.info("✅ Data dimuat. Klik **▶ Jalankan Analisis** di sidebar.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# JALANKAN ANALISIS — identik dengan logika file PJJ asli
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    sw_aktif = st.session_state["stopwords"]
    teks_list = df['masukan_nlp'].tolist()

    # ── Step 1: Cari K optimal — identik PJJ asli ────────────────────────
    progress = st.progress(0, "🔍 Mencari K optimal...")
    tfidf_baseline = TfidfVectorizer(
        max_features=400, min_df=3,
        ngram_range=(1,2), stop_words=list(sw_aktif)
    )
    X_baseline = tfidf_baseline.fit_transform(teks_list)

    sil_k = {}
    for idx, k in enumerate(range(3, 20)):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_baseline)
        sc     = silhouette_score(X_baseline, labels, sample_size=min(800,len(teks_list)), random_state=42)
        sil_k[k] = round(sc, 4)
        progress.progress((idx+1)/17, f"🔍 Silhouette K={k}: {sc:.4f}")

    if auto_k:
        N_TOPICS = max(sil_k, key=sil_k.get)
    else:
        N_TOPICS = manual_k

    # ── Step 2: Grid Search — identik PJJ asli ───────────────────────────
    if run_grid:
        GRID = {
            'max_features': [100,200,300,400,500,700,1000,1500,2000],
            'min_df'      : [2,3,5],
            'ngram_range' : [(1,1),(1,2)],
        }
        semua_kombinasi = list(itertools.product(
            GRID['max_features'], GRID['min_df'], GRID['ngram_range']))
        progress2 = st.progress(0, f"⚙️ Grid Search ({len(semua_kombinasi)} kombinasi)...")
        hasil_grid = []
        for ci, (max_f, min_d, ngram) in enumerate(semua_kombinasi):
            t0  = time.time()
            vec = TfidfVectorizer(max_features=max_f, min_df=min_d,
                                   ngram_range=ngram, stop_words=list(sw_aktif))
            X_grid = vec.fit_transform(teks_list)
            km     = KMeans(n_clusters=N_TOPICS, random_state=42, n_init=10)
            labels = km.fit_predict(X_grid)
            sc     = silhouette_score(X_grid, labels, sample_size=min(800,len(teks_list)), random_state=42)
            durasi = time.time() - t0
            hasil_grid.append({
                'max_features': max_f, 'min_df': min_d,
                'ngram_range': str(ngram), 'silhouette': round(sc,5),
                'n_fitur_aktual': X_grid.shape[1], 'waktu_detik': round(durasi,2),
            })
            progress2.progress((ci+1)/len(semua_kombinasi),
                               f"⚙️ {ci+1}/{len(semua_kombinasi)}: max_f={max_f} min_df={min_d} ngram={ngram} → {sc:.4f}")

        df_grid       = pd.DataFrame(hasil_grid).sort_values('silhouette', ascending=False)
        baris_terbaik = df_grid.iloc[0]
        BEST_MAX_F    = int(baris_terbaik['max_features'])
        BEST_MIN_DF   = int(baris_terbaik['min_df'])
        BEST_NGRAM    = eval(baris_terbaik['ngram_range'])
        BEST_SCORE    = baris_terbaik['silhouette']
    else:
        BEST_MAX_F  = manual_max_f
        BEST_MIN_DF = manual_min_d
        BEST_NGRAM  = eval(manual_ngram)
        BEST_SCORE  = 0.0
        df_grid     = pd.DataFrame()

    # ── Step 3: TF-IDF Final — identik PJJ asli ──────────────────────────
    with st.spinner(f"🤖 TF-IDF final & {teknik_utama}..."):
        # Vektorisasi — sesuai kebutuhan teknik
        if teknik_utama == "LDA":
            vec_final = CountVectorizer(
                max_features=BEST_MAX_F, min_df=BEST_MIN_DF,
                ngram_range=BEST_NGRAM, stop_words=list(sw_aktif))
        elif teknik_utama == "CorEx":
            vec_final = CountVectorizer(
                max_features=BEST_MAX_F, min_df=BEST_MIN_DF,
                ngram_range=(1,1), stop_words=list(sw_aktif), binary=True)
        else:
            vec_final = TfidfVectorizer(
                max_features=BEST_MAX_F, min_df=BEST_MIN_DF,
                ngram_range=BEST_NGRAM, stop_words=list(sw_aktif))

        X_final  = vec_final.fit_transform(teks_list)
        fn_final = vec_final.get_feature_names_out()

        # ── Step 4: Modeling — teknik yang dipilih ────────────────────────
        if teknik_utama == "NMF":
            # ── IDENTIK PJJ ASLI — tidak ada perubahan ────────────────────
            nmf = NMF(n_components=N_TOPICS, random_state=42, max_iter=500)
            W   = nmf.fit_transform(X_final)
            df['topik_id'] = W.argmax(axis=1)
            df['kekuatan'] = W.max(axis=1)
            topics_raw     = top_kata(nmf.components_, fn_final)
            freq_per_topik = {i: (df['topik_id']==i).sum() for i in range(N_TOPICS)}
            urutan_baru    = sorted(range(N_TOPICS), key=lambda x: -freq_per_topik[x])
            old_to_new     = {old:new for new,old in enumerate(urutan_baru)}
            new_to_old     = {new:old for new,old in enumerate(urutan_baru)}
            df['topik_id'] = df['topik_id'].map(old_to_new)
            topics         = {i: topics_raw[new_to_old[i]] for i in range(N_TOPICS)}
            model_obj      = nmf
            BEST_SCORE_TEKNIK = BEST_SCORE
        else:
            # ── Teknik lain via run_teknik_pembanding ─────────────────────
            anchor_t_utama = tuple(sorted(anchor_utama.items()))
            sw_t    = tuple(sorted(sw_aktif))
            teks_t  = tuple(teks_list)
            tid, kuat, topics, sil_t = run_teknik_pembanding(
                teks_t, teknik_utama, N_TOPICS,
                BEST_MAX_F, BEST_MIN_DF, BEST_NGRAM,
                sw_t, anchor_t_utama,
            )
            df['topik_id'] = tid
            df['kekuatan'] = kuat
            model_obj      = None
            BEST_SCORE_TEKNIK = sil_t
            # Gunakan X_final & fn_final untuk export kata (NMF tidak tersedia)
            nmf = None

        # ── Step 5: Penamaan Topik — identik PJJ asli ────────────────────
        # Default: top 4 kata (bisa diedit di tab Edit)
        NAMA_TOPIK = {i: ", ".join(topics[i][:4]) for i in range(N_TOPICS)}
        df['nama_topik'] = df['topik_id'].map(NAMA_TOPIK)

        # ── Step 6: Analisis Regional — identik PJJ asli ─────────────────
        TOPIK_SHORT = [f"T{i+1}" for i in range(N_TOPICS)]
        cross_n = pd.crosstab(df['Pulau'], df['topik_id'])
        cross_n.columns = [f"T{c+1}" for c in cross_n.columns]
        cross_n = cross_n.reindex(PULAU_ORDER, fill_value=0)
        cross_n['TOTAL'] = cross_n.sum(axis=1)
        cross_n.loc['TOTAL_NASIONAL'] = cross_n.sum()

        cross_pct = pd.crosstab(df['Pulau'], df['topik_id'], normalize='index') * 100
        cross_pct.columns = [f"T{c+1}" for c in cross_pct.columns]
        cross_pct = cross_pct.reindex(PULAU_ORDER, fill_value=0).round(1)

        nasional_pct = pd.crosstab(df['topik_id'], 'all', normalize=True) * 100
        nasional_pct = nasional_pct.T
        nasional_pct.columns = [f"T{c+1}" for c in nasional_pct.columns]
        nasional_pct.index   = ['NASIONAL']
        cross_pct_full = pd.concat([cross_pct, nasional_pct]).round(1)

        nasional_arr = nasional_pct.values[0]
        deviasi = cross_pct.copy()
        for col_i, col in enumerate(cross_pct.columns):
            deviasi[col] = (cross_pct[col] - nasional_arr[col_i]).round(1)

        profil_rows = []
        for pulau in PULAU_VALID:
            if pulau not in deviasi.index: continue
            row   = deviasi.loc[pulau]
            n_wil = int(cross_n.loc[pulau,'TOTAL']) if pulau in cross_n.index else 0
            above = sorted([(t,float(row[t])) for t in deviasi.columns if float(row[t])>=5.0], key=lambda x:-x[1])
            below = sorted([(t,float(row[t])) for t in deviasi.columns if float(row[t])<=-5.0], key=lambda x:x[1])
            profil_rows.append({
                'Wilayah': pulau, 'n': n_wil,
                'Topik Menonjol (≥+5pp)': ', '.join([f"{t}(+{v:.1f}pp)" for t,v in above[:3]]) or '—',
                'Topik Minim (≤-5pp)':    ', '.join([f"{t}({v:.1f}pp)" for t,v in below[:2]]) or '—',
            })
        df_profil_reg = pd.DataFrame(profil_rows)

        kuat_rows = []
        for pulau in PULAU_VALID:
            for t in range(N_TOPICS):
                sub = df[(df['Pulau']==pulau)&(df['topik_id']==t)]['kekuatan']
                kuat_rows.append({
                    'Pulau'        : pulau,
                    'Topik'        : f"T{t+1}",
                    'Nama'         : NAMA_TOPIK[t],
                    'n'            : len(sub),
                    'Kekuatan_mean': round(float(sub.mean()), 3) if len(sub) > 0 else 0.0,
                    'Kekuatan_std' : round(float(sub.std()),  3) if len(sub) > 1 else 0.0,
                })
        df_kuat_reg = pd.DataFrame(kuat_rows)
        # Guard: pastikan kolom ada dan tidak semua NaN sebelum pivot
        if 'Kekuatan_mean' in df_kuat_reg.columns and not df_kuat_reg['Kekuatan_mean'].isna().all():
            pivot_kuat_reg = df_kuat_reg.pivot_table(
                index='Pulau', columns='Topik',
                values='Kekuatan_mean', aggfunc='mean'
            ).reindex(PULAU_VALID).fillna(0.0).round(3)
        else:
            # Fallback: buat pivot kosong dengan kolom topik
            pivot_kuat_reg = pd.DataFrame(
                0.0,
                index=PULAU_VALID,
                columns=[f"T{t+1}" for t in range(N_TOPICS)]
            )

        df_kuat = ringkasan_kekuatan(df, N_TOPICS, NAMA_TOPIK)

    # Simpan ke session_state
    new_run_id = st.session_state.get("run_id", 0) + 1
    for i in range(50): st.session_state.pop(f"edit_nama_{new_run_id-1}_{i}", None)

    st.session_state.update({
        "done": True, "df": df.copy(), "topics": topics,
        "NAMA_TOPIK": NAMA_TOPIK, "N_TOPICS": N_TOPICS,
        "run_id": new_run_id,
        "teknik_utama": teknik_utama,
        "history": {teknik_utama: {  # reset history, simpan teknik utama sebagai baseline
            "sil": BEST_SCORE_TEKNIK, "K": N_TOPICS, "topics": topics.copy(),
            "dist": {i: int((df["topik_id"]==i).sum()) for i in range(N_TOPICS)},
            "topik_id": df["topik_id"].values.copy(),
            "kekuatan": df["kekuatan"].values.copy(),
        }},
        "hasil": {
            "sil_k": sil_k, "df_grid": df_grid,
            "BEST_MAX_F": BEST_MAX_F, "BEST_MIN_DF": BEST_MIN_DF,
            "BEST_NGRAM": BEST_NGRAM, "BEST_SCORE": BEST_SCORE,
            "BEST_SCORE_TEKNIK": BEST_SCORE_TEKNIK,
            "X_final": X_final, "fn_final": fn_final, "nmf": nmf,
            "cross_n": cross_n, "cross_pct": cross_pct,
            "cross_pct_full": cross_pct_full, "deviasi": deviasi,
            "df_profil_reg": df_profil_reg, "df_kuat_reg": df_kuat_reg,
            "pivot_kuat_reg": pivot_kuat_reg, "df_kuat": df_kuat,
            "TOPIK_SHORT": TOPIK_SHORT,
        },
    })
    st.success(f"✅ **{teknik_utama}** selesai! K={N_TOPICS} | Config terbaik: max_f={BEST_MAX_F} min_df={BEST_MIN_DF} ngram={BEST_NGRAM} | Silhouette={BEST_SCORE_TEKNIK:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# JALANKAN TEKNIK PEMBANDING
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("done") and uploaded and run_teknik_btn:
    df_base   = st.session_state["df"]
    N_TOPICS  = st.session_state["N_TOPICS"]
    H         = st.session_state["hasil"]
    BEST_MAX_F  = H["BEST_MAX_F"]
    BEST_MIN_DF = H["BEST_MIN_DF"]
    BEST_NGRAM  = H["BEST_NGRAM"]
    sw_aktif  = st.session_state["stopwords"]
    teks_t    = tuple(df_base["masukan_nlp"].tolist())
    sw_t      = tuple(sorted(sw_aktif))
    anchor_t  = tuple(sorted(st.session_state["anchor_words"].items()))

    new_history = dict(st.session_state["history"])

    for teknik in teknik_pilihan:
        if not cek_tersedia(teknik):
            st.warning(f"⚠️ {teknik} tidak tersedia. Install: pip install {' '.join(TEKNIK_KATALOG[teknik]['req'])}")
            continue

        with st.spinner(f"🔄 Menjalankan {teknik}..."):
            try:
                tid, kuat, topics_t, sil_t = run_teknik_pembanding(
                    teks_t, teknik, N_TOPICS,
                    BEST_MAX_F, BEST_MIN_DF, BEST_NGRAM,
                    sw_t, anchor_t,
                )
                new_history[teknik] = {
                    "sil": sil_t, "K": N_TOPICS,
                    "topics": topics_t.copy(),
                    "dist": {i: int((tid==i).sum()) for i in range(N_TOPICS)},
                    "topik_id": tid.copy(),
                    "kekuatan": kuat.copy(),
                }
                st.success(f"✅ {teknik} selesai — Silhouette: {sil_t:.4f}")
            except Exception as e:
                st.error(f"❌ {teknik} error: {e}")

    st.session_state["history"] = new_history

# ─────────────────────────────────────────────────────────────────────────────
# AMBIL STATE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.get("done"):
    st.stop()

df         = st.session_state["df"]
topics     = st.session_state["topics"]
NAMA_TOPIK = st.session_state["NAMA_TOPIK"]
N_TOPICS   = st.session_state["N_TOPICS"]
run_id     = st.session_state["run_id"]
teknik_aktif = st.session_state.get("teknik_utama", "NMF")
H          = st.session_state["hasil"]

sil_k        = H["sil_k"]
df_grid      = H["df_grid"]
BEST_MAX_F   = H["BEST_MAX_F"]
BEST_MIN_DF  = H["BEST_MIN_DF"]
BEST_NGRAM   = H["BEST_NGRAM"]
BEST_SCORE   = H["BEST_SCORE"]
BEST_SCORE_TEKNIK = H.get("BEST_SCORE_TEKNIK", BEST_SCORE)
X_final      = H["X_final"]
fn_final     = H["fn_final"]
nmf          = H["nmf"]
cross_n      = H["cross_n"]
cross_pct    = H["cross_pct"]
cross_pct_full = H["cross_pct_full"]
deviasi      = H["deviasi"]
df_profil_reg  = H["df_profil_reg"]
df_kuat_reg    = H["df_kuat_reg"]
pivot_kuat_reg = H["pivot_kuat_reg"]
df_kuat        = H["df_kuat"]
TOPIK_SHORT    = H["TOPIK_SHORT"]

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_peta, tab_topik, tab_gridsearch, tab_regional, tab_bandingkan, tab_edit, tab_data, tab_export = st.tabs([
    "🗺️  Peta", f"🤖  Topik ({teknik_aktif})", "🔍  Grid Search",
    "📊  Regional", "⚖️  Bandingkan", "✏️  Edit", "📋  Data", "💾  Export",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB — PETA
# ══════════════════════════════════════════════════════════════════════════════
with tab_peta:
    sh("🗺️","Distribusi Masukan per Wilayah — Peta Indonesia")

    pc = df[df["Pulau"].isin(PULAU_ORDER)].groupby("Pulau").size().reset_index(name="n")
    pc["pct"] = (pc["n"]/pc["n"].sum()*100).round(1)
    pd_dict   = {r["Pulau"]:{"n":int(r["n"]),"pct":float(r["pct"])} for _,r in pc.iterrows()}

    ca, cb = st.columns([2,2])
    with ca:
        skema_opts = {
            "🔵 Biru":("#061224","#1e4080","#5b8dee"),
            "🟢 Hijau":("#071a14","#0d6e56","#3ecfac"),
            "🟠 Oranye":("#1a0d00","#8a4800","#f5a623"),
            "🔴 Merah":("#1a0505","#8a1010","#f06b6b"),
            "🟣 Ungu":("#100820","#5a1e8a","#a78bfa"),
        }
        skema = st.selectbox("Skema warna", list(skema_opts), key="peta_skema")
        c0,c1,c2 = skema_opts[skema]

    st.markdown('<div style="border-radius:12px;overflow:hidden;border:1px solid #1e2438">',
                unsafe_allow_html=True)
    st_folium(buat_peta_choropleth(pd_dict,c0,c1,c2), width="100%", height=480,
              returned_objects=[], key="peta_utama")
    st.markdown("</div>", unsafe_allow_html=True)

    ca, cb = st.columns([3,2])
    with ca:
        pcs = pc.sort_values("n")
        fig_bar = go.Figure(go.Bar(
            x=pcs["n"], y=pcs["Pulau"], orientation="h",
            marker=dict(color=[WARNA_PULAU.get(p,"#5b8dee") for p in pcs["Pulau"]]),
            text=pcs.apply(lambda r:f"  {r['n']:,} ({r['pct']}%)",axis=1),
            textposition="outside", textfont=dict(color="#8b95b0",size=10),
        ))
        fig_bar.update_layout(**plotly_dark(), title="Jumlah Masukan per Pulau", height=280)
        st.plotly_chart(fig_bar, use_container_width=True)
    with cb:
        fig_tm = px.treemap(pc, path=["Pulau"], values="n",
            color="pct", color_continuous_scale=["#1c2438",c2],
            title="Treemap Proporsi")
        fig_tm.update_traces(texttemplate="<b>%{label}</b><br>%{value:,}",
                             textfont=dict(size=11,color="white"))
        fig_tm.update_layout(**plotly_dark(), height=280)
        st.plotly_chart(fig_tm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB — TOPIK NMF
# ══════════════════════════════════════════════════════════════════════════════
with tab_topik:
    sh(f"🤖","Hasil {teknik_aktif} Topic Modeling")
    info_ak = TEKNIK_KATALOG.get(teknik_aktif, TEKNIK_KATALOG["NMF"])
    st.markdown(f"""
    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:.8rem">
      <div style="background:{info_ak['color']}22;border:1px solid {info_ak['color']}55;
           border-radius:7px;padding:.3rem .9rem;font-family:Syne;font-size:.88rem;
           color:{info_ak['color']};font-weight:800">{teknik_aktif} · K={N_TOPICS}</div>
      <div style="color:#8b95b0;font-size:.82rem;align-self:center">
          TF-IDF: max_features={BEST_MAX_F} · min_df={BEST_MIN_DF} · ngram={BEST_NGRAM}
          &nbsp;|&nbsp; Silhouette: <b style="color:#3ecfac">{BEST_SCORE_TEKNIK:.5f}</b>
      </div>
    </div>""", unsafe_allow_html=True)

    # Distribusi topik
    sh("📊","Distribusi & Kekuatan")
    ca, cb = st.columns([3,2])
    with ca:
        rows = [{"lbl":f"T{i+1}: {NAMA_TOPIK[i][:32]}",
                 "n":int((df["topik_id"]==i).sum()),
                 "pct":round((df["topik_id"]==i).sum()/len(df)*100,1)}
                for i in range(N_TOPICS)]
        dfs = pd.DataFrame(rows).sort_values("n")
        fig_d = go.Figure()
        for _,r in dfs.iterrows():
            i_idx = int(r["lbl"][1])-1 if r["lbl"][1].isdigit() else 0
            fig_d.add_trace(go.Bar(
                x=[r["n"]], y=[r["lbl"]], orientation="h", showlegend=False,
                marker_color=COLORS[i_idx%len(COLORS)],
                text=[f"  {r['n']} ({r['pct']}%)"], textposition="outside",
                textfont=dict(color="#8b95b0",size=10)))
        fig_d.update_layout(**plotly_dark(), title="Item per Topik",
                             height=max(260,N_TOPICS*36), barmode="overlay")
        st.plotly_chart(fig_d, use_container_width=True)
    with cb:
        fig_k = go.Figure()
        for i in range(N_TOPICS):
            fig_k.add_trace(go.Box(
                y=df[df["topik_id"]==i]["kekuatan"], name=f"T{i+1}",
                marker_color=COLORS[i%len(COLORS)], boxmean="sd", showlegend=False))
        fig_k.update_layout(**plotly_dark(), title="Kekuatan per Topik",
                             height=max(260,N_TOPICS*36))
        st.plotly_chart(fig_k, use_container_width=True)

    # Top kata
    sh("🔑","Top Kata per Topik")
    cols3 = st.columns(min(3,N_TOPICS))
    for i in range(N_TOPICS):
        with cols3[i%3]:
            c = COLORS[i%len(COLORS)]
            pills = "".join(
                f'<span style="display:inline-block;padding:.15rem .6rem;border-radius:999px;'
                f'font-size:.72rem;font-weight:600;color:{c};background:{c}22;'
                f'border:1px solid {c}44;margin:.1rem">{w}</span>'
                for w in topics[i][:8])
            n_sub = (df["topik_id"]==i).sum()
            st.markdown(f"""
            <div style="background:#12151e;border:1px solid {c}33;border-left:3px solid {c};
                 border-radius:0 8px 8px 0;padding:.7rem .9rem;margin:.35rem 0">
              <div style="font-family:Syne;font-size:.82rem;font-weight:800;color:{c};margin-bottom:.35rem">
                T{i+1} · {NAMA_TOPIK[i][:38]}
                <span style="color:#434b66;font-size:.68rem;font-weight:400"> · {n_sub} item</span>
              </div>{pills}
            </div>""", unsafe_allow_html=True)

    # Word Cloud
    sh("☁️","Word Cloud per Topik")
    nc = st.select_slider("Kolom", [2,3,4], value=3, key="wc_col")
    cwc = st.columns(nc)
    cmaps = ["Blues","Oranges","Greens","Purples","Reds","YlOrBr","BuGn","RdPu","YlGnBu","PuRd","GnBu","OrRd"]
    for i in range(N_TOPICS):
        with cwc[i%nc]:
            teks = " ".join(df[df["topik_id"]==i]["masukan_item"].tolist())
            if teks.strip():
                wc = WordCloud(width=600, height=280, background_color="#12151e",
                               colormap=cmaps[i%len(cmaps)], max_words=50,
                               collocations=False,
                               stopwords=st.session_state["stopwords"]).generate(teks)
                fig_wc, ax = plt.subplots(figsize=(5,2.5))
                fig_wc.patch.set_facecolor("#12151e")
                ax.imshow(wc.to_array(), interpolation="bilinear"); ax.axis("off")
                buf = buf_img(fig_wc)
                st.image(buf, caption=f"T{i+1}: {NAMA_TOPIK[i][:26]} ({(df['topik_id']==i).sum()})",
                         use_container_width=True)
                plt.close(fig_wc)

    # PCA 2D — gunakan TruncatedSVD agar tidak perlu toarray() (hemat memori)
    sh("🔵","Sebaran Topik — PCA 2D")
    with st.spinner("Menghitung PCA..."):
        try:
            # TruncatedSVD bekerja langsung pada sparse matrix — tidak butuh .toarray()
            svd2 = TruncatedSVD(n_components=2, random_state=42)
            X_2d = svd2.fit_transform(X_final)
            var_ratio = svd2.explained_variance_ratio_
            fig_pca = go.Figure()
            for i in range(N_TOPICS):
                mask = df['topik_id'] == i
                fig_pca.add_trace(go.Scatter(
                    x=X_2d[mask,0], y=X_2d[mask,1], mode="markers",
                    name=f"T{i+1}: {NAMA_TOPIK[i][:20]}",
                    marker=dict(color=COLORS[i%len(COLORS)], size=5, opacity=0.4),
                ))
            fig_pca.update_layout(
                **plotly_dark(),
                title=f"SVD 2D — PC1 ({var_ratio[0]*100:.1f}%) × PC2 ({var_ratio[1]*100:.1f}%)",
                height=420,
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        except Exception as e:
            st.warning(f"Visualisasi PCA 2D tidak dapat ditampilkan: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_gridsearch:
    sh("🔍","Hasil Grid Search TF-IDF")

    # Silhouette per K
    sh("📈","Silhouette Score per K")
    ca, cb = st.columns([3,1])
    with ca:
        ks = list(sil_k.keys()); ss = list(sil_k.values())
        bk = max(sil_k, key=sil_k.get)
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=ks, y=ss, mode="lines+markers",
            line=dict(color="#5b8dee",width=2),
            marker=dict(size=8, color=["#f06b6b" if k==bk else "#5b8dee" for k in ks]),
            fill="tozeroy", fillcolor="rgba(91,141,238,.08)"))
        fig_sil.add_vline(x=bk, line_dash="dot", line_color="#f06b6b", opacity=.6)
        fig_sil.update_layout(**plotly_dark(), title=f"Silhouette per K (optimal K={bk})",
                               height=260, xaxis_title="K", yaxis_title="Silhouette")
        st.plotly_chart(fig_sil, use_container_width=True)
    with cb:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=BEST_SCORE,
            number={"valueformat":".5f","font":{"family":"Syne","size":20,"color":"#edf0f8"}},
            title={"text":"Best Silhouette","font":{"family":"Syne","size":11,"color":"#8b95b0"}},
            gauge={"axis":{"range":[0,.5]},"bar":{"color":"#5b8dee"},"bgcolor":"#1d2236",
                   "bordercolor":"#1d2236",
                   "steps":[{"range":[0,.2],"color":"#3d1a1a"},
                             {"range":[.2,.35],"color":"#3d2d0a"},
                             {"range":[.35,.5],"color":"#0d2d1c"}]}))
        fig_g.update_layout(**plotly_dark(), height=260)
        st.plotly_chart(fig_g, use_container_width=True)

    if not df_grid.empty:
        sh("📊","Hasil Grid Search — Top 10")
        top10 = df_grid.head(10).copy()
        top10["ngram_range"] = top10["ngram_range"].astype(str)
        st.dataframe(top10.style.format({
            "silhouette":"{:.5f}","waktu_detik":"{:.2f}s"
        }).background_gradient(subset=["silhouette"], cmap="Blues"),
        use_container_width=True, hide_index=True)

        sh("📈","Silhouette vs max_features")
        fig_gs = go.Figure()
        for ngram_val, color in [("(1, 2)","#5b8dee"),("(1, 1)","#f06b6b")]:
            sub_g   = df_grid[df_grid["ngram_range"]==ngram_val]
            grouped = sub_g.groupby("max_features")["silhouette"].mean()
            fig_gs.add_trace(go.Scatter(
                x=grouped.index, y=grouped.values, mode="lines+markers",
                name=f"ngram={ngram_val}", line=dict(color=color,width=2),
                marker=dict(size=7)))
        fig_gs.add_vline(x=BEST_MAX_F, line_dash="dot", line_color="#f5a623", opacity=.7)
        fig_gs.update_layout(**plotly_dark(), title="Rata-rata Silhouette per max_features",
                              height=300, xaxis_title="max_features", yaxis_title="Silhouette")
        st.plotly_chart(fig_gs, use_container_width=True)

        # Heatmap max_features x min_df
        sh("🗂️","Heatmap max_features × min_df")
        best_ngram_str = str(BEST_NGRAM)
        # ngram_range di df_grid disimpan sebagai "(1, 2)" atau "(1, 1)"
        heat_data = df_grid[df_grid["ngram_range"].astype(str) == best_ngram_str]
        if heat_data.empty:
            # Fallback: ambil ngram apapun yang ada
            heat_data = df_grid.groupby(["min_df","max_features"])["silhouette"].mean().reset_index()
            heat_pivot = heat_data.pivot_table(
                index="min_df", columns="max_features",
                values="silhouette", aggfunc="mean").fillna(0)
        else:
            heat_pivot = heat_data.pivot_table(
                index="min_df", columns="max_features",
                values="silhouette", aggfunc="mean").fillna(0)
        if not heat_pivot.empty:
            fig_hm = go.Figure(go.Heatmap(
                z=heat_pivot.values, x=[str(c) for c in heat_pivot.columns],
                y=[f"min_df={v}" for v in heat_pivot.index],
                colorscale=[[0,"#0d1520"],[.5,"#1e4080"],[1,"#5b8dee"]],
                text=heat_pivot.values.round(4), texttemplate="%{text}",
                textfont=dict(size=9,color="white"),
                colorbar=dict(title="Silhouette",tickfont=dict(color="#8b95b0"))))
            fig_hm.update_layout(**plotly_dark(),
                title=f"Heatmap Silhouette (ngram={best_ngram_str})", height=280)
            st.plotly_chart(fig_hm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB — REGIONAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_regional:
    sh("📊","Analisis Regional — Distribusi Topik NMF per Wilayah")

    ca, cb = st.columns(2)
    with ca:
        mat = cross_pct.reindex(PULAU_VALID).values
        fig_h1 = go.Figure(go.Heatmap(
            z=mat, x=TOPIK_SHORT, y=PULAU_VALID,
            colorscale=[[0,"#0d1520"],[.5,"#1e4080"],[1,"#5b8dee"]],
            text=mat.round(1), texttemplate="%{text}%",
            textfont=dict(size=8,color="white"),
            colorbar=dict(title="%",tickfont=dict(color="#8b95b0"))))
        fig_h1.update_layout(**plotly_dark(), title="Proporsi (%) per Topik per Pulau", height=320)
        st.plotly_chart(fig_h1, use_container_width=True)
    with cb:
        dm = deviasi.reindex(PULAU_VALID).values
        dm_valid = dm[~np.isnan(dm)] if dm.size > 0 else np.array([])
        if dm.size == 0 or dm_valid.size == 0:
            st.warning("⚠️ Tidak ada data simpangan untuk ditampilkan.")
        else:
            lim = max(abs(float(np.nanmin(dm))), abs(float(np.nanmax(dm))), 1)
            fig_h2 = go.Figure(go.Heatmap(
                z=dm, x=TOPIK_SHORT, y=PULAU_VALID,
                colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim,
                text=np.round(dm, 1), texttemplate="%{text}",
                textfont=dict(size=8, color="white"),
                colorbar=dict(title="pp", tickfont=dict(color="#8b95b0"))))
            fig_h2.update_layout(**plotly_dark(), title="Simpangan dari Nasional (pp)", height=320)
            st.plotly_chart(fig_h2, use_container_width=True)

    sh("📊","Stacked Bar 100% per Wilayah")
    wilayah_sorted = cross_n.drop("TOTAL_NASIONAL").sort_values("TOTAL",ascending=False).index.tolist()
    wilayah_sorted = [w for w in wilayah_sorted if w in cross_pct.index]
    fig_stk = go.Figure()
    for t in range(N_TOPICS):
        tcol = f"T{t+1}"
        vals = [float(cross_pct.loc[w,tcol]) if w in cross_pct.index else 0 for w in wilayah_sorted]
        fig_stk.add_trace(go.Bar(
            x=wilayah_sorted, y=vals, name=f"T{t+1}: {NAMA_TOPIK[t][:20]}",
            marker=dict(color=COLORS[t%len(COLORS)], line=dict(color="white",width=.5))))
    fig_stk.update_layout(**plotly_dark(), barmode="stack",
        title="Komposisi Topik per Wilayah", height=380, yaxis_title="%")
    st.plotly_chart(fig_stk, use_container_width=True)

    sh("📋","Profil Unik per Wilayah")
    st.dataframe(df_profil_reg, use_container_width=True, hide_index=True)

    sh("🔍","Drill-down per Pulau")
    sel_p = st.selectbox("Pilih Pulau:", ["— Pilih —"]+PULAU_VALID, key="reg_pulau")
    if sel_p != "— Pilih —":
        ca, cb = st.columns(2)
        with ca:
            pr = cross_pct.loc[sel_p] if sel_p in cross_pct.index else pd.Series()
            if not pr.empty:
                fig_pr = go.Figure(go.Bar(
                    x=pr.index, y=pr.values,
                    marker_color=[COLORS[int(t[1:])-1%len(COLORS)] for t in pr.index],
                    text=pr.values.round(1), textposition="outside",
                    textfont=dict(color="#8b95b0")))
                fig_pr.update_layout(**plotly_dark(), title=f"Proporsi Topik — {sel_p}", height=280)
                st.plotly_chart(fig_pr, use_container_width=True)
        with cb:
            dr = deviasi.loc[sel_p] if sel_p in deviasi.index else pd.Series()
            if not dr.empty:
                fig_dr = go.Figure(go.Bar(
                    x=dr.index, y=dr.values,
                    marker=dict(color=["#34d399" if v>0 else "#f06b6b" for v in dr.values]),
                    text=dr.values.round(1), textposition="outside",
                    textfont=dict(color="#8b95b0")))
                fig_dr.add_hline(y=0, line_dash="dash", line_color="#434b66")
                fig_dr.add_hline(y=5, line_dash="dot", line_color="#34d399", opacity=.3)
                fig_dr.add_hline(y=-5, line_dash="dot", line_color="#f06b6b", opacity=.3)
                fig_dr.update_layout(**plotly_dark(), title=f"Simpangan — {sel_p}", height=280)
                st.plotly_chart(fig_dr, use_container_width=True)

        df_p = df[df["Pulau"]==sel_p]
        for i in range(min(N_TOPICS,4)):
            sub_ex = df_p[df_p["topik_id"]==i].nlargest(3,"kekuatan")
            if sub_ex.empty: continue
            c = COLORS[i%len(COLORS)]
            with st.expander(f"T{i+1}: {NAMA_TOPIK[i]} — {len(df_p[df_p['topik_id']==i])} item"):
                for _,r in sub_ex.iterrows():
                    st.markdown(
                        f'<div style="border-left:3px solid {c};padding:.4rem .8rem;'
                        f'margin:.3rem 0;background:#12151e;border-radius:0 6px 6px 0;'
                        f'font-size:.8rem;color:#8b95b0">{r["masukan_item"][:260]}</div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB — BANDINGKAN TEKNIK
# ══════════════════════════════════════════════════════════════════════════════
with tab_bandingkan:
    sh("⚖️","Perbandingan Teknik Analisis")
    history = st.session_state.get("history", {})

    if len(history) < 1:
        st.info("Jalankan **▶ Jalankan Analisis** terlebih dahulu, lalu pilih teknik pembanding "
                "di sidebar dan klik **⚖️ Jalankan Pembanding**.")

    else:
        # ── Info katalog teknik ──────────────────────────────────────────
        with st.expander("📖 Panduan Memilih Teknik", expanded=False):
            cols_info = st.columns(3)
            for ci, (tname, info) in enumerate(TEKNIK_KATALOG.items()):
                with cols_info[ci % 3]:
                    avail = cek_tersedia(tname)
                    st.markdown(f"""
                    <div style="background:#12151e;border:1px solid {info['color']}33;
                         border-left:3px solid {info['color']};border-radius:6px;
                         padding:.6rem .8rem;margin:.3rem 0;font-size:.75rem">
                      <b style="color:{info['color']}">{tname}</b>
                      {'<span style="color:#34d399"> ✅</span>' if avail else '<span style="color:#f5a623"> ⚠️ perlu install</span>'}<br>
                      <span style="color:#8b95b0">{info['desc']}</span>
                    </div>""", unsafe_allow_html=True)

        # ── Silhouette bar ───────────────────────────────────────────────
        sh("📊","Silhouette Score per Teknik")
        cmp_rows = [{"Teknik": t, "Silhouette": round(v["sil"],4), "K": v["K"]}
                    for t, v in history.items()]
        df_cmp = pd.DataFrame(cmp_rows).sort_values("Silhouette", ascending=False)

        ca, cb = st.columns([2,3])
        with ca:
            fig_cs = go.Figure(go.Bar(
                x=df_cmp["Silhouette"], y=df_cmp["Teknik"], orientation="h",
                marker=dict(color=[TEKNIK_KATALOG.get(t,{}).get("color","#5b8dee")
                                   for t in df_cmp["Teknik"]]),
                text=df_cmp["Silhouette"].round(4), textposition="outside",
                textfont=dict(color="#8b95b0"),
            ))
            fig_cs.add_vline(x=0.25, line_dash="dot", line_color="#f5a623", opacity=.5)
            fig_cs.add_vline(x=0.35, line_dash="dot", line_color="#34d399", opacity=.5)
            fig_cs.update_layout(**plotly_dark(), title="Silhouette Score",
                                  height=max(280, len(history)*50))
            st.plotly_chart(fig_cs, use_container_width=True)
        with cb:
            st.dataframe(
                df_cmp.style
                    .bar(subset=["Silhouette"], color="#5b8dee33")
                    .format({"Silhouette":"{:.4f}"}),
                use_container_width=True, hide_index=True,
            )
            if len(df_cmp) > 0:
                best = df_cmp.iloc[0]["Teknik"]
                c    = TEKNIK_KATALOG.get(best,{}).get("color","#5b8dee")
                st.markdown(f"""
                <div style="background:{c}11;border:1px solid {c}44;border-radius:8px;
                     padding:.6rem 1rem;font-size:.8rem;color:#8b95b0;margin-top:.4rem">
                  🏆 <b style="color:{c}">{best}</b> menghasilkan silhouette tertinggi untuk dataset ini.<br>
                  <span style="font-size:.72rem">{TEKNIK_KATALOG.get(best,{}).get('desc','')}</span>
                </div>""", unsafe_allow_html=True)

        # ── Distribusi topik per teknik ──────────────────────────────────
        sh("📊","Distribusi Item per Topik per Teknik")
        for tname, hdata in history.items():
            color = TEKNIK_KATALOG.get(tname,{}).get("color","#5b8dee")
            dists = hdata["dist"]; total = sum(dists.values())
            st.markdown(
                f'<div style="font-family:Syne;font-size:.82rem;font-weight:700;'
                f'color:{color};margin:.6rem 0 .15rem">'
                f'{tname} — K={hdata["K"]} · Silhouette {hdata["sil"]:.4f}</div>',
                unsafe_allow_html=True)
            bar_html = ""
            for i, n in dists.items():
                pct = n/total*100 if total>0 else 0
                bc  = COLORS[i%len(COLORS)]
                bar_html += (
                    f'<div style="display:flex;align-items:center;gap:.5rem;margin:.1rem 0">'
                    f'<div style="font-size:.72rem;color:#8b95b0;width:30px">T{i+1}</div>'
                    f'<div style="flex:1;background:#1d2236;border-radius:4px;height:13px;overflow:hidden">'
                    f'<div style="width:{pct:.1f}%;height:100%;background:{bc};border-radius:4px"></div></div>'
                    f'<div style="font-size:.72rem;color:#edf0f8;width:80px;text-align:right;'
                    f'font-family:DM Mono,monospace">{n} ({pct:.1f}%)</div></div>'
                )
            st.markdown(bar_html, unsafe_allow_html=True)

        # ── Top kata lintas teknik ────────────────────────────────────────
        sh("🔑","Top Kata Topik Terbesar per Teknik")
        cols_tk = st.columns(min(len(history), 4))
        for ci, (tname, hdata) in enumerate(history.items()):
            with cols_tk[ci % 4]:
                c = TEKNIK_KATALOG.get(tname,{}).get("color","#5b8dee")
                top0 = hdata["topics"].get(0, [])[:8]
                pills = "".join(
                    f'<span style="display:inline-block;padding:.12rem .5rem;border-radius:999px;'
                    f'font-size:.7rem;color:{c};background:{c}22;border:1px solid {c}44;margin:.1rem">'
                    f'{w}</span>' for w in top0)
                st.markdown(f"""
                <div style="background:#12151e;border:1px solid {c}33;border-left:3px solid {c};
                     border-radius:0 8px 8px 0;padding:.6rem .8rem;margin:.3rem 0">
                  <div style="font-family:Syne;font-size:.8rem;font-weight:800;
                       color:{c};margin-bottom:.3rem">{tname} · Topik Terbesar</div>
                  {pills}
                </div>""", unsafe_allow_html=True)

        # ── Panduan teknik berdasarkan data ──────────────────────────────
        sh("💡","Panduan Memilih Teknik untuk Data Lain")
        panduan = [
            ("📄 Teks Pendek (feedback, survei, tiket)","NMF","Topik sparse & langsung terbaca"),
            ("📰 Dokumen Panjang (laporan, berita, makalah)","LDA","Distribusi topik probabilistik"),
            ("🔍 Eksplorasi Awal / Dataset Besar","LSA","Cepat, semantik laten"),
            ("🌐 Teks Informal / Singkatan / Multibahasa","BERTopic","Memahami konteks bukan frekuensi"),
            ("🏛️ Domain Spesifik (pajak, hukum, medis)","CorEx","Dipandu kata kunci domain"),
            ("🔀 Tidak Yakin Jumlah Topik","Top2Vec","Temukan K otomatis"),
            ("🎲 Ingin Probabilitas Keanggotaan","GMM atau LDA","Soft-assignment"),
            ("📍 Hard Clustering Cepat & Skalabel","KMeans","Mudah dijelaskan stakeholder"),
            ("🌲 Ingin Lihat Hierarki Topik","Agglomerative","Struktur hierarki topik"),
        ]
        for use, best, why in panduan:
            c = TEKNIK_KATALOG.get(best.split(" ")[0],{}).get("color","#5b8dee")
            st.markdown(f"""
            <div style="display:flex;gap:.8rem;padding:.4rem 0;border-bottom:1px solid #1d2236;
                 align-items:flex-start">
              <div style="font-size:.78rem;color:#8b95b0;flex:2">{use}</div>
              <div style="font-size:.76rem;font-weight:800;font-family:Syne;
                   color:{c};flex:1">→ {best}</div>
              <div style="font-size:.74rem;color:#434b66;flex:2">{why}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB — EDIT
# ══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    ca, cb = st.columns([1,1])

    with ca:
        sh("✏️","Edit Nama Topik")
        st.caption("Nama default = top 4 kata. Edit sesuai interpretasi domain Anda.")
        nama_baru = {}
        for i in range(N_TOPICS):
            sub   = df[df["topik_id"]==i]
            color = COLORS[i%len(COLORS)]
            top_k = ", ".join(topics[i][:5])
            st.markdown(f'<div style="font-size:.68rem;color:{color};font-weight:700;'
                        f'margin-top:.7rem">T{i+1} · {len(sub)} item · {top_k}</div>',
                        unsafe_allow_html=True)
            nama_baru[i] = st.text_input(
                f"t{i+1}", value=NAMA_TOPIK.get(i, top_k),
                key=f"edit_nama_{run_id}_{i}", label_visibility="collapsed")

        if st.button("💾 Simpan Nama Topik", type="primary", use_container_width=True):
            st.session_state["NAMA_TOPIK"] = nama_baru
            df["nama_topik"] = df["topik_id"].map(nama_baru)
            st.session_state["df"] = df.copy()
            NAMA_TOPIK = nama_baru
            st.success("✅ Nama topik diperbarui!"); st.rerun()

    with cb:
        sh("🚫","Stopwords")
        sw_set = st.session_state["stopwords"]
        st.markdown("**Tambah kata** (pisah koma)")
        sw_in = st.text_area("sw_in", height=80, key="sw_input_main",
                              label_visibility="collapsed",
                              placeholder="laporan, fitur, realisasi")
        ca2, cb2 = st.columns(2)
        with ca2:
            if st.button("➕ Tambah", use_container_width=True, key="sw_tb"):
                added = 0
                for w in sw_in.replace("\n",",").split(","):
                    w = w.strip().lower()
                    if w and w not in sw_set: sw_set.add(w); added+=1
                if added: st.toast(f"✅ {added} kata ditambahkan"); st.rerun()
        with cb2:
            if st.button("🔄 Reset", use_container_width=True, key="sw_rs"):
                st.session_state["stopwords"] = set(STOPWORDS_ID); st.rerun()

        st.divider()
        st.markdown(f"**Daftar aktif — {len(sw_set)} kata** · pilih untuk hapus")
        sw_cari = st.text_input("cari_sw", key="sw_cari_main",
                                 label_visibility="collapsed",
                                 placeholder="🔍 filter...")
        sw_list = sorted(sw_set)
        if sw_cari: sw_list = [w for w in sw_list if sw_cari.lower() in w]

        hapus_pilihan = st.multiselect("Pilih kata yang ingin dihapus:",
                                        options=sw_list, default=[], key="sw_multi",
                                        placeholder="Pilih kata...")
        if hapus_pilihan:
            if st.button(f"🗑️ Hapus {len(hapus_pilihan)} kata", type="primary",
                         use_container_width=True, key="sw_del_btn"):
                for w in hapus_pilihan: sw_set.discard(w)
                st.toast(f"🗑️ {len(hapus_pilihan)} kata dihapus"); st.rerun()

        if sw_list:
            chips = " ".join(f'<span class="sw">{w}</span>' for w in sw_list[:80])
            st.markdown(chips, unsafe_allow_html=True)

        st.divider()
        cd, cu = st.columns(2)
        with cd:
            st.download_button("📥 Download .txt", "\n".join(sorted(sw_set)),
                                "stopwords.txt","text/plain",use_container_width=True)
        with cu:
            sw_up = st.file_uploader("📂 Upload .txt", type=["txt"], key="sw_up_main",
                                      label_visibility="collapsed")
            if sw_up:
                kata = {w.strip().lower() for w in sw_up.read().decode().replace(","," ").split() if w.strip()}
                baru = kata - sw_set; sw_set.update(kata)
                st.toast(f"✅ {len(baru)} kata diimpor"); st.rerun()
        st.caption("Upload .txt — satu kata per baris")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    sh("📋","Data Berlabel")
    c1,c2,c3 = st.columns(3)
    with c1: fp = st.multiselect("Pulau", PULAU_ORDER, key="d_pulau")
    with c2: ft = st.multiselect("Topik",[f"T{i+1}: {NAMA_TOPIK[i]}" for i in range(N_TOPICS)],key="d_topik")
    with c3: fs = st.text_input("🔍 Cari kata", key="d_cari", placeholder="kata kunci...")

    dfv = df.copy()
    if fp: dfv = dfv[dfv["Pulau"].isin(fp)]
    if ft:
        ids = [int(t.split(":")[0].replace("T",""))-1 for t in ft]
        dfv = dfv[dfv["topik_id"].isin(ids)]
    if fs: dfv = dfv[dfv["masukan_item"].str.contains(fs,case=False,na=False)]

    st.markdown(f'<div style="color:#8b95b0;font-size:.78rem;margin-bottom:.4rem">Menampilkan <b style="color:#edf0f8">{len(dfv):,}</b> dari <b style="color:#edf0f8">{len(df):,}</b> item</div>',
                unsafe_allow_html=True)
    sc = [c for c in ["masukan_item","Kanwil","Pulau","nama_topik","kekuatan","jml_kata"] if c in dfv.columns]
    st.dataframe(dfv[sc].head(1000).style.format({"kekuatan":"{:.3f}"}),
                 use_container_width=True, height=460)

# ══════════════════════════════════════════════════════════════════════════════
# TAB — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    sh("💾","Export Hasil Analisis")

    oc = [c for c in ["id_asal","no_item","total_item","masukan_item",
                       "Kanwil","Pulau","topik_id","nama_topik","kekuatan","panjang","jml_kata"]
          if c in df.columns]

    ringkasan_topik = []
    kata_rows = []
    fn_list   = list(fn_final)
    for i in range(N_TOPICS):
        sub = df[df["topik_id"]==i]
        ringkasan_topik.append({
            "Topik":f"T{i+1}","Nama":NAMA_TOPIK[i],
            "Jumlah Item":len(sub),"Persen (%)":round(len(sub)/len(df)*100,1),
            "Baris Asal Unik":sub["id_asal"].nunique(),
            "Top 10 Kata":", ".join(topics[i][:10]),
            "Avg Panjang":round(sub["panjang"].mean(),0),
            "Avg Kekuatan":round(sub["kekuatan"].mean(),3),
            "Std Kekuatan":round(sub["kekuatan"].std(),3),
            "Min Kekuatan":round(sub["kekuatan"].min(),3),
            "Median Kekuatan":round(sub["kekuatan"].median(),3),
            "Max Kekuatan":round(sub["kekuatan"].max(),3),
        })
        for rank,kata in enumerate(topics[i],1):
            if nmf is not None and kata in fn_list:
                bobot = round(float(nmf.components_[i][fn_list.index(kata)]), 5)
            else:
                bobot = 0.0
            kata_rows.append({"Teknik":teknik_aktif,
                               "Topik":f"T{i+1}: {NAMA_TOPIK[i]}",
                               "Rank":rank,"Kata":kata,"Bobot":bobot})

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df[oc].to_excel(writer, sheet_name="Data_Berlabel", index=False)
        pd.DataFrame(ringkasan_topik).to_excel(writer, sheet_name="Ringkasan_Topik", index=False)
        pd.DataFrame(kata_rows).to_excel(writer, sheet_name="Kata_per_Topik", index=False)
        if not df_grid.empty:
            df_grid.to_excel(writer, sheet_name="Hasil_Grid_Search", index=False)
        df_kuat.to_excel(writer, sheet_name="Kekuatan_Topik", index=False)
        # Sheet regional — identik dengan file PJJ asli (FIX D)
        cross_n.to_excel(writer, sheet_name="Regional_Frekuensi")
        cross_pct_full.to_excel(writer, sheet_name="Regional_Proporsi_Pct")
        deviasi.to_excel(writer, sheet_name="Regional_Simpangan_pp")
        df_profil_reg.to_excel(writer, sheet_name="Regional_Profil_Wilayah", index=False)
        pivot_kuat_reg.to_excel(writer, sheet_name="Regional_Kekuatan")
        df_kuat_reg.to_excel(writer, sheet_name="Regional_Kekuatan_Detail", index=False)

    st.download_button(
        "📥 Download Excel (11 sheet — identik output PJJ asli)",
        buf.getvalue(),
        file_name="Hasil_PJJ_MasukanVertikal.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.caption("Sheet: Data_Berlabel · Ringkasan_Topik · Kata_per_Topik · Hasil_Grid_Search · "
               "Kekuatan_Topik · Regional_Frekuensi · Regional_Proporsi_Pct · "
               "Regional_Simpangan_pp · Regional_Profil_Wilayah · Regional_Kekuatan · Regional_Kekuatan_Detail")

    sh("📊","Preview Ringkasan Topik")
    st.dataframe(pd.DataFrame(ringkasan_topik), use_container_width=True, hide_index=True)