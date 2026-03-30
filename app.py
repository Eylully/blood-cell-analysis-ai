import streamlit as st
import numpy as np
import tensorflow as tf
import os
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    confusion_matrix, cohen_kappa_score, mean_absolute_error,
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.applications.resnet50 import preprocess_input
# ─────────────────────────────────────────────
# 1. TEMEL AYARLAR VE SABİTLER
# ─────────────────────────────────────────────
st.set_page_config(page_title="Kan Hücresi Analizi | Sağlık Bilişimi", layout="wide")

CLASS_NAMES_TR = [
    "Bazofil", "Eozinofil", "Eritroblast", "Lenfosit",
    "Monosit", "Nötrofil", "Olgunlaşmamış Granülosit", "Trombosit"
]
NUM_CLASSES = len(CLASS_NAMES_TR)
IMG_SIZE = (160, 160)



# ─────────────────────────────────────────────
# 2. YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def goruntu_hazirlik(image_file):
    """Kullanıcının yüklediği resmi modele (ResNet50) uygun hale getirir."""
    img = Image.open(image_file).convert('RGB')
    img_cropped = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Modele gidecek dizi
    img_array = np.array(img_cropped, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) # İŞTE DÜZELTİLEN YER: Modelin anadili!
    
    return img_cropped, img_array

def reverse_preprocess(img_array):
    """X_test.npy içindeki pikselleri ekranda gösterebilmek için normale çevirir."""
    img = img_array.copy()
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 123.68
    img = img[..., ::-1] # BGR'den RGB'ye
    img = np.clip(img, 0, 255).astype('uint8')
    return img

import gdown

@st.cache_resource
def load_model_cached():
    model_path = 'bloodcell_finetuned.keras'
    
    # Dosya yoksa Drive'dan indir
    if not os.path.exists(model_path):
        with st.spinner("Yapay zeka modeli buluttan indiriliyor, lütfen bekleyin..."):
            # BURADAKİ TIRNAK İÇİNE KENDİ FILE_ID'Nİ YAPIŞTIR
            file_id = '1GhVqCQcPSVqxxMN7bNATuxBLqgiz1ihp' 
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
    
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        st.stop()

@st.cache_data
def load_test_data():
    """X_test ve y_test demo dosyalarını yükler."""
    try:
        # İSİMLERİ BURADA _demo OLACAK ŞEKİLDE GÜNCELLİYORUZ
        X_test = np.load('X_test_demo.npy') 
        y_test = np.load('y_test_demo.npy')
        return X_test, y_test
    except Exception:
        st.error("X_test_demo.npy veya y_test_demo.npy bulunamadı! Simülasyon çalışmayabilir.")
        return None, None

# ─────────────────────────────────────────────
# 3. SESSION STATE BAŞLAT
# ─────────────────────────────────────────────
for key in ['y_true', 'y_pred', 'y_pred_probs']:
    if key not in st.session_state:
        st.session_state[key] = []

# ─────────────────────────────────────────────
# 4. MODEL VE VERİ YÜKLEME
# ─────────────────────────────────────────────
model = load_model_cached()
X_test, y_test = load_test_data()

# ─────────────────────────────────────────────
# 5. SIDEBAR (MENÜ)
# ─────────────────────────────────────────────
st.sidebar.title("🔬 Kan Hücresi Analizi")
st.sidebar.markdown("**Sağlık Bilişimi Uzman Asistan Sistemi**")
st.sidebar.divider()

secim = st.sidebar.radio(
    "Sayfa Seçin",
    [
        "Ana Sayfa",
        "Veri Seti ve Model Bilgileri",
        "Tahmin ve Analiz",
        "Güncel Performans",
        "Model Performansı",
        "Sonuç ve Kaynakça"
    ]
)

st.sidebar.divider()
if st.sidebar.button("🗑️ Canlı Verileri Sıfırla", type="secondary"):
    st.session_state['y_true'] = []
    st.session_state['y_pred'] = []
    st.session_state['y_pred_probs'] = []
    st.sidebar.success("✅ Veriler sıfırlandı.")
    st.rerun()

# ─────────────────────────────────────────────
# SAYFA 1: ANA SAYFA
# ─────────────────────────────────────────────
if secim == "Ana Sayfa":
    st.title("🔬 Yapay Zeka Destekli Kan Hücresi Analizi")
    st.markdown("""
    **Projenin Amacı:**
    Bu proje, mikroskop görüntülerindeki **8 farklı kan hücresi tipini** derin öğrenme (ResNet50) kullanarak
    otomatik ve yüksek doğrulukla sınıflandırmayı amaçlar.

    Sistem, laboratuvar süreçlerini hızlandırarak **Lösemi (Kan Kanseri), Anemi ve ağır enfeksiyonların**
    erken teşhisinde doktorlara destek olmayı hedeflemektedir.
    """)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🧫 Toplam Görüntü", "17.092")
    col_b.metric("🏷️ Sınıf Sayısı", "8")
    col_c.metric("🎯 Test Doğruluğu", "%98.01") # Zirve doğruluk oranın!

    st.divider()
    st.subheader("🧬 Kan Hücresi Rehberi")
    
    cell_info = [
        ("Nötrofil", "notrofil.jpg", "Bağışıklık sisteminin ilk savunma hattıdır."
        "Kanda en çok bulunan akyuvar türüdür, ani artış (Nötrofili), vücudun genellikle bakteriyel bir enfeksiyonla (zatürre, apandisit vb.) savaştığını gösterir."),
        ("Lenfosit", "lenfosit.jpg", "Antikor üreterek virüsleri yok eder."
        "Viral enfeksiyonlarda veya Lösemi (Kan kanseri) durumlarında sayıları kontrolsüzce artar."),
        ("Eozinofil", "eozinofil.jpg", "Büyük parazitlere karşı korur ve alerjik reaksiyonları kontrol eder, sayıları artar."),
        ("Bazofil", "bazofil.jpg", "Zehirli böcek sokmaları veya şiddetli alerjik şoklarda (Anafilaksi) artış gösteri, vücudu uyarırlar."),
        ("Monosit", "monosit.jpg", "Tüberküloz (verem) gibi kronik enfeksiyonlarda ve dokuların iyileşme evrelerinde kanda artış gösterir"),
        ("Trombosit", "trombosit.jpg", "Kan damarlarındaki hasarları onaran ve kanamayı durduran küçük kan pulcuklarıdır"
        "Kanın pıhtılaşmasını sağlar. Eksikliği kanamaya, fazlalığı pıhtı atmasına (felç riski) yol açabilir."),
        ("Eritroblast", "eritroblast.jpg", "Alyuvarların henüz çekirdeğini kaybetmemiş, olgunlaşmamış halidir. "
        "Kanda bulunmamalıdır."
        "Kanda görülmesi şiddetli anemi (kansızlık) veya kemik iliği hasarının acil bir sinyalidir."),
        ("IG (Olgunlaşmamış Granülosit)", "ig.jpg", "Nötrofillerin gelişimini tamamlamamış halidir. Ağır enfeksiyon (Sepsis) durumunda kemik iliğinden kana karışırlar."
        "Sağlıklı bir insanda kanda %1'den az bulunmalıdır.")
    ]

    for i in range(0, len(cell_info), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(cell_info):
                title, img_name, desc = cell_info[i + j]
                with cols[j]:
                    st.divider()
                    st.markdown(f"**{title}**")
                    img_path = os.path.join("ornek_veriler", img_name)
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"{title} Örneği", use_container_width=True)
                    else:
                        st.warning(f"'{img_name}' bulunamadı — ornek_veriler/ klasörünü kontrol edin")
                    st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)
# ═════════════════════════════════════════════
# SAYFA 2 — VERİ SETİ VE MODEL
# ═════════════════════════════════════════════
elif secim == "Veri Seti ve Model Bilgileri":
    st.title("Veri Seti ve Model")

    # ─────────────────────────────────────────
    # 1. VERİ SETİ
    # ─────────────────────────────────────────
    st.subheader("1. Veri Seti")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        | Özellik | Değer |
        |---|---|
        | **Kaynak** | Kaggle — Blood Cells Image Dataset |
        | **Toplam Görüntü** | 17.092 |
        | **Sınıf Sayısı** | 8 |
        | **Görüntü Boyutu** | 160 × 160 piksel |
        | **Link** | [Kaggle Linki](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset) |
        """)

    with col2:
        if os.path.exists("sinif_dagilimi.png"):
            st.image("sinif_dagilimi.png", caption="Sınıf Dağılımı", use_container_width=True)
        else:
            st.warning("sinif_dagilimi.png bulunamadı.")

    st.divider()

    # ─────────────────────────────────────────
    # 2. VERİ ÖN İŞLEME
    # ─────────────────────────────────────────
    st.subheader("2. Veri Ön İşleme ve Augmentasyon")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        **Veri Bölümü:**
        - Eğitim: **%80**
        - Doğrulama: **%10**
        - Test: **%10**

        **Görüntü İşleme:**
        - Görüntüler **center-crop + resize (160×160)** yapıldı
        - **ResNet50 için `preprocess_input` kullanılarak normalize edildi 
          (RGB → BGR dönüşümü + zero-centering)
        """)

    with col4:
        st.markdown("""
        **Veri Artırma (Augmentasyon):**
        - Yatay / dikey çevirme
        - Parlaklık
        - Kontrast
        - Doygunluk

        **Sınıf Dengeleme:**
        - `compute_class_weight('balanced')`
        - Dengesiz veri problemini azalttı
        """)

    st.divider()

    # ─────────────────────────────────────────
    # 3. MODEL
    # ─────────────────────────────────────────
    st.subheader("3. Model Mimarisi — ResNet50 (Transfer Learning)")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("""
        **Temel Model:**
        - ImageNet üzerinde eğitilmiş **ResNet50**

        **Ek Katmanlar:**
        - GlobalAveragePooling2D
        - BatchNormalization
        - Dropout(0.5)
        - Dense(8, softmax)
        """)

    with col6:
        st.markdown("""
        **Eğitim Stratejisi:**

        🔹 *Aşama 1 — Ön Isınma*
        - 5 epoch
        - LR = 0.0005

        🔹 *Aşama 2 — Fine-Tuning*
        - Son 50 katman açıldı
        - EarlyStopping (patience=5)

        **Hiperparametreler:**
        - Batch Size: 32
        - Dropout: 0.5
        - Min LR: 1e-8
        """)

    st.divider()

    # ─────────────────────────────────────────
    # 4. GRAFİKLER
    # ─────────────────────────────────────────
    st.subheader("📈 4. Eğitim Süreci")

    if os.path.exists("accuracy_loss.png"):
        st.image(
            "accuracy_loss.png",
            caption="Accuracy & Loss Eğrileri — Warmup + Fine-Tuning",
            use_container_width=True
        )
    else:
        st.warning("accuracy_loss.png bulunamadı.")
# ─────────────────────────────────────────────
# SAYFA 3: TAHMİN VE SİMÜLASYON (X_test'ten Alıyor!)
# ─────────────────────────────────────────────
elif secim == "Tahmin ve Analiz":
    st.title("Görüntü Analizi")
    st.markdown("Test verilerinden (`X_test.npy`) otomatik olarak hücre çekerek analizi yapın veya manuel resim yükleyin.")
    
    tab1, tab2 = st.tabs(["Bilgisayardan Yükle", "Otomatik Analiz"])
    
    with tab1:
        col_in1, col_res1 = st.columns([1, 1])
        with col_in1:
            uploaded_file = st.file_uploader("Bilgisayardan Görüntü Yükleyin", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                true_label_text = st.selectbox("Bu görüntünün GERÇEK sınıfını biliyorsanız seçin:", CLASS_NAMES_TR)
                
        with col_res1:
            if st.button("Hücreyi Analiz Et", type="primary") and uploaded_file:
                img_cropped, img_array = goruntu_hazirlik(uploaded_file)
                preds = model.predict(img_array)[0]
                pred_label_idx = int(np.argmax(preds))
                pred_label_text = CLASS_NAMES_TR[pred_label_idx]
                confidence = float(preds[pred_label_idx]) * 100
                
                st.subheader(f"Tahmin: {pred_label_text}")
                st.metric("Güven Skoru", f"%{confidence:.1f}")
                st.image(img_cropped, caption="Modele Giren Görüntü", width=160)
                
                true_label_idx = CLASS_NAMES_TR.index(true_label_text)
                st.session_state['y_true'].append(true_label_idx)
                st.session_state['y_pred'].append(pred_label_idx)
                st.session_state['y_pred_probs'].append(preds)
                st.success("✅ Canlı panoya eklendi.")

    with tab2:
        col_in2, col_res2 = st.columns([1, 2])
        with col_in2:
            st.markdown("**Test Setinden (`X_test`) Rastgele Hücre Çek:**")
            sim_images_num = st.slider("Analiz edilecek örnek sayısı:", 1, 50, 10)
            batch_sim_getir = st.button("▶️ Analizi Başlat", type="primary")
            
        with col_res2:
            if batch_sim_getir and X_test is not None:
                st.subheader(f"{sim_images_num} Hücre İçin Analiz Sonuçları")
                progress_bar = st.progress(0)
                results_data = []
                
                # X_test içerisinden rastgele indeksler seç
                random_indices = random.sample(range(len(X_test)), sim_images_num)
                
                for i, idx in enumerate(random_indices):
                    # Veriyi çek ve modelin istediği [1, 160, 160, 3] formatına getir
                    img_array = X_test[idx]
                    img_input = np.expand_dims(img_array, axis=0)
                    
                    # Gerçek etiketi bul
                    y_t = y_test[idx]
                    true_label_idx = int(np.argmax(y_t)) if y_t.ndim > 0 and len(y_t) > 1 else int(y_t)
                    true_label_text = CLASS_NAMES_TR[true_label_idx]
                    
                    # Tahmin yap
                    preds = model.predict(img_input, verbose=0)[0]
                    pred_label_idx = int(np.argmax(preds))
                    pred_label_text = CLASS_NAMES_TR[pred_label_idx]
                    confidence = float(preds[pred_label_idx]) * 100
                    
                    # Ekranda göstermek için numpy dizisini resme çevir
                    display_img = reverse_preprocess(img_array)
                    
                    # Kayıtları tut
                    st.session_state['y_true'].append(true_label_idx)
                    st.session_state['y_pred'].append(pred_label_idx)
                    st.session_state['y_pred_probs'].append(preds)
                    
                    dogru_mu = "✅" if true_label_text == pred_label_text else "❌"
                    results_data.append({
                        "Gerçek Sınıf": true_label_text,
                        "Tahmin": pred_label_text,
                        "Güven": f"%{confidence:.1f}",
                        "Durum": dogru_mu
                    })
                    
                    # Arayüzde küçük resimlerle göster
                    cols = st.columns([1, 4])
                    cols[0].image(display_img, width=80)
                    cols[1].markdown(f"**Gerçek:** {true_label_text} | **Tahmin:** {pred_label_text} {dogru_mu}")
                    
                    progress_bar.progress(int(((i + 1) / sim_images_num) * 100))
                    
                progress_bar.empty()
                st.success(f"✅ Analiz bitti. Tüm sonuçlar Canlı Performans Panosuna aktarıldı!")


# ═════════════════════════════════════════════
# SAYFA 6 — Güncel PERFORMANS
# ═════════════════════════════════════════════
# ═════════════════════════════════════════════
# SAYFA 4 — GÜNCEL PERFORMANS (CANLI PANO)
# ═════════════════════════════════════════════
elif secim == "Güncel Performans":
    st.title("📡 Model Performans Panosu")
    st.markdown(
        "Tekli veya toplu tahmin sayfalarından sisteme veri besledikçe "
        "aşağıdaki tüm metrikler ve grafikler **anlık güncellenir**."
    )

    n = len(st.session_state['y_true'])

    if n == 0:
        st.warning(
            "⚠️ Henüz veri yok. Lütfen **Tahmin ve Analiz** "
            "sayfasından önce örnek analiz ettirin."
        )
    else:
        y_true_arr  = np.array(st.session_state['y_true'])
        y_pred_arr  = np.array(st.session_state['y_pred'])
        y_probs_arr = np.array(st.session_state['y_pred_probs'])

        # ── Metrikler
        acc   = accuracy_score(y_true_arr, y_pred_arr)
        prec  = precision_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
        rec   = recall_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
        f1    = f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
        kappa = cohen_kappa_score(y_true_arr, y_pred_arr)
        mae   = mean_absolute_error(y_true_arr, y_pred_arr)

        st.subheader(f"Anlık Performans — Toplam {n} Analiz")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy",      f"{acc:.3f}")
        m2.metric("Precision",     f"{prec:.3f}")
        m3.metric("Recall",        f"{rec:.3f}")
        m4.metric("F1-Score",      f"{f1:.3f}")
        m5.metric("Cohen's Kappa", f"{kappa:.3f}")
        m6.metric("MAE",           f"{mae:.3f}")

        st.divider()

        # ── Confusion Matrix + ROC yan yana
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("**Confusion Matrix (Anlık)**")
            cm = confusion_matrix(y_true_arr, y_pred_arr, labels=np.arange(NUM_CLASSES))
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5), dpi=130)
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_TR, yticklabels=CLASS_NAMES_TR,
                ax=ax_cm, cbar=True, linewidths=0.5
            )
            ax_cm.set_title('Karmaşıklık Matrisi (Anlık)', fontsize=11)
            ax_cm.set_ylabel('Gerçek Sınıf', fontsize=9)
            ax_cm.set_xlabel('Tahmin Edilen Sınıf', fontsize=9)
            plt.xticks(rotation=35, ha='right', fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig_cm)

        with g2:
            st.markdown("**ROC-AUC Eğrileri (Anlık)**")
            unique_classes = np.unique(y_true_arr)
            if len(unique_classes) < 2:
                st.info("ROC eğrisi için en az 2 farklı sınıftan veri gereklidir.")
            else:
                y_true_bin = label_binarize(y_true_arr, classes=np.arange(NUM_CLASSES))
                if y_true_bin.ndim == 1 or y_true_bin.shape[1] < 2:
                    st.info("ROC için daha fazla farklı sınıf verisi gereklidir.")
                else:
                    fig_roc, ax_roc = plt.subplots(figsize=(6, 5), dpi=130)
                    colors_roc = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
                    plotted = False
                    for i in range(NUM_CLASSES):
                        if np.sum(y_true_arr == i) > 0 and np.sum(y_true_bin[:, i]) > 0:
                            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_arr[:, i])
                            roc_auc = auc(fpr, tpr)
                            ax_roc.plot(
                                fpr, tpr, color=colors_roc[i], lw=2,
                                label=f'{CLASS_NAMES_TR[i]} (AUC={roc_auc:.2f})'
                            )
                            plotted = True
                    if plotted:
                        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
                        ax_roc.set_xlabel('False Positive Rate', fontsize=9)
                        ax_roc.set_ylabel('True Positive Rate', fontsize=9)
                        ax_roc.set_title('ROC-AUC Eğrileri (Anlık)', fontsize=11)
                        ax_roc.legend(loc='lower right', fontsize=7)
                        ax_roc.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_roc)

        st.divider()

        # ── Sınıf bazında F1 (anlık, tek renk bar)
        st.markdown("**Sınıf Bazında F1-Score (Anlık)**")
        f1_per_class = f1_score(
            y_true_arr, y_pred_arr, average=None,
            labels=np.arange(NUM_CLASSES), zero_division=0
        )
        fig_f1l, ax_f1l = plt.subplots(figsize=(10, 3.5), dpi=120)
        
        bars_l = ax_f1l.bar(CLASS_NAMES_TR, f1_per_class, color="#2196F3", edgecolor='white')
        
        for bar, score in zip(bars_l, f1_per_class):
            ax_f1l.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        ax_f1l.set_ylim(0, 1.15)
        ax_f1l.set_ylabel("F1-Score")
        ax_f1l.set_title("Sınıf Bazında F1-Score (Anlık)")
        plt.xticks(rotation=30, ha='right', fontsize=8)
        ax_f1l.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_f1l)

        st.divider()

        # ── Sınıf bazlı tablo
        st.markdown("**Sınıf Bazlı Detaylı Rapor (Anlık)**")
        report_dict = classification_report(
            y_true_arr, y_pred_arr,
            labels=np.arange(NUM_CLASSES),
            target_names=CLASS_NAMES_TR,
            output_dict=True, zero_division=0
        )
        live_df = pd.DataFrame(report_dict).T.loc[
            CLASS_NAMES_TR, ['precision', 'recall', 'f1-score', 'support']
        ]
        live_df['support'] = live_df['support'].astype(int)
        
        st.dataframe(
            live_df.style.format("{:.3f}", subset=['precision', 'recall', 'f1-score']),
            use_container_width=True
        )


# ═════════════════════════════════════════════
# SAYFA: MODEL PERFORMANSI (STATİK)
# ═════════════════════════════════════════════
elif secim == "Model Performansı":
    st.title("📈 Model Performans Analizi (Test Seti)")

    # 1. Modelin Eğitimden Çıkan Kesin (Sabit) Sonuçları
    accuracy = 0.9801
    precision = 0.9807
    recall = 0.9789
    f1 = 0.9798 
    kappa = 0.9768
    mae = 0.0520

    st.subheader("Genel Performans Metrikleri")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"%{accuracy*100:.2f}")
    m2.metric("Precision", f"{precision:.4f}")
    m3.metric("Recall", f"{recall:.4f}")
    m4.metric("F1-Score", f"{f1:.4f}")
    m5.metric("Cohen's Kappa", f"{kappa:.4f}")
    m6.metric("MAE", f"{mae:.4f}")

    # Formüller
    with st.expander("📐 Metrik Formülleri"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown("**Accuracy**")
                st.latex(r"\frac{TP + TN}{TP + TN + FP + FN}")
                st.markdown("**Precision**")
                st.latex(r"\frac{TP}{TP + FP}")
            with fc2:
                st.markdown("**Recall (Sensitivity)**")
                st.latex(r"\frac{TP}{TP + FN}")
                st.markdown("**F1-Score**")
                st.latex(r"2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}")
            with fc3:
                st.markdown("**Cohen's Kappa**")
                st.latex(r"\kappa = \frac{p_o - p_e}{1 - p_e}")
                st.markdown("**MAE**")
                st.latex(r"\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|")

    st.divider()


    # 2. İndirdiğin (Sabit) Grafikleri Ekrana Basma
    st.subheader("Performans Grafikleri")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("**Confusion Matrix (Karmaşıklık Matrisi)**")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.warning("⚠️ confusion_matrix.png proje klasöründe bulunamadı.")

    with g2:
        st.markdown("**ROC-AUC Eğrileri**")
        if os.path.exists("roc_auc.png"):
            st.image("roc_auc.png", use_container_width=True)
        else:
            st.warning("⚠️ roc_auc.png proje klasöründe bulunamadı.")

    st.divider()

    # F1-Score Grafiğini Ekleme Kısmı
    st.subheader("Sınıf Bazında F1-Score Grafiği")
    # Dosya adını f1_per_class.png olarak ayarladım. Kendi indirdiğin resmin adını buna göre değiştirebilirsin.
    if os.path.exists("f1_per_class.png"):
        st.image("f1_per_class.png", use_container_width=True)
    else:
        st.warning("⚠️ f1_per_class.png proje klasöründe bulunamadı.")


    st.divider()

    # 3. Modelden Alınan Sınıf Bazlı Sabit Rapor (Sade ve Renksiz Tablo)
    st.subheader("Sınıf Bazlı Detaylı Rapor")
    
    report_data = {
        "Sınıf": CLASS_NAMES_TR,
        "Precision": [0.98, 0.99, 0.99, 0.99, 0.95, 0.97, 0.97, 1.00],
        "Recall":    [0.97, 0.99, 0.99, 0.98, 0.98, 0.98, 0.95, 1.00],
        "F1-Score":  [0.98, 0.99, 0.99, 0.99, 0.97, 0.98, 0.96, 1.00],
        "Support":   [122,  312,  155,  121,  142,  333,  290,  235]
    }
    
    report_df = pd.DataFrame(report_data).set_index("Sınıf")
    # Tablodaki renkli tasarım kaldırıldı, düz ve net bir formata çevrildi
    st.dataframe(
        report_df.style.format("{:.2f}", subset=['Precision', 'Recall', 'F1-Score']),
        use_container_width=True
    )

    st.divider()

    st.subheader("Sonuçların Yorumlanması")
    st.success(
        "✅ **%98.01 Test Doğruluğu** ve **0.9768 Cohen's Kappa** değerleriyle model 'mükemmel uyum' "
        "kategorisindedir. AUC = 0.9992, tüm sınıflarda neredeyse kusursuz ayırt edici güç anlamına gelir. "
        "En düşük F1 skoru 0.96 ile (Olgunlaşmamış Granülosit) klinik kullanım için "
        "yeterli güvenilirlik sunmaktadır."
    )
    st.info(
        "ℹ️ MAE = 0.052 değeri; yanlış tahminlerde bile sınıf indeksi farkının sıfıra çok yakın olduğunu, "
        "yani modelin tamamen uzak sınıflara değil, biyolojik olarak birbirine yakın/akraba sınıflara karışıklık yaşadığını gösterir."
    )

    # ═════════════════════════════════════════════
# SAYFA: SONUÇ VE KAYNAKÇA (YENİ)
# ═════════════════════════════════════════════
elif secim == "Sonuç ve Kaynakça":
    st.title("📌 Sonuç ve Kaynakça")
    
    st.subheader("Genel Değerlendirme ve Sonuç")
    st.markdown("""
    Bu bitirme projesi kapsamında, 8 farklı kan hücresi sınıfını ayırt edebilen yapay zeka destekli bir **Uzman Asistan Sistemi** başarıyla geliştirilmiştir. 
    
    Projenin mühendislik ve sağlık bilişimi açısından en temel başarıları şunlardır:
    - Tıbbi görüntüleme gibi hata toleransının düşük olduğu bir alanda **%98.01** gibi yüksek bir test doğruluğuna ulaşılmıştır.
    - **Transfer Learning (ResNet50)** yöntemi kullanılarak kısıtlı tıbbi veriyle dahi aşırı öğrenmenin (overfitting) önüne geçilmiş ve modelin genelleme yeteneği maksimuma çıkarılmıştır.
    """)
    
    st.divider()
    
    st.subheader("📚 Kaynakça")
    st.markdown("""
    1. **Veri Seti:** Unclesamulus. (2023). *Blood Cells Image Dataset*, Kaggle. Erişim adresi: [Kaggle](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset)
    2. **Model Mimarisi:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 770-778.
    3. **Derin Öğrenme Çerçevesi:** Abadi, M. et al. (2015). *TensorFlow: Large-scale machine learning on heterogeneous systems*.
    4. **Web Arayüzü ve Dağıtım:** Streamlit. (2024). *Streamlit: The fastest way to build and share data apps*.
    5. **Tıbbi Görselleştirme:** Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. Computing in science & engineering, 9(3), 90-95.
    """)
    
