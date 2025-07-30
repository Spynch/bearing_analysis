import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import spectrogram, filtfilt, iirnotch
from math import cos, radians
import io
from fpdf import FPDF
import os

st.set_page_config(layout="wide")

st.title("⚙️ Мультифазный анализ токов с диагностикой подшипников и PDF-отчётом")

uploaded_files = st.file_uploader(
    "Загрузите один или несколько CSV-файлов с колонками current_R, current_S, current_T",
    type=["csv"],
    accept_multiple_files=True
)

st.sidebar.header("Параметры подшипника")
rpm = st.sidebar.number_input("Скорость вращения (RPM)", value=1800, min_value=1)
n = st.sidebar.number_input("Число тел качения (n)", value=8, min_value=1)
d = st.sidebar.number_input("Диаметр тела качения d (мм)", value=8.0)
D = st.sidebar.number_input("Диаметр посадки D (мм)", value=40.0)
beta_deg = st.sidebar.number_input("Угол контакта β (градусы)", value=0.0)

FS = 10000
T = 1.0 / FS
PHASES = ['current_R', 'current_S', 'current_T']

if uploaded_files:
    dfs = [pd.read_csv(f) for f in uploaded_files]
    df = pd.concat(dfs, ignore_index=True)
    st.success(f"Объединено строк: {df.shape[0]}")

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="💾 Скачать объединённый merged_dataset.csv",
        data=csv_buffer.getvalue().encode('utf-8'),
        file_name='merged_dataset.csv',
        mime='text/csv'
    )

    f_rot = rpm / 60
    beta = radians(beta_deg)
    bpfo = (n / 2) * f_rot * (1 - (d / D) * cos(beta))
    bpfi = (n / 2) * f_rot * (1 + (d / D) * cos(beta))
    ftf = 0.5 * f_rot * (1 - (d / D) * cos(beta))
    bsf = (D / (2 * d)) * f_rot * (1 - ((d / D)**2 * cos(beta)**2))

    bearing_freqs = {"BPFO": bpfo, "BPFI": bpfi, "FTF": ftf, "BSF": bsf}

    st.subheader("Анализ всех фаз")
    pdf = FPDF()
    pdf.add_page()
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)
    pdf.cell(200, 10, txt="Отчет по анализу токов и диагностике подшипников", ln=True, align='C')

    for phase in PHASES:
        st.markdown(f"### Фаза: `{phase}`")
        signal_full = df[phase].values
        t_full = np.linspace(0.0, len(signal_full) * T, len(signal_full))

        start = st.number_input(f"Начальная точка (фаза {phase})", 0, len(signal_full)-1000, 0, step=1000, key=f"start_{phase}")
        end = st.number_input(f"Конечная точка (фаза {phase})", start+1000, len(signal_full), min(start+1000, len(signal_full)), step=1000, key=f"end_{phase}")

        signal = signal_full[start:end]
        t = t_full[start:end]
        N = len(signal)

        def notch_filter(signal, fs, target_freq, Q=30):
            b, a = iirnotch(target_freq, Q, fs)
            return filtfilt(b, a, signal)

        cleaned = signal.copy()
        for f0 in range(50, 351, 50):
            cleaned = notch_filter(cleaned, FS, f0)

        yf = fft(signal)
        xf = fftfreq(N, T)[:N // 2]
        amplitude = 2.0 / N * np.abs(yf[:N // 2])

        yf_clean = fft(cleaned)
        amp_clean = 2.0 / N * np.abs(yf_clean[:N // 2])
        reconstructed = ifft(yf_clean).real

        st.line_chart(pd.DataFrame({"Оригинал": signal, "Фильтр": reconstructed}, index=t))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xf, amplitude, label="Оригинал", alpha=0.5)
        ax.plot(xf, amp_clean, label="Фильтрованный", linestyle='--')
        ax.set_xlim(0, 500)
        ax.grid()
        for name, freq in bearing_freqs.items():
            ax.axvline(freq, color='r', linestyle=':', label=name)
        st.pyplot(fig)

        st.markdown("**Увеличение спектра (0–150 Гц)**")
        fig_zoom, axz = plt.subplots(figsize=(10, 3))
        axz.plot(xf, amplitude, color='blue')
        axz.set_xlim(0, 150)
        axz.grid()
        st.pyplot(fig_zoom)

        MAX_LEN = min(1_000_000, N)
        signal_for_spec = signal[:MAX_LEN]
        f_spec, t_spec, Sxx = spectrogram(signal_for_spec, fs=FS, nperseg=512)
        fig_spec, axs = plt.subplots(figsize=(10, 3))
        im = axs.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx), shading='gouraud')
        axs.set_ylim(0, 500)
        axs.set_xlabel("Время (с)")
        axs.set_ylabel("Частота (Гц)")
        fig_spec.colorbar(im, ax=axs, label='Амплитуда (дБ)')
        st.pyplot(fig_spec)

        detection_log = ""
        threshold = np.max(amplitude) * 0.3
        for name, freq in bearing_freqs.items():
            nearby = amplitude[np.abs(xf - freq) < 2]
            if len(nearby) and np.max(nearby) > threshold:
                detection_log += f"❗ Обнаружен пик вблизи {name} ({freq:.1f} Гц)\n"
            else:
                detection_log += f"✔ Нет пика около {name} ({freq:.1f} Гц)\n"
        st.text_area(f"Диагностика для {phase}", detection_log.strip(), height=120)

        pdf.multi_cell(0, 10, txt=f"Фаза {phase}:\n{detection_log}")

    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    st.download_button(
        label="📄 Скачать PDF-отчет",
        data=pdf_buffer.getvalue(),
        file_name="diagnostic_report.pdf",
        mime="application/pdf"
    )
