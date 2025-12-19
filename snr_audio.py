import streamlit as st
from numpy import *
from matplotlib.pyplot import *
import soundfile

st.markdown("### 🔊 Signal-to-Noise Ratio (SNR)")
st.markdown(
    "Experience how different noise levels affect an audio signal. "
    "**Positive SNR** means the signal is stronger; **Negative SNR** means the noise is stronger."
)

signal,fe = soundfile.read('par8.wav')

t=arange(0,len(signal))/fe;
signal_power_dB=10*log10(var(signal)) #-14,4dB
noise = random.normal(0, 1, signal.shape)
noise_power_dB=10*log10(var(noise))   #0dB
SNR_dB=signal_power_dB-noise_power_dB #76dB
#Create a noisy signal with de SNR of x dB = increase the noise power by (76-x)dB
noise_ampl=noise*sqrt(10**((-14.4-snr)/10.0))
signal_plus_noise=signal+noise_ampl

fig,ax=subplots(figsize=(10,4))
plot(t,signal_plus_noise);
xlabel('time [s]')
title('signal + noise');
ylim(-2,2)
text(0.53,1.65,'SNR [dB]='+str(around(snr,2)),fontsize='xx-large')
st.pyplot(fig)

st.audio(signal_plus_noise,sample_rate=fe)

with st.expander("Open for comments"):
   st.markdown('''SNR is defined as the ratio of the power of a signal to that of the additional 
               noise: ''')
   st.latex('''SNR = \sigma_x^2 / \sigma_n^2 ''')
   st.markdown('''or, in deciblels:''')
   st.latex('''SNR [dB]= 10 \ \log_{10}(\sigma_x^2 / \sigma_n^2) ''')
   st.markdown('''The _SNR_ imposed above is obtained by measuring the power of the input speech, 
               and adding noise with unity power (i.e., 0 dB), multiplied by an amplitude 
               factor $\sigma_n$: ''')
   st.latex('''\sigma_n = \sqrt{\sigma_x^2 / SNR} ''')
   st.markdown('''After playing with the _SNR_ slider, it is clear that noise can be quickly seen 
   and heard when _SNR_ <40''')


# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# # --- Page Config & CSS ---
# st.set_page_config(
#     page_title="Signal-to-Noise Ratio (SNR)",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# st.markdown("""
#     <style>
#         .block-container {
#             padding-top: 1rem;
#             padding-bottom: 2rem;
#             padding-left: 2rem;
#             padding-right: 2rem;
#         }
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         header {visibility: hidden;}
#     </style>
# """, unsafe_allow_html=True)

# # --- Title ---
# st.markdown("### 🔊 Signal-to-Noise Ratio (SNR)")
# st.markdown(
#     "Experience how different noise levels affect an audio signal. "
#     "**Positive SNR** means the signal is stronger; **Negative SNR** means the noise is stronger."
# )

# # --- Controls ---
# with st.container(border=True):
#     # Slider with a default of 20dB (clear but audible noise)
#     target_snr_db = st.slider("Target SNR (dB)", -10, 40, 20, 1, 
#                               help="Higher = Clearer. Lower = Noisier.")

# # --- Signal Generation (Synthetic) ---
# # We replace 'par8.wav' with a synthetic AM signal so this runs standalone.
# fs = 16000
# duration = 2.0
# t = np.arange(int(fs * duration)) / fs

# # Create a "pulse" like signal (Amplitude Modulation)
# # Carrier: 440Hz (A4), Modulator: 2Hz (makes it pulse)
# clean_signal = 0.8 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))

# # --- SNR Calculation & Mixing ---
# # 1. Measure Power of Clean Signal
# # Power = Mean Square
# sig_power = np.mean(clean_signal ** 2)
# sig_power_db = 10 * np.log10(sig_power)

# # 2. Calculate Required Noise Power
# # Target SNR_dB = 10 * log10(P_signal / P_noise)
# # P_noise = P_signal / 10^(SNR/10)
# noise_power_req = sig_power / (10 ** (target_snr_db / 10))

# # 3. Generate White Noise
# white_noise = np.random.normal(0, 1, clean_signal.shape)

# # Scale noise to match required power
# # Current noise power is approx 1 (since std=1). We scale by sqrt(req / current)
# current_noise_power = np.mean(white_noise ** 2)
# scale_factor = np.sqrt(noise_power_req / current_noise_power)
# noise_scaled = white_noise * scale_factor

# # 4. Mix
# noisy_signal = clean_signal + noise_scaled

# # 5. Safe Audio Normalization
# # If we add loud noise, the values might exceed +/- 1.0, causing clipping distortion.
# # We normalize the FINAL mixed signal to fit within [-1, 1] for playback.
# max_amp = np.max(np.abs(noisy_signal))
# if max_amp > 1.0:
#     audio_signal = noisy_signal / max_amp
# else:
#     audio_signal = noisy_signal

# # --- Visualization ---
# # We split into two columns: Plot and Audio/Stats
# col1, col2 = st.columns([2, 1])

# with col1:
#     fig, ax = plt.subplots(figsize=(8, 3.5))
#     fig.patch.set_alpha(0)
#     ax.patch.set_alpha(0)
    
#     # Plot a snippet (0.5 seconds)
#     zoom_samples = int(0.5 * fs)
#     t_zoom = t[:zoom_samples]
    
#     # Plot Clean Signal (Faint)
#     ax.plot(t_zoom, clean_signal[:zoom_samples], 
#             label="Clean Signal", color="limegreen", alpha=0.6, lw=1.5, ls="--")
    
#     # Plot Noisy Signal
#     ax.plot(t_zoom, noisy_signal[:zoom_samples], 
#             label="Noisy Signal", color="white", alpha=0.7, lw=0.8)
    
#     ax.set_title(f"Waveform (First 0.5s) | SNR = {target_snr_db} dB", color="gray", loc="left")
#     ax.set_xlabel("Time [s]", color="gray")
#     ax.set_ylim(-3, 3) # Fixed limits so we see the noise grow
    
#     # Styling
#     ax.legend(loc="upper right", frameon=False, fontsize="small")
#     ax.grid(True, alpha=0.2, ls="--")
#     ax.spines['bottom'].set_color('gray')
#     ax.spines['left'].set_color('gray')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.tick_params(axis='x', colors='gray')
#     ax.tick_params(axis='y', colors='gray')
    
#     st.pyplot(fig, use_container_width=True)

# with col2:
#     st.markdown("#### 🎧 Listen")
#     st.audio(audio_signal, sample_rate=fs)
    
#     st.markdown("#### 📊 Stats")
#     st.markdown(f"""
#     - **Signal Power:** {sig_power_db:.1f} dB
#     - **Noise Power:** {10*np.log10(noise_power_req):.1f} dB
#     """)
#     if target_snr_db < 0:
#         st.error(f"Noise is {abs(target_snr_db)} dB louder than signal!", icon="⚠️")
#     elif target_snr_db < 10:
#         st.warning("Signal is barely visible.", icon="📉")
#     else:
#         st.success("Signal is clear.", icon="✅")

# # --- Explanation ---
# with st.expander("📚 The Math Behind SNR"):
#     st.markdown(r"""
#     **Signal-to-Noise Ratio (SNR)** compares the level of a desired signal to the level of background noise.
    
#     $$ \text{SNR}_{dB} = 10 \log_{10} \left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right) $$
    
#     Where $P$ is power (variance of the signal).
    
#     In this demo, we calculate the noise amplitude needed to reach your target SNR using:
    
#     $$ \sigma_{\text{noise}} = \sqrt{ \frac{P_{\text{signal}}}{10^{\frac{\text{SNR}}{10}}} } $$
    
    
#     """)



