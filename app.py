import streamlit as st
import cv2
import math
import numpy as np
from datetime import datetime
from util import get_parking_spots_bboxes, empty_or_not

# Streamlit UI untuk memilih CCTV Video
st.title("Pemantauan Tempat Parkir")

video_option = st.selectbox("Pilih CCTV Video", ["CCTV Video 1", "CCTV Video 2"])

# Tentukan mask_path dan video_path berdasarkan pilihan CCTV Video
if video_option == "CCTV Video 1":
    mask_path = './mask_crop.png'
    video_path = './data/parking_crop_loop.mp4'
elif video_option == "CCTV Video 2":
    mask_path = './mask_1920_1080.png'
    video_path = './data/parking_1920_1080_loop.mp4'

# Load mask
mask = cv2.imread(mask_path, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# Variabel status, timer, dan perbedaan frame
spots_status = [None for _ in spots]
timers = [None for _ in spots]
diffs = [None for _ in spots]
previous_frame = None
step = 30  # Jumlah frame sebelum pembaruan status

# Tambahkan placeholder untuk status parkir
parking_status_placeholder = st.empty()

# Fungsi untuk menghitung perbedaan frame
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Fungsi untuk memperbarui status parkir
def update_parking_status(index, status, waktu_masuk=None, waktu_keluar=None, lama_parkir=None, harga=None):
    if not status:
        parking_status_placeholder.write(
            f"""
            **Lahan {index + 1}:**
            - Status: Terisi
            - Waktu Masuk: {waktu_masuk}
            - Waktu Keluar: {waktu_keluar}
            - Lama Parkir: {lama_parkir}
            - Total Harga: Rp. {harga}
            """
        )

# Fungsi untuk memulai pemantauan
def start_monitoring():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Gagal membuka video. Periksa path dan file.")
        return

    frame_nmr = 0
    st_frame = st.empty()  # Tempat untuk menampilkan video
    previous_frame = None  # Inisialisasi previous_frame

    # Membuat dua kolom: kolom pertama untuk video, kolom kedua untuk status parkir
    col1, col2 = st.columns([5, 1])  # Kolom pertama lebih lebar

    # Di kolom pertama, tampilkan video
    with col1:
        st_frame = st.empty()  # Tempat untuk menampilkan video

    # Di kolom kedua, tampilkan status parkir
    with col2:
        parking_status_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Berhenti jika video habis

        # Inisialisasi timer pada frame pertama
        if frame_nmr == 0:
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spots_status[spot_indx] = empty_or_not(spot_crop)
                if spots_status[spot_indx]:  # Jika lahan terisi, inisialisasi timer
                    timers[spot_indx] = datetime.now()

        # Perbarui status setiap 'step' frame
        if frame_nmr % step == 0:
            if previous_frame is not None:
                for spot_indx, spot in enumerate(spots):
                    x1, y1, w, h = spot
                    spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                    diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

            for spot_indx in range(len(spots)):
                x1, y1, w, h = spots[spot_indx]
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = empty_or_not(spot_crop)

                # Update status jika ada perubahan
                if spots_status[spot_indx] != spot_status:
                    spots_status[spot_indx] = spot_status

                    if spot_status:  
                        if timers[spot_indx] is not None:
                            parked_time = datetime.now() - timers[spot_indx]
                            parked_seconds = parked_time.total_seconds()

                            num_intervals = math.ceil(parked_seconds / 6)  
                            parked_price = num_intervals * 3000  

                            update_parking_status(
                                spot_indx, False,
                                waktu_keluar=datetime.now(),
                                lama_parkir=f"{parked_time}",
                                harga=parked_price,
                                waktu_masuk=timers[spot_indx]
                            )
                            timers[spot_indx] = None 
                        
                    else:  
                        if timers[spot_indx] is None:  
                            timers[spot_indx] = datetime.now()

        # Visualisasi hasil
        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]  # Status lahan (kosong atau terisi)
            x1, y1, w, h = spots[spot_indx]

            if spot_status: 
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                timers[spot_indx] = None  

            else:  
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

                # Pastikan timer di-reset saat lahan kosong
                if timers[spot_indx] is None:
                    timers[spot_indx] = datetime.now()  # Set timer jika lahan pertama kali terisi

                # Hitung waktu parkir untuk lahan yang terisi
                parked_time = datetime.now() - timers[spot_indx]
                parked_seconds = parked_time.total_seconds()  # Mengambil total detik

                # Pembulatan ke atas untuk kelipatan 6 detik
                num_intervals = math.ceil(parked_seconds / 6)  # Pembulatan ke atas berdasarkan 6 detik

                # Hitung harga berdasarkan kelipatan 6 detik
                parked_price = num_intervals * 3000 

                # Tampilkan waktu parkir dan estimasi harga di layar
                parked_time_str = str(parked_time).split('.')[0]  # Waktu parkir dalam format detik
                price_str = f"Rp. {parked_price}"  # Harga dalam format Rp.

                # No Lahan
                cv2.putText(frame, f"No: {spot_indx + 1}", (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

                # Tampilkan waktu parkir
                cv2.putText(frame, f"{parked_time_str}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

                # Tampilkan harga parkir
                cv2.putText(frame, f"{price_str}", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

        # Tampilkan frame ke halaman Streamlit
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Simpan frame sebelumnya
        if frame_nmr % step == 0:
            previous_frame = frame.copy()

        frame_nmr += 1

    cap.release()

# Fungsi untuk memulai pemantauan saat tombol "Mulai Pemantauan" ditekan
if st.button("Mulai Pemantauan"):
    start_monitoring()
