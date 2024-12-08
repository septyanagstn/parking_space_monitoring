# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# from util import get_parking_spots_bboxes, empty_or_not


# def calc_diff(im1, im2):
#     return np.abs(np.mean(im1) - np.mean(im2))


# mask = './mask_1920_1080.png'
# video_path = './data/parking_1920_1080_loop.mp4'


# mask = cv2.imread(mask, 0)

# cap = cv2.VideoCapture(video_path)

# connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# spots = get_parking_spots_bboxes(connected_components)

# spots_status = [None for j in spots]
# diffs = [None for j in spots]

# previous_frame = None

# frame_nmr = 0
# ret = True
# step = 30
# while ret:
#     ret, frame = cap.read()

#     if frame_nmr % step == 0 and previous_frame is not None:
#         for spot_indx, spot in enumerate(spots):
#             x1, y1, w, h = spot

#             spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

#             diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

#         print([diffs[j] for j in np.argsort(diffs)][::-1])

#     if frame_nmr % step == 0:
#         if previous_frame is None:
#             arr_ = range(len(spots))
#         else:
#             arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
#         for spot_indx in arr_:
#             spot = spots[spot_indx]
#             x1, y1, w, h = spot

#             spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

#             spot_status = empty_or_not(spot_crop)

#             spots_status[spot_indx] = spot_status

#     if frame_nmr % step == 0:
#         previous_frame = frame.copy()

#     for spot_indx, spot in enumerate(spots):
#         spot_status = spots_status[spot_indx]
#         x1, y1, w, h = spots[spot_indx]

#         if spot_status:
#             frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
#         else:
#             frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

#     cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
#     cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

#     frame_nmr += 1

# cap.release()
# cv2.destroyAllWindows()


import cv2
import math
import numpy as np
from datetime import datetime
from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Path file
mask = './mask_crop.png'
video_path = './data/parking_crop_loop.mp4'

# Load mask dan video
mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# Variabel status, timer, dan perbedaan frame
spots_status = [None for _ in spots]
timers = [None for _ in spots]
diffs = [None for _ in spots]

previous_frame = None
frame_nmr = 0
ret = True
step = 10

while ret:
    ret, frame = cap.read()

    if frame is None:  # Hentikan jika tidak ada frame lagi
        break

    # Inisialisasi timer pada frame pertama
    if frame_nmr == 0:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status
            if spot_status:  # Jika lahan terisi, inisialisasi timer
                timers[spot_indx] = datetime.now()

    # Perbarui frame setiap 'step' frame
    if frame_nmr % step == 0:
        if previous_frame is not None:
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        arr_ = range(len(spots)) if previous_frame is None else [
            j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

        for spot_indx in arr_:
            x1, y1, w, h = spots[spot_indx]
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)

            # Update status spot jika statusnya berubah
            if spots_status[spot_indx] != spot_status:
                spots_status[spot_indx] = spot_status  # Update status spot

                if spot_status:  
                    if timers[spot_indx] is not None:
                        parked_time = datetime.now() - timers[spot_indx]
                        parked_seconds = parked_time.total_seconds()

                        num_intervals = math.ceil(parked_seconds / 6)  
                        parked_price = num_intervals * 3000  

                        print(f"Lahan {spot_indx + 1} terisi. Waktu masuk: {timers[spot_indx]}")
                        print(f"Lahan {spot_indx + 1} kosong. Waktu Keluar: {datetime.now()}")
                        print(f"Lahan {spot_indx + 1}. Lama parkir: {parked_seconds}") 
                        print(f"Lahan {spot_indx + 1}, Total Harga : Rp. {parked_price}")
                        timers[spot_indx] = None 
                      
                else:  
                    if timers[spot_indx] is None:  
                        timers[spot_indx] = datetime.now()

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

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

            # Tampilkan waktu parkir
            cv2.putText(frame, f"{parked_time_str}", (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # Tampilkan harga parkir
            cv2.putText(frame, f"{price_str}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            

    # Tampilkan jumlah lahan kosong
    cv2.rectangle(frame, (45, 420), (330, 450), (0, 0, 0), -1)
    cv2.putText(frame, f'Available spots: {str(sum(spots_status))} / {len(spots_status)}', 
                (58, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Tampilkan frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
