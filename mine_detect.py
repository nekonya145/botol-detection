import os
import sys
import cv2
import numpy as np
import requests
import uuid
import argparse
import time
from ultralytics import YOLO
import qrcode
from PIL import Image

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")', default="my_model.pt")
parser.add_argument('--source', help='Image source, can be image file, image folder, video file, or camera index (e.g., "usb0" or "picamera0")', default="usb0")
parser.add_argument('--resolution', help='Resolution in WxH (example: "640x480")', default="1280x720")
args = parser.parse_args()

user_res = args.resolution

# Validasi Model Path
if not os.path.exists(args.model):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

model = YOLO(args.model, task='detect')
labels = model.names
img = np.full((720, 1280, 3), (255,255,255), dtype=np.uint8)  # Buat BG Putih

# Fungsi untuk hitung mundur
def countdown():
    window_name = "Countdown"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i in range(3, 0, -1):  # Countdown dari 3 ke 1
        img[:] = (217, 149, 41)
        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 10
        font_thickness = 20
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2

        cv2.putText(img, text, (text_x, text_y), font, font_scale, (194, 220, 242), font_thickness)
        cv2.imshow("Countdown", img)
        cv2.waitKey(1000)

# Fungsi untuk mengambil gambar dari sumber
def capture_image():
    countdown()

    window_name = "Mengambil Gambar"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    text = str("Sedang Mengambil Gambar")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    img[:] = (194, 190, 202)
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (140, 118, 20), font_thickness)
    
    cv2.imshow("Mengambil Gambar", img)
    cv2.waitKey(1000)
    cv2.destroyWindow("Countdown")  # Tutup window setelah countdown selesai

    if args.source.startswith('usb') or args.source.startswith('picamera'):
        cam_index = int(args.source.replace('usb', '').replace('picamera', ''))
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print('ERROR: Kamera tidak terdeteksi.')
            sys.exit(0)

        if args.resolution:
            width, height = map(int, args.resolution.split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ret, frame = cap.read()
        cap.release()
        if not ret:
            print('ERROR: Gagal mengambil gambar.')
            sys.exit(0)

        return frame

    elif os.path.isfile(args.source):
        frame = cv2.imread(args.source)
        if frame is None:
            print('ERROR: Gagal membaca file gambar.')
            sys.exit(0)
        return frame

    elif os.path.isdir(args.source):
        images = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        if not images:
            print('ERROR: Tidak ada gambar valid di folder.')
            sys.exit(0)
        print(f'Memproses {len(images)} gambar dari folder {args.source}')
        return images

    else:
        print('ERROR: Sumber gambar tidak valid.')
        sys.exit(0)
   
# Fungsi untuk mendeteksi objek dan menghitung jumlah botol plastik dan kaca
def detect_objects(frame):
    results = model(frame, verbose=False)
    detections = results[0].boxes

    jumlah_botol_plastik = 0
    jumlah_botol_kaca = 0

    for i in range(len(detections)):
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()
        x1, y1, x2, y2 = map(int, detections[i].xyxy[0].tolist())

        if conf > 0.5:
            if classname == 'plastik':
                jumlah_botol_plastik += 1
            elif classname == 'kaca':
                jumlah_botol_kaca += 1

            # Gambar bounding box dan label
            bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
            color = bbox_colors[classidx % 10]
            label_text = f'{classname}: {int(conf*100)}%'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Simpan dan tampikan hasil deteksi
    cv2.imwrite('capture.png', frame)
    cv2.namedWindow("Hasil Deteksi", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Hasil Deteksi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Hasil Deteksi", frame)
    cv2.waitKey(3000)
    cv2.destroyWindow("Mengambil Gambar")

    return jumlah_botol_plastik, jumlah_botol_kaca

# Fungsi untuk mengirim data ke API
def send_data(jumlah_botol_plastik, jumlah_botol_kaca):
    # URL API endpoint
    url = 'https://daur-uang.my.id/api/points'
    response = requests.get(url)
    data = response.json()
    nilai_plastik = data.get('nilai_plastik')
    nilai_kaca = data.get('nilai_kaca')

    jumlah_point_transaksi = jumlah_botol_plastik * nilai_plastik + jumlah_botol_kaca * nilai_kaca
    token = str(uuid.uuid4().hex[:6])
    jawa_adalah_kunci = "76andi92me45l77i"

    print('\nData yang akan dikirim:')
    print(f'Jumlah Botol Plastik: {jumlah_botol_plastik}')
    print(f'Jumlah Botol Kaca: {jumlah_botol_kaca}')
    print(f'Jumlah Point Transaksi: {jumlah_point_transaksi}')
    print(f'Token: {token}')

    data = {
        'token': token,
        'botol_plastik': jumlah_botol_plastik,
        'botol_kaca': jumlah_botol_kaca,
        'jumlah_point': jumlah_point_transaksi,
        'api_key' : jawa_adalah_kunci
    }

    try:
        response = requests.post('https://daur-uang.my.id/api/points', json=data)
        if response.status_code == 200:
            print('Data berhasil dikirim:', response.json())
        else:
            print(f'Gagal mengirim data. Status code: {response.status_code}', response.text)
    except Exception as e:
        print('Error saat mengirim data:', e)
    
    # --- TAMPILKAN QR + TULISAN ---
    cv2.namedWindow("QR Token Display", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("QR Token Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Buat background kosong
    img = np.full((720, 1280, 3), (194, 190, 202), dtype=np.uint8)

    # Generate QR Code menggunakan PIL
    qr = qrcode.QRCode(box_size=10, border=2)
    link = "https://daur-uang.my.id/scan-token/" + token
    qr.add_data(link)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    # Konversi PIL -> OpenCV (np.array)
    qr_img = np.array(qr_img)
    qr_img = cv2.cvtColor(qr_img, cv2.COLOR_RGB2BGR)
    qr_img = cv2.resize(qr_img, (300, 300))

    # Tempelkan QR ke tengah layar
    x_offset = img.shape[1]//2 - qr_img.shape[1]//2
    y_offset = img.shape[0]//2 - qr_img.shape[0]//2 - 100
    img[y_offset:y_offset+qr_img.shape[0], x_offset:x_offset+qr_img.shape[1]] = qr_img

    # Tampilkan informasi di bawah QR
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    warna_teks = (0, 0, 0)

    texts = [
        f"Total Plastik = {jumlah_botol_plastik} pcs",
        f"Total Kaca    = {jumlah_botol_kaca} pcs",
        f"Anda mendapatkan sebanyak {jumlah_point_transaksi} pts",
        f"",
        f"Token Anda adalah {token}"
    ]

    for i, line in enumerate(texts):
        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = y_offset + qr_img.shape[0] + 50 + (i * 40)
        cv2.putText(img, line, (text_x, text_y), font, font_scale, warna_teks, font_thickness)

    cv2.imshow("QR Token Display", img)
    cv2.waitKey(1000)
    cv2.destroyWindow("Hasil Deteksi")
    while True:
        key = cv2.waitKey(1)
        if key == ord('l'):
            break
    cv2.destroyWindow("QR Token Display")

# Fungsi utama
def main_second():
    frame = capture_image()
    if user_res:
        resize = True
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
        frame = cv2.resize(frame,(resW,resH))
    jumlah_botol_plastik, jumlah_botol_kaca = detect_objects(frame)

    if jumlah_botol_plastik != 0 or jumlah_botol_kaca != 0:
        send_data(jumlah_botol_plastik, jumlah_botol_kaca)

    print('Secondary Function Telah Berjalan.')
    main()

def main():
    # Buka video dalam mode fullscreen
    video_path = "vid/main.mp4"
    waitingroom = cv2.VideoCapture(video_path)
    cv2.namedWindow("WaitingRoom", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("WaitingRoom", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Ambil FPS dari video dan hitung delay yang sesuai
    fps = waitingroom.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30  # Default delay jika FPS tidak terdeteksi

    while waitingroom.isOpened():
        ret, frame = waitingroom.read()
        if not ret:
            waitingroom.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Ulangi video jika selesai
            continue

        cv2.imshow("WaitingRoom", frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('p'):  # Trigger ketika 'p' ditekan
            waitingroom.release()
            time.sleep(0.1)
            main_second()
            break

        if key == ord('q'):  # Keluar jika 'q' ditekan
            break  

if __name__ == '__main__':
    main()
