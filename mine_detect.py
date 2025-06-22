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
img = np.full((720, 1280, 3), (255,255,255), dtype=np.uint8)  # Background putih default

# Satu window utama
WINDOW_NAME = "MainBotolDetection"
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Fungsi tampilan frame fullscreen
def show_fullscreen_frame(frame, delay=1000):
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(delay)

# Fungsi tampilan teks di tengah layar
def show_text_screen(text, bg_color=(217, 149, 41), font_color=(194, 220, 242),
                     font_scale=5, thickness=10, delay=1000):
    img[:] = bg_color
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness)
    show_fullscreen_frame(img, delay)

# Countdown tampilan
def countdown():
    for i in range(3, 0, -1):
        show_text_screen(str(i), bg_color=(217, 149, 41), font_color=(194, 220, 242), font_scale=10, thickness=20, delay=1000)

# Ambil gambar dari sumber
def capture_image():
    countdown()
    show_text_screen("Sedang Mengambil Gambar", bg_color=(194, 190, 202), font_color=(140, 118, 20),
                     font_scale=1, thickness=3, delay=1000)

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

# Deteksi objek
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
            bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
                           (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
            color = bbox_colors[classidx % 10]
            label_text = f'{classname}: {int(conf*100)}%'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite('capture.png', frame)
    show_fullscreen_frame(frame, delay=3000)
    return jumlah_botol_plastik, jumlah_botol_kaca

# Kirim data ke API dan tampilkan QR
def send_data(jumlah_botol_plastik, jumlah_botol_kaca):
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

    # Tampilkan QR + info
    img[:] = (194, 190, 202)
    qr = qrcode.QRCode(box_size=10, border=2)
    link = "https://daur-uang.my.id/scan-token/" + token
    qr.add_data(link)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_img = np.array(qr_img)
    qr_img = cv2.cvtColor(qr_img, cv2.COLOR_RGB2BGR)
    qr_img = cv2.resize(qr_img, (300, 300))

    x_offset = img.shape[1]//2 - qr_img.shape[1]//2
    y_offset = img.shape[0]//2 - qr_img.shape[0]//2 - 100
    img[y_offset:y_offset+qr_img.shape[0], x_offset:x_offset+qr_img.shape[1]] = qr_img

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

    show_fullscreen_frame(img, delay=1000)
    while True:
        key = cv2.waitKey(1)
        if key == ord('l'):
            break

# Fungsi utama kedua
def main_second():
    frame = capture_image()
    if user_res:
        resW, resH = map(int, user_res.split('x'))
        frame = cv2.resize(frame, (resW, resH))
    jumlah_botol_plastik, jumlah_botol_kaca = detect_objects(frame)

    if jumlah_botol_plastik != 0 or jumlah_botol_kaca != 0:
        send_data(jumlah_botol_plastik, jumlah_botol_kaca)
    else:
        cv2.waitKey(1000)

    print('Secondary Function Telah Berjalan.')
    main()

# Fungsi utama
def main():
    video_path = "vid/main.mp4"
    waitingroom = cv2.VideoCapture(video_path)

    fps = waitingroom.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    while waitingroom.isOpened():
        ret, frame = waitingroom.read()
        if not ret:
            waitingroom.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('p'):
            waitingroom.release()
            time.sleep(0.1)
            main_second()
            break
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
