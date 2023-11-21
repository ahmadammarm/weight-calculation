# Anggota Kelompok
# 1. Abdullah Sholum (220535608686)
# 2. Adil Zakaria (2205356080453)
# 3. Ahmad Ammar Musyaffa (220535601431)
# 4. Dafa Fadhilah Hilmi (220535610309)
# 5. Emiriopriimo Nadyzar Baruna (220535604509)


import cv2
import mediapipe as mp

def calculate_weight(shoulder_width):
    k = 2
    scaling_factor = 0.4
    estimated_weight = scaling_factor * k * shoulder_width
    return round(estimated_weight) 

def calculate_height(shoulder_width):
    k = 5
    scaling_factor = 0.4
    estimated_height = scaling_factor * k * shoulder_width
    return round(estimated_height)

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Ubah warna menjadi RGB dan deteksi landmark tubuh
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Ambil landmark untuk bahu kiri dan kanan
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                
                # Hitung lebar bahu berdasarkan landmark (dalam piksel)
                shoulder_width_px = abs(right_shoulder.x - left_shoulder.x) * image.shape[1]
                
                # Tampilkan estimasi tinggi badan
                estimated_height = calculate_height(shoulder_width_px)
                cv2.putText(image, f"Estimated height: {estimated_height} cm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Hitung estimasi berat badan berdasarkan lebar bahu
                estimated_weight = calculate_weight(shoulder_width_px)
                cv2.putText(image, f"Estimated weight: {estimated_weight} kg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                
                # Tampilkan landmark pada tubuh
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Weight Measurement', image)
            key = cv2.waitKey(5)
            if key == 27 or cv2.getWindowProperty('Weight Measurement', cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

main()
