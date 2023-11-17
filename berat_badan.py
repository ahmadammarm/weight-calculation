import cv2
import mediapipe as mp

def calculate_weight(shoulder_width):
    k = 2
    estimated_weight = k * shoulder_width 
    return round(estimated_weight)

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)  # Gunakan kamera
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
                
                # Hitung estimasi berat badan berdasarkan lebar bahu
                estimated_weight = calculate_weight(shoulder_width_px)
                
                # Tampilkan hasil estimasi
                cv2.putText(image, f"Estimated weight: {estimated_weight} kg", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Tampilkan landmark pada tubuh
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Weight Measurement', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
