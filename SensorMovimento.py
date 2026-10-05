import cv2

CAMERA_INDEX = 0
WINDOW_NAME = "Diferenca"

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Não foi possível abrir a webcam.")
        cap.release()
        return

    try:
        ret, frame1 = cap.read()
        if not ret:
            print("Não foi possível capturar imagens da webcam.")
            return

        ret, frame2 = cap.read()
        if not ret:
            print("Não foi possível capturar imagens da webcam.")
            return

        while True:
            dif = cv2.absdiff(frame1, frame2)
            frame_gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilat = cv2.dilate(thresh, None, iterations=2)

            contornos, _ = cv2.findContours(dilat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contornos:
                if cv2.contourArea(contour) < 5000:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print('Alerta, movimento detectado!')

            cv2.imshow(WINDOW_NAME, frame1)

            frame1 = frame2
            ret, frame2 = cap.read()
            if not ret:
                print("A captura da webcam foi interrompida.")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
