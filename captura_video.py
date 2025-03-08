import cv2
import mediapipe as mp
import os
import csv

# Configuración
CARPETA_DATASET = "LESSA_Dataset"     
MAX_FRAMES_ESTATICA = 150            
FRAMES_MINIMOS_INICIO = 4            
FRAMES_MINIMOS_FINAL = 2             
MIN_DETECCION_CONFIANZA = 0.5        
MIN_SEGUIMIENTO_CONFIANZA = 0.5      
INDICE_CAMARA = 0                     
MOSTRAR_VIDEO = True                  


os.makedirs(CARPETA_DATASET, exist_ok=True)


etiqueta = input("Ingrese la etiqueta de la seña: ").strip()

tipo_senia = input("¿La seña es estática o dinámica? (E/D): ").strip().lower()
if tipo_senia not in ['e', 'd']:
    print("Entrada no válida. Ingrese 'E' para estática o 'D' para dinámica.")
    exit(1)


carpeta_senia = os.path.join(CARPETA_DATASET, etiqueta)
os.makedirs(carpeta_senia, exist_ok=True)


intentos_existentes = [d for d in os.listdir(carpeta_senia) if d.startswith("intento_")]


if intentos_existentes:
    numeros_intentos = [int(d.split("_")[1]) for d in intentos_existentes if d.split("_")[1].isdigit()]
    numero_intento = max(numeros_intentos) + 1 if numeros_intentos else 1
else:
    numero_intento = 1


carpeta_intento = os.path.join(carpeta_senia, f"intento_{numero_intento}")
os.makedirs(carpeta_intento, exist_ok=True)


archivo_csv = os.path.join(carpeta_intento, "coordenadas.csv")

print(f"\nGuardando datos en: {carpeta_intento}\n")


mp_holistic = mp.solutions.holistic
mp_dibujo = mp.solutions.drawing_utils


cap = cv2.VideoCapture(INDICE_CAMARA)
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit(1)


frames_con_manos = 0
frames_sin_manos = 0
capturando = False
contador_frames = 0


with mp_holistic.Holistic(min_detection_confidence=MIN_DETECCION_CONFIANZA,
                          min_tracking_confidence=MIN_SEGUIMIENTO_CONFIANZA) as holistic, \
     open(archivo_csv, 'w', newline='') as csvfile:

    escritor_csv = csv.writer(csvfile)
    

    encabezados = []
    for i in range(33):  
        encabezados += [f"cuerpo_{i}_x", f"cuerpo_{i}_y", f"cuerpo_{i}_z"]
    for i in range(21):  
        encabezados += [f"mano_izquierda_{i}_x", f"mano_izquierda_{i}_y", f"mano_izquierda_{i}_z"]
    for i in range(21): 
        encabezados += [f"mano_derecha_{i}_x", f"mano_derecha_{i}_y", f"mano_derecha_{i}_z"]
    escritor_csv.writerow(encabezados)

    print(">> Esperando detección de manos para iniciar captura...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer la cámara. Finalizando.")
            break

        
        imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_rgb.flags.writeable = False  
        resultados = holistic.process(imagen_rgb)  
        imagen_rgb.flags.writeable = True  
        imagen_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)  

        
        if resultados.pose_landmarks:
            mp_dibujo.draw_landmarks(imagen_bgr, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if resultados.left_hand_landmarks:
            mp_dibujo.draw_landmarks(imagen_bgr, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if resultados.right_hand_landmarks:
            mp_dibujo.draw_landmarks(imagen_bgr, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        
        manos_detectadas = resultados.left_hand_landmarks or resultados.right_hand_landmarks

        if manos_detectadas:
            frames_con_manos += 1
            frames_sin_manos = 0  
        else:
            frames_con_manos = 0
            frames_sin_manos += 1

        
        if not capturando and frames_con_manos >= FRAMES_MINIMOS_INICIO:
            capturando = True
            print(">> Manos detectadas de forma estable: iniciando captura...\n")

      
        if capturando and frames_sin_manos >= FRAMES_MINIMOS_FINAL:
            print(">> Manos no detectadas durante un tiempo prolongado: finalizando captura.\n")
            break

        if capturando:
            contador_frames += 1
            nombre_imagen = f"frame_{contador_frames}.jpg"
            
           
            cv2.imwrite(os.path.join(carpeta_intento, nombre_imagen), frame)

            
            coordenadas = []
            if resultados.pose_landmarks:
                for lm in resultados.pose_landmarks.landmark:
                    coordenadas += [lm.x, lm.y, lm.z]
            else:
                coordenadas += [0.0, 0.0, 0.0] * 33
            if resultados.left_hand_landmarks:
                for lm in resultados.left_hand_landmarks.landmark:
                    coordenadas += [lm.x, lm.y, lm.z]
            else:
                coordenadas += [0.0, 0.0, 0.0] * 21
            if resultados.right_hand_landmarks:
                for lm in resultados.right_hand_landmarks.landmark:
                    coordenadas += [lm.x, lm.y, lm.z]
            else:
                coordenadas += [0.0, 0.0, 0.0] * 21

           
            escritor_csv.writerow(coordenadas)

           
            if tipo_senia == 'e' and contador_frames >= MAX_FRAMES_ESTATICA:
                print(f">> Se alcanzó el máximo de {MAX_FRAMES_ESTATICA} frames. Captura detenida.\n")
                break

        
        cv2.putText(imagen_bgr, "Grabando" if capturando else "Esperando manos...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Captura de Señas", imagen_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            print(">> Captura interrumpida por el usuario.\n")
            break


cap.release()
cv2.destroyAllWindows()
print(f"\nCaptura completada. {contador_frames} frames guardados en '{carpeta_intento}'.\n")
