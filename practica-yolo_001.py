
import cv2
from ultralytics import YOLO

# 1. Cargar el modelo YOLO preentrenado
modelo = YOLO("yolo26n.pt")

# 2. Abrir la cámara web
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("YOLO iniciado correctamente.")
print("Presiona Q para salir.")

# 3. Procesar imágenes en tiempo real
while True:

    # Capturar un fotograma
    correcto, imagen = camara.read()

    if not correcto:
        print("Error al capturar imagen.")
        break

    # 4. Ejecutar detección con YOLO
    resultados = modelo.predict(
        source=imagen,
        conf=0.50,
        device="cpu",
        verbose=False
    )

    resultado = resultados[0]

    # 5. Contar los objetos detectados
    total_objetos = len(resultado.boxes)

    # 6. Dibujar cajas y etiquetas
    imagen_resultado = resultado.plot()

    # 7. Mostrar cantidad de objetos
    cv2.putText(
        imagen_resultado,
        f"Objetos detectados: {total_objetos}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # 8. Mostrar imagen procesada
    cv2.imshow("Deteccion de objetos - YOLO", imagen_resultado)

    # 9. Salir al presionar Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 10. Liberar recursos
camara.release()
cv2.destroyAllWindows()
