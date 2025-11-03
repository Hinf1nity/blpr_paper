from PIL import Image, ImageEnhance
import os
import cv2


def juntar_imagenes(carpeta, imagenes_por_fila=7):
    # Lista de imágenes en la carpeta con nombres numéricos
    imagenes = [f for f in os.listdir(
        carpeta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Ordenar las imágenes por su número (suponiendo que el nombre de archivo es solo un número)
    # Ordena por el número antes de la extensión
    imagenes.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # Cargar todas las imágenes
    imagenes_cargadas = [Image.open(
        os.path.join(carpeta, img)) for img in imagenes]

    # Determinar el tamaño máximo de cada imagen (suponiendo que todas tienen el mismo tamaño)
    ancho, alto = imagenes_cargadas[0].size
    filas = (len(imagenes_cargadas) // imagenes_por_fila) + \
        (1 if len(imagenes_cargadas) % imagenes_por_fila > 0 else 0)

    # Crear una nueva imagen con el tamaño adecuado
    nueva_imagen = Image.new('RGB', (imagenes_por_fila * ancho +
                             (imagenes_por_fila-1)*50, filas * alto + (filas-1)*50), (255, 255, 255))

    # Colocar las imágenes en la nueva imagen
    for i, img in enumerate(imagenes_cargadas):
        fila = i // imagenes_por_fila
        columna = i % imagenes_por_fila
        nueva_imagen.paste(img, (columna * (ancho + 50), fila * (alto+50)))
    # Guardar la imagen resultante
    nueva_imagen.save("imagenes_combinadas_2.png")

    # # Dibujar las etiquetas en cada imagen
    # texto = ["a)", "b)", "c)", "d)", "e)", "f)"]
    # font = cv2.FONT_HERSHEY_TRIPLEX
    # font_scale = 4.5
    # thickness = 6
    # color = (0, 0, 0)
    # canvas = cv2.imread("imagenes_combinadas_2.png")
    # for i in range(len(imagenes_cargadas)):
    #     fila = i // imagenes_por_fila
    #     columna = i % imagenes_por_fila
    #     x = columna * (ancho + 50) + ancho//2 - 20
    #     y = fila * (alto+160) + alto + 120
    #     cv2.putText(canvas, texto[i], (x, y), font,
    #                 font_scale, color, thickness, cv2.LINE_AA)
    # # Guardar la imagen con las etiquetas
    # cv2.imwrite("imagenes_combinadas_3.png", canvas)


# Usar la función
# Cambia esto a la ruta donde están tus imágenes
carpeta = "joins"
juntar_imagenes(carpeta)
