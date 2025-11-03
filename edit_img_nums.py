# import cv2
# import numpy as np

# # === 1. Cargar imagen y aumentar brillo ===
# img = cv2.imread("imagenes_combinadas_2.png")

# # === 2. Definir dimensiones del canvas ===
# margin_bottom = 60  # espacio para etiquetas X
# margin_left = 110    # espacio para etiquetas Y

# img_h, img_w = img.shape[:2]
# canvas_h = img_h + margin_bottom
# canvas_w = img_w + margin_left

# # Crear canvas transparente y pegar imagen
# canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
# canvas[0:img_h, margin_left:margin_left+img_w] = img

# # === 3. Parámetros de la grilla ===
# rows, cols = 4, 7
# cell_height = img_h // rows
# cell_width = img_w // cols

# x_labels = [-60, -40, -20, 0, 20, 40, 60]
# y_labels = [60, 40, 20, 0]

# # === 4. Estilo del texto ===
# font = cv2.FONT_HERSHEY_TRIPLEX
# font_scale = 2
# thickness = 6
# color = (0, 0, 0)

# # === 5. Dibujar etiquetas eje X (abajo del canvas) ===
# for i, angle in enumerate(x_labels):
#     x = margin_left + i * cell_width + cell_width // 2
#     y = canvas_h - 15
#     if i <= 2:
#         x -= 200
#     else:
#         x -= 50
#     cv2.putText(canvas, str(angle), (x, y), font,
#                 font_scale, color, thickness, cv2.LINE_AA)

# # === 6. Dibujar etiquetas eje Y (izquierda del canvas) ===
# for i, angle in enumerate(y_labels):
#     x = 10
#     y = i * cell_height + cell_height // 2 + 10
#     if i > 2:
#         x = 50
#     text = str(angle) + "\u00B0"
#     cv2.putText(canvas, text, (x, y), font,
#                 font_scale, color, thickness, cv2.LINE_AA)

# # === 7. Guardar imagen final ===
# cv2.imwrite("imagen_annotada_final.png", canvas)


from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np

# === 1. Cargar imagen y aumentar brillo ===
img = Image.open("imagenes_combinadas_2.png").convert("RGB")
font_path = "/System/Library/Fonts/Supplemental/Times_New_Roman.ttf"

# === 2. Crear canvas con márgenes ===
margin_bottom = 110   # espacio para etiquetas X
margin_left = 140    # espacio para etiquetas Y

img_w, img_h = img.size
canvas_w = img_w + margin_left
canvas_h = img_h + margin_bottom

# Crear canvas blanco y pegar imagen
canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")
canvas.paste(img, (margin_left, 0))

# === 3. Parámetros de la grilla ===
rows, cols = 4, 7
cell_height = img_h // rows
cell_width = img_w // cols

x_labels = [-60, -40, -20, 0, 20, 40, 60]
y_labels = [60, 40, 20, 0]

# === 4. Preparar para dibujar ===
draw = ImageDraw.Draw(canvas)

# Cargar fuente (usa una que soporte símbolos Unicode)
try:
    font = ImageFont.truetype(font_path, 90)
except IOError:
    font = ImageFont.load_default()
    print("No se pudo cargar la fuente personalizada, usando fuente por defecto.")

# === 5. Dibujar etiquetas eje X ===
for i, angle in enumerate(x_labels):
    x = margin_left + i * cell_width + cell_width // 2
    y = canvas_h - 90

    # Ajuste manual para alineación (opcional)
    offset_x = -50 if i > 2 else -200
    text = f"{angle}°"
    draw.text((x + offset_x, y), text, font=font, fill=(0, 0, 0))

# === 6. Dibujar etiquetas eje Y ===
for i, angle in enumerate(y_labels):
    x = 10 if i <= 2 else 50
    y = i * cell_height + cell_height // 2

    text = f"{angle}°"
    draw.text((x, y), text, font=font, fill=(0, 0, 0))

# === 7. Guardar imagen final ===
canvas.save("imagen_annotada_final_pil.png")
