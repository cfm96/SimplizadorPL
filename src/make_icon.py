from PIL import Image, ImageDraw, ImageFont
import os

# Crear lienzo con transparencia
img = Image.new('RGBA', (256, 256), (255, 255, 255, 0))
draw = ImageDraw.Draw(img)

# Fondo blanco con esquinas redondeadas
draw.rounded_rectangle([16, 16, 240, 240], radius=48, fill=(255, 255, 255, 255))

# Letra S azul
text = 'S'
color = (0, 102, 204)

# Intentar fuente TrueType; fallback a default si no está
try:
    font = ImageFont.truetype('arial.ttf', 170)
except Exception:
    # La default es pequeña; ajustamos dibujo si es el caso
    font = ImageFont.load_default()

# Calcular posición centrada
bbox = draw.textbbox((0, 0), text, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
x = (256 - text_w) // 2
# Ajuste fino vertical para que visualmente quede centrada
y = (256 - text_h) // 2 - 6

# Dibujar texto
draw.text((x, y), text, fill=color, font=font)

# Guardar .ico con múltiples tamaños
ico_path = os.path.abspath('icon.ico')
img.save(ico_path, sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(24,24),(16,16)])
print(ico_path)
