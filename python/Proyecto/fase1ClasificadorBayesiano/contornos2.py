import cv2

# Leer la imagen en escala de grises
img = cv2.imread('/Users/chocoplot/Documents/codeLAB/signal_Recognition/imagen_Ccontornos.png', cv2.IMREAD_GRAYSCALE)

# Aplicar umbral para obtener una imagen binaria
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Encontrar los contornos en la imagen binaria
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Para cada contorno
for cnt in contours:
    # Aproximar un polígono que tenga cuatro lados
    approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt,True),True)

    # Comprobar si el polígono aproximado es un cuadrilátero convexo
    if len(approx) == 4 and cv2.isContourConvex(approx):
        # Dibujar un rectángulo alrededor del contorno
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# Mostrar la imagen con los contornos detectados
cv2.imshow('Contornos', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
