import os
import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
import imgaug.augmenters as iaa
import imageio


class Dataset:
    def __init__(self,path):
        self.path = path
        self.foldersDataset=[]
        self.filesDataset=[]

    def getFolder(self,pathsb):
        # print(self.path)
        # print(pathsb)
        with os.scandir(pathsb) as ficheros:
            content = [fichero.name for fichero in ficheros if fichero.is_dir()]
        return content

    def treeFolders(self,path=''):
            try:
                # print("Hiw!",path,sep="xxjjjxx")
                ruta = self.path if path==''  else path
                folders=self.getFolder(ruta)
                memoryFolder=None 
                if len(self.foldersDataset)==0:
                    self.foldersDataset.append(ruta)
                else:
                    self.foldersDataset.append(ruta.replace(self.foldersDataset[0],''))
                for subfolder in folders:            
                    intern = f'{ruta}/{subfolder}'             
                    if os.path.isdir(intern):
                        self.treeFolders(intern)
                    if memoryFolder!=subfolder:
                        memoryFolder=subfolder
            except FileNotFoundError:
                print(f"Directory {path} not found!", file=sys.stderr)
                return
            except PermissionError:
                print(f"Insufficient permission to read {path}!", file=sys.stderr)
                return
            except IsADirectoryError:
                print(f"{path} is a directory!", file=sys.stderr)
                return

    def treeFiles(self,dir_path):
        try:    
            files = os.listdir(dir_path)
            for file in files:
                path = f'{dir_path}/{file}'
                if file=='.DS_Store':
                    continue
                if os.path.isdir(path):
                    self.treeFiles(path)
                else:
                    self.filesDataset.append(path)
        except FileNotFoundError:
                print(f"Directory {path} not found!", file=sys.stderr)
                return
        except PermissionError:
            print(f"Insufficient permission to read {path}!", file=sys.stderr)
            return
        except IsADirectoryError:
            print(f"{path} is a directory!", file=sys.stderr)
            return

class Imagen:
    def __init__(self,listImages):
        self.listImages = listImages
        self.df= pd.DataFrame(columns=['imagen_path','clase','h_mean', 's_mean',  'v_mean','mask_meanb', 'mask_meang', 'mask_meany', 'mask_meanrd', 'mask_meanrl', 'mask_stdb', 'mask_stdg', 'mask_stdy', 'mask_stdrd', 'mask_stdrl'])
        'h_mean', 's_mean',  'v_mean',
        # self.df = pd.DataFrame(columns=['imagen_path', 'width', 'height', 'formato', 'modo', 'bits','gray_mean','gray_std','gray_m1','gray_m2','gray_m3','gray_m4','gray_m5','gray_m6','gray_m7','b_mean','b_std','b_m1','b_m2','b_m3','b_m4','b_m5','b_m6','b_m7','g_mean','g_std','g_m1','g_m2','g_m3','g_m4','g_m5','g_m6','g_m7','r_mean','r_std','r_m1','r_m2','r_m3','r_m4','r_m5','r_m6','r_m7','h_mean','h_std','h_m1','h_m2','h_m3','h_m4','h_m5','h_m6','h_m7','s_mean','s_std','s_m1','s_m2','s_m3','s_m4','s_m5','s_m6','s_m7','v_mean','v_std','v_m1','v_m2','v_m3','v_m4','v_m5','v_m6','v_m7','l_mean','l_std','l_m1','l_m2','l_m3','l_m4','l_m5','l_m6','l_m7','a_mean','a_std','a_m1','a_m2','a_m3','a_m4','a_m5','a_m6','a_m7','bb_mean','bb_std','bb_m1','bb_m2','bb_m3','bb_m4','bb_m5','bb_m6','bb_m7'])

    def __resizeImagen(self,pathImagen,img):
        try:
            # sourcery skip: avoid-builtin-shadow
            new_width = 256
            new_height = 256

            # Redimensionar la imagen
            imagen_resized = img.resize((new_width, new_height))

            # Guardar la imagen con el mismo nombre en su ubicación de origen
            dir=os.path.dirname(pathImagen)            
            filename=os.path.basename(pathImagen)            
            nombre_archivo, extension = os.path.splitext(filename)
            pathSave=dir +'/resize25/' +nombre_archivo+'_resized' + extension
            imagen_resized.save(pathSave)
            print(f"resize of imagen --->{filename} !exitoso")

        except (FileNotFoundError, NameError):
            print(f"Directory {pathSave} not found!", file=sys.stderr)
            return
        except PermissionError:
            print(f"Insufficient permission to read {pathSave}!", file=sys.stderr)
            return
        except IsADirectoryError:
            print(f"{pathSave} is a directory!", file=sys.stderr)
            return
        # print(dir +'/resize/' +nombre_archivo+'_resized' + extension)
    def __rotationImagen(self,pathImagen,img):
        # Definir el directorio de destino de las nuevas imágenes generadas
        output_dir = "/Users/chocoplot/Documents/codeLAB/signal_Recognition/python/Proyecto/datasets/Dataset5/aumento"
        print(pathImagen,img)
        # Definir el número de imágenes a generar por imagen original
        n_augmentations_per_image = 5

        # Definir las transformaciones a aplicar a las imágenes
        augmentations = [
            iaa.Affine(rotate=(-25, 25)),
            iaa.Fliplr(),
            iaa.Affine(scale=(1.0, 1.3))
        ]

        # Iterar sobre cada imagen en el directorio de origen
        # img = Image.open(pathImagen)

        filename=os.path.basename(pathImagen)
        # Aplicar las transformaciones a la imagen
        augmented_images = []
        img = imageio.imread(pathImagen)
        for i in range(n_augmentations_per_image):
            aug = iaa.Sequential(augmentations)
            augmented = aug(image=img)
            augmented_images.append(augmented)

        # Guardar las nuevas imágenes en el directorio de destino
        for i, augmented in enumerate(augmented_images):
            output_path = os.path.join(output_dir, f"{filename[:-4]}_aug{i}.JPEG")
            print(output_path)
            imageio.imwrite(output_path,augmented)

    def imagenAugmentation(self,pathImagen,img):
        self.__rotationImagen(pathImagen,img)

    def imagenBrilloSaturation(self,pathImagen):
        img = cv2.imread(pathImagen)
        # Convertimos la imagen en el espacio de color HSV
        b, g, r = cv2.split(img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray = cv2.imread(pathImagen, cv2.IMREAD_GRAYSCALE)
        print(pathImagen)

        # print(clase)
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        lower_green = np.array([60, 50, 50])
        upper_green = np.array([90, 255, 255])
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([40, 255, 255])
        lower_red_dark = np.array([0,100,100])
        upper_red_dark = np.array([10,255,255])
        lower_red_light = np.array([170,100,100])
        upper_red_light = np.array([180,255,255])

        maskb = cv2.inRange(hsv, lower_blue, upper_blue)
        maskg = cv2.inRange(hsv, lower_green, upper_green)
        masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
        maskrd = cv2.inRange(hsv, lower_red_dark, upper_red_dark)
        maskrl = cv2.inRange(hsv, lower_red_light, upper_red_light)

        mask_meanb  = np.mean(maskb)
        mask_meang  = np.mean(maskg)
        mask_meany  = np.mean(masky)
        mask_meanrd = np.mean(maskrd)
        mask_meanrl = np.mean(maskrl)
        mask_stdb   = np.std(maskb)
        mask_stdg   = np.std(maskg)
        mask_stdy   = np.std(masky)
        mask_stdrd  = np.std(maskrd)
        mask_stdrl  = np.std(maskrl)
        
        # Separamos los canales de color HSV
        h, s, v = cv2.split(hsv)
        l, bb, a = cv2.split(lab)
        # Calculamos la saturación y el brillo medios de la imagen
        gray_mean =np.mean(gray)
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)
        l_mean = np.mean(l)
        a_mean = np.mean(a)
        bb_mean = np.mean(bb)

        gray_std = np.std(gray)
        b_std = np.std(b)
        g_std = np.std(g)
        r_std = np.std(r)
        h_std = np.std(h)
        s_std = np.std(s)
        v_std = np.std(v)
        l_std = np.std(l)
        a_std = np.std(a)
        bb_std = np.std(bb)

        
        gray_moments = cv2.moments(gray)
        gray_hu_moments = cv2.HuMoments(gray_moments)        
        gray_m1=gray_hu_moments[0][0]
        gray_m2=gray_hu_moments[1][0]
        gray_m3=gray_hu_moments[2][0]
        gray_m4=gray_hu_moments[3][0]
        gray_m5=gray_hu_moments[4][0]
        gray_m6=gray_hu_moments[5][0]
        gray_m7=gray_hu_moments[6][0]

        b_moments = cv2.moments(b)
        b_hu_moments = cv2.HuMoments(b_moments)
        b_m1=b_hu_moments[0][0]
        b_m2=b_hu_moments[1][0]
        b_m3=b_hu_moments[2][0]
        b_m4=b_hu_moments[3][0]
        b_m5=b_hu_moments[4][0]
        b_m6=b_hu_moments[5][0]
        b_m7=b_hu_moments[6][0]

        g_moments = cv2.moments(g)
        g_hu_moments = cv2.HuMoments(g_moments)
        g_m1=g_hu_moments[0][0]
        g_m2=g_hu_moments[1][0]
        g_m3=g_hu_moments[2][0]
        g_m4=g_hu_moments[3][0]
        g_m5=g_hu_moments[4][0]
        g_m6=g_hu_moments[5][0]
        g_m7=g_hu_moments[6][0]

        r_moments = cv2.moments(r)
        r_hu_moments = cv2.HuMoments(r_moments)
        r_m1=r_hu_moments[0][0]
        r_m2=r_hu_moments[1][0]
        r_m3=r_hu_moments[2][0]
        r_m4=r_hu_moments[3][0]
        r_m5=r_hu_moments[4][0]
        r_m6=r_hu_moments[5][0]
        r_m7=r_hu_moments[6][0]

        h_moments = cv2.moments(h)
        h_hu_moments = cv2.HuMoments(h_moments)
        h_m1=h_hu_moments[0][0]
        h_m2=h_hu_moments[1][0]
        h_m3=h_hu_moments[2][0]
        h_m4=h_hu_moments[3][0]
        h_m5=h_hu_moments[4][0]
        h_m6=h_hu_moments[5][0]
        h_m7=h_hu_moments[6][0]

        s_moments = cv2.moments(s)
        s_hu_moments = cv2.HuMoments(s_moments)
        s_m1=s_hu_moments[0][0]
        s_m2=s_hu_moments[1][0]
        s_m3=s_hu_moments[2][0]
        s_m4=s_hu_moments[3][0]
        s_m5=s_hu_moments[4][0]
        s_m6=s_hu_moments[5][0]
        s_m7=s_hu_moments[6][0]

        v_moments = cv2.moments(v)
        v_hu_moments = cv2.HuMoments(v_moments)
        v_m1=v_hu_moments[0][0]
        v_m2=v_hu_moments[1][0]
        v_m3=v_hu_moments[2][0]
        v_m4=v_hu_moments[3][0]
        v_m5=v_hu_moments[4][0]
        v_m6=v_hu_moments[5][0]
        v_m7=v_hu_moments[6][0]

        v_moments = cv2.moments(v)
        v_hu_moments = cv2.HuMoments(v_moments)
        v_m1=v_hu_moments[0][0]
        v_m2=v_hu_moments[1][0]
        v_m3=v_hu_moments[2][0]
        v_m4=v_hu_moments[3][0]
        v_m5=v_hu_moments[4][0]
        v_m6=v_hu_moments[5][0]
        v_m7=v_hu_moments[6][0]

        l_moments = cv2.moments(l)
        l_hu_moments = cv2.HuMoments(l_moments)
        l_m1=l_hu_moments[0][0]
        l_m2=l_hu_moments[1][0]
        l_m3=l_hu_moments[2][0]
        l_m4=l_hu_moments[3][0]
        l_m5=l_hu_moments[4][0]
        l_m6=l_hu_moments[5][0]
        l_m7=l_hu_moments[6][0]

        a_moments = cv2.moments(a)
        a_hu_moments = cv2.HuMoments(a_moments)
        a_m1=a_hu_moments[0][0]
        a_m2=a_hu_moments[1][0]
        a_m3=a_hu_moments[2][0]
        a_m4=a_hu_moments[3][0]
        a_m5=a_hu_moments[4][0]
        a_m6=a_hu_moments[5][0]
        a_m7=a_hu_moments[6][0]

        bb_moments = cv2.moments(bb)
        bb_hu_moments = cv2.HuMoments(bb_moments)
        bb_m1=bb_hu_moments[0][0]
        bb_m2=bb_hu_moments[1][0]
        bb_m3=bb_hu_moments[2][0]
        bb_m4=bb_hu_moments[3][0]
        bb_m5=bb_hu_moments[4][0]
        bb_m6=bb_hu_moments[5][0]
        bb_m7=bb_hu_moments[6][0]


        # edged = cv2.Canny(img, 30, 200)
        # contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Calcular área
        # area = cv2.contourArea(contours[0])

        # # Calcular perímetro
        # perimeter = cv2.arcLength(contours[0], True)

        # # Calcular momentos
        # M = cv2.moments(contours[0])
        # # cx = int(M["m10"] / M["m00"])
        # # cy = int(M["m01"] / M["m00"])
        # Calcular los momentos de Hu
        # moments = cv2.moments(gray)
        # hu_moments = cv2.HuMoments(moments)
        # # print(contours[0])
        # print(f'contornos: {len(contours)}')
        # print(f'area: {area}')
        # print(f'perimetro: {perimeter}')
        # print(f'cx: {cx}')
        # print(f'cy: {cy}')

        # Mostrar los momentos de Hu
        # print("Momentos invariantes de Hu:")
        # for i in range(0,7):
        #     print("Hu Momento #", (i+1), ": ", hu_moments[i][0])

        # Guardar la imagen resultante
        # filename=os.path.basename(pathImagen)
        # con=cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        # cv2.imwrite(f'imagen_dividida-{filename}.png', con)
        # cv2.imwrite('imagen_divididal.png', l)
        # cv2.imwrite('imagen_divididaa.png', a)
        # cv2.imwrite('imagen_divididab.png', b)
        
        return h_mean,s_mean,v_mean,mask_meanb, mask_meang, mask_meany, mask_meanrd, mask_meanrl, mask_stdb, mask_stdg, mask_stdy, mask_stdrd, mask_stdrl
        # return gray_mean,gray_std,gray_m1,gray_m2,gray_m3,gray_m4,gray_m5,gray_m6,gray_m7,b_mean,b_std,b_m1,b_m2,b_m3,b_m4,b_m5,b_m6,b_m7,g_mean,g_std,g_m1,g_m2,g_m3,g_m4,g_m5,g_m6,g_m7,r_mean,r_std,r_m1,r_m2,r_m3,r_m4,r_m5,r_m6,r_m7,h_mean,h_std,h_m1,h_m2,h_m3,h_m4,h_m5,h_m6,h_m7,s_mean,s_std,s_m1,s_m2,s_m3,s_m4,s_m5,s_m6,s_m7,v_mean,v_std,v_m1,v_m2,v_m3,v_m4,v_m5,v_m6,v_m7,l_mean,l_std,l_m1,l_m2,l_m3,l_m4,l_m5,l_m6,l_m7,a_mean,a_std,a_m1,a_m2,a_m3,a_m4,a_m5,a_m6,a_m7,bb_mean,bb_std,bb_m1,bb_m2,bb_m3,bb_m4,bb_m5,bb_m6,bb_m7






    def getInfo(self):  # sourcery skip: avoid-builtin-shadow
        # df = pd.DataFrame(columns=['imagen_path', 'width', 'height', 'formato', 'modo', 'bits'])
        for imagen in self.listImages:           
            carpetas_split=os.path.dirname(imagen).split('/')
            clase=carpetas_split[-1][1:2]
            clase='medicamento'+clase
            # Carga la imagen en un objeto Image
            img = Image.open(imagen)

            # Obtiene el tamaño de la imagen
            width, height = img.size
            # Obtiene el formato de la imagen
            formato = img.format
            # Obtiene el modo de color de la imagen
            modo = img.mode
            # Obtiene la profundidad de bits de la imagen
            bits = img.bits
            h_mean,s_mean,v_mean,mask_meanb, mask_meang, mask_meany, mask_meanrd, mask_meanrl, mask_stdb, mask_stdg, mask_stdy, mask_stdrd, mask_stdrl = self.imagenBrilloSaturation(imagen)
            new_row=pd.DataFrame({'imagen_path': imagen,'clase':clase, 'h_mean': h_mean, 's_mean':s_mean,  'v_mean':v_mean ,'mask_meanb':mask_meanb,'mask_meang':mask_meang,'mask_meany':mask_meany,'mask_meanrd':mask_meanrd,'mask_meanrl':mask_meanrl,'mask_stdb':mask_stdb,'mask_stdg':mask_stdg,'mask_stdy':mask_stdy,'mask_stdrd':mask_stdrd,'mask_stdrl':mask_stdrl}, index=[0])
            # gray_mean,gray_std,gray_m1,gray_m2,gray_m3,gray_m4,gray_m5,gray_m6,gray_m7,b_mean,b_std,b_m1,b_m2,b_m3,b_m4,b_m5,b_m6,b_m7,g_mean,g_std,g_m1,g_m2,g_m3,g_m4,g_m5,g_m6,g_m7,r_mean,r_std,r_m1,r_m2,r_m3,r_m4,r_m5,r_m6,r_m7,h_mean,h_std,h_m1,h_m2,h_m3,h_m4,h_m5,h_m6,h_m7,s_mean,s_std,s_m1,s_m2,s_m3,s_m4,s_m5,s_m6,s_m7,v_mean,v_std,v_m1,v_m2,v_m3,v_m4,v_m5,v_m6,v_m7,l_mean,l_std,l_m1,l_m2,l_m3,l_m4,l_m5,l_m6,l_m7,a_mean,a_std,a_m1,a_m2,a_m3,a_m4,a_m5,a_m6,a_m7,bb_mean,bb_std,bb_m1,bb_m2,bb_m3,bb_m4,bb_m5,bb_m6,bb_m7=self.imagenBrilloSaturation(imagen)
            # new_row = pd.DataFrame({'imagen_path': imagen, 'width': width, 'height': height,'formato': formato, 'modo': modo, 'bits': bits ,'gray_mean':gray_mean,'gray_std':gray_std,'gray_m1':gray_m1,'gray_m2':gray_m2,'gray_m3':gray_m3,'gray_m4':gray_m4,'gray_m5':gray_m5,'gray_m6':gray_m6,'gray_m7':gray_m7,'b_mean' :b_mean,'b_std':b_std,'b_m1':b_m1,'b_m2':b_m2,'b_m3':b_m3,'b_m4':b_m4,'b_m5':b_m5,'b_m6':b_m6,'b_m7':b_m7,'g_mean' :g_mean,'g_std':g_std,'g_m1':g_m1,'g_m2':g_m2,'g_m3':g_m3,'g_m4':g_m4,'g_m5':g_m5,'g_m6':g_m6,'g_m7':g_m7,'r_mean' :r_mean,'r_std':r_std,'r_m1':r_m1,'r_m2':r_m2,'r_m3':r_m3,'r_m4':r_m4,'r_m5':r_m5,'r_m6':r_m6,'r_m7':r_m7,'h_mean' :h_mean,'h_std':h_std,'h_m1':h_m1,'h_m2':h_m2,'h_m3':h_m3,'h_m4':h_m4,'h_m5':h_m5,'h_m6':h_m6,'h_m7':h_m7,'s_mean' :s_mean,'s_std':s_std,'s_m1':s_m1,'s_m2':s_m2,'s_m3':s_m3,'s_m4':s_m4,'s_m5':s_m5,'s_m6':s_m6,'s_m7':s_m7,'v_mean' :v_mean,'v_std':v_std,'v_m1':v_m1,'v_m2':v_m2,'v_m3':v_m3,'v_m4':v_m4,'v_m5':v_m5,'v_m6':v_m6,'v_m7':v_m7,'l_mean' :l_mean,'l_std':l_std,'l_m1':l_m1,'l_m2':l_m2,'l_m3':l_m3,'l_m4':l_m4,'l_m5':l_m5,'l_m6':l_m6,'l_m7':l_m7,'a_mean' :a_mean,'a_std':a_std,'a_m1':a_m1,'a_m2':a_m2,'a_m3':a_m3,'a_m4':a_m4,'a_m5':a_m5,'a_m6':a_m6,'a_m7':a_m7,'bb_mean':bb_mean,'bb_std':bb_std,'bb_m1':bb_m1,'bb_m2':bb_m2,'bb_m3':bb_m3,'bb_m4':bb_m4,'bb_m5':bb_m5,'bb_m6':bb_m6,'bb_m7':bb_m7}, index=[0])
            # print(new_row)
            # self.df.append(new_row, ignore_index=True)
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            # print(imagen)
            # self.__resizeImagen(imagen,img)
            # self.imagenAugmentation(imagen,img)

            # print(imagen,width,height,formato,modo,bits,sep="---")

            # Obtiene la información de los metadatos de la imagen
            # metadata = img.info
            # # print('Metadatos:', metadata)

            # metadata2 = img.getexif()
            # # Imprime la metadata
            # print(metadata2)
            # exif_table={}
            # for k, v in img.getexif().items():
            #     tag=TAGS.get(k)
            #     exif_table[tag]=v
            # print(exif_table)
            
            # # Busca la marca y modelo de la cámara en la metadata
            # camera_make = None
            # camera_model = None
            # if metadata:
            #     # Tag 271 indica la marca de la cámara
            #     camera_make = metadata2.get(271)
            #     # Tag 272 indica el modelo de la cámara
            #     camera_model = metadata2.get(272)

            # # Imprime la marca y modelo de la cámara
            # print('Marca de la cámara:', camera_make)
            # print('Modelo de la cámara:', camera_model)            
            print("-"*20)
        # print(self.df)

def minmax_norm(df_input):
    return (dft - dft.min()) / ( dft.max() - dft.min())

def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)


if __name__=='__main__':
    # x = Dataset('../dataset')
    x = Dataset('/Users/chocoplot/Documents/codeLAB/signal_Recognition/python/Proyecto/datasets/Dataset_Final_3Clases_Test')
    # x = Dataset('/Users/chocoplot/Documents/codeLAB/signal_Recognition/python/Proyecto/datasets/DatasetClass3')
    # x = Dataset('/Users/chocoplot/Documents/codeLAB/signal_Recognition/python/Proyecto/datasets/Dataset_2')
    # tree('/Users/chocoplot/Documents/codeLAB/signal_Recognition/python/Proyecto/dataset')
    x.treeFolders()
    x.treeFiles(x.path)
    # y=x.foldersDataset
    z=x.filesDataset[:]    
    i=Imagen(z)
    i.getInfo()
    print(len(z))
    print(z is i.listImages)
    dft=i.df.drop(['imagen_path','clase'],axis=1)
    dft=dft.drop(['mask_meanrl', 's_mean', 'mask_meang', 'v_mean', 'mask_meanrd', 'mask_stdb', 'mask_stdg', 'mask_stdy', 'mask_stdrd', 'mask_stdrl'],axis=1)
    # dft=dft.drop(['mask_meany','mask_meang','mask_meanrd','mask_meanrl','mask_meanb', 'mask_stdb', 'mask_stdg', 'mask_stdy', 'mask_stdrd', 'mask_stdrl'],axis=1)
    print(dft.head())
    # dft = dft.replace(0, 0.01)
    # dft = minmax_norm(dft)
    # dft = mean_norm(dft)
    print(dft.head())
    # max_position = dft.idxmax(axis=1)
    # dft['Max_Position'] = max_position
    # dft.loc[dft['Max_Position'] == "mask_meany", 'Max_Position'] = 1
    # dft.loc[dft['Max_Position'] == "mask_meang", 'Max_Position'] = 2
    # dft.loc[dft['Max_Position'] == "mask_meanrd", 'Max_Position'] = 3
    # dft=dft.drop(['mask_meany','mask_meang','mask_meanrd'],axis=1)
    # dft=dft.sample(n=40)
    print(dft.head())
    dft=dft.transpose()
    print(list(dft.columns))
    dft.to_csv('Fase3_Test_InputT.csv', index=False)
    targets=i.df[['clase']]
    targets=targets.transpose()
    targets.to_csv('Fase3_Test_TargetsT.csv', index=False)
    dft=i.df.drop(['imagen_path','clase'],axis=1)
    dft=dft.drop(['mask_meanrl', 's_mean', 'mask_meang', 'v_mean', 'mask_meanrd', 'mask_stdb', 'mask_stdg', 'mask_stdy', 'mask_stdrd', 'mask_stdrl'],axis=1)
    # dft=dft.drop(['mask_meany','mask_meang','mask_meanrd','mask_meanrl','mask_meanb', 'mask_stdb', 'mask_stdg', 'mask_stdy', 'mask_stdrd', 'mask_stdrl'],axis=1)
    # dft=dft.transpose()
    dft = dft.replace(0, 0.01)
    # dft = minmax_norm(dft)
    # h_mean,s_mean,v_mean
    # dft = mean_norm(dft)
    # covariance_matrix1 = dft.loc[:99].cov()
    # covariance_matrix2 = dft.loc[100:199].cov()
    # covariance_matrix3 = dft.loc[200:299].cov()
    # print(covariance_matrix1)
    # print(covariance_matrix2)
    # print(covariance_matrix3)
    # print(dft.info())
    # dfth = pd.DataFrame({'h_mean': range(300)})
    # print(dfth)
    # Calcular el promedio de las filas de la 100 a la 200 de la columna 'A'
    # promedio1 = dft.loc[:99, 'v_mean'].mean()
    # promedio2 = dft.loc[100:199, 'v_mean'].mean()
    # promedio3= dft.loc[200:299, 'v_mean'].mean()
    # print(promedio1,promedio2,promedio3, sep=";")
    # max_position = dft.idxmax(axis=1)
    # dft['Max_Position'] = max_position
    # dft.loc[dft['Max_Position'] == "mask_meany", 'Max_Position'] = 1
    # dft.loc[dft['Max_Position'] == "mask_meang", 'Max_Position'] = 2
    # dft.loc[dft['Max_Position'] == "mask_meanrd", 'Max_Position'] = 3
    # dft=dft.drop(['mask_meany','mask_meang','mask_meanrd'],axis=1)
    # dft=dft.sample(n=40)
    print(list(dft.columns))
    dft.to_csv('Fase3_Test_Input.csv', index=False)
    targets=i.df[['clase']]
    # targets=targets.transpose()
    targets.to_csv('Fase3_Test_Targets.csv', index=False)
    # i.df.to_csv('resumenInfoDatasetFinaltr.csv', index=False,header=False)

    # sample_image = cv2.imread('../dataset/Class_01/IMG_0558_1.JPEG')
    # img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,(256,256))

    # plt.axis('off');
    # plt.imshow(img)



