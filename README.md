# jmpeg
JPEG and MPEG Compression using Python

Requiere python>=3.0 y ffmpeg

Uso:

1) Para instalar las dependencias necesarias en su entorno de python: pip install -r requirements.txt
2) Para comprimir una secuencia de cuadros almacenada en un archivo 
npy o mat ejecute: python mpeg.py --ndrop 32 --ipath input_video.npy --opath output_video.avi
3) Para comprimir y ver la reconstrucion de un archivo .bmp
ejecute: python jpeg.py --ndrop 32 --ipath input_image.bmp --opath output_image.b
