import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLOv10
 
if __name__ == '__main__':
    model = YOLOv10('ultralytics/cfg/models/v10/yolov10s-C2f_deformable_LKA+dysample.yaml')
    model.load('yolov10S.pt') # loading pretrain weights
    model.train(data='yolo-bvn.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=16,
                close_mosaic=10,
                device='0',
                optimizer='SGD', # using SGD
                project='runs/train',
                name='4s-C2f_deformable_LKA+dysample',
                )
 
 