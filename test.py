from ultralytics import YOLOv10

if __name__ == '__main__':
    model = YOLOv10(r'best.pt')
    model.val(data='yolo-bvn.yaml',
              split='test',
              batch=1,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )