from ultralytics import YOLO
import torch
import cv2
import wandb
# from wandb.integration.ultralytics import add_wandb_callback

if __name__ == '__main__':
    wandb.login(key='445c46b947567ded092431c9f5908b737d9843e9')
    # wandb.init(project="yolo_training", name="yolov8l_n1")
    torch.cuda.empty_cache()
    model = YOLO('yolov8n.pt')
    print(model)
    # model.load('lpr_modify.pt')
    # add_wandb_callback(model, enable_model_checkpointing=True)
    model.train(data='datasets/lpr_dataset/data.yaml', epochs=100,
                batch=20, save=True, name='yolo8n_modified', project='yolo_training')
    model.val()
    # wandb.finish()
    model.export()
    # model = YOLO('runs/detect/train3/weights/best.pt')  # custom model n1
    # # custom model n2
    # model2 = YOLO('runs/detect/train4/weights/best.pt')  # custom model n2
    # # print(model)
    # img = cv2.imread('5.png')
    # img = cv2.resize(img, (640, 640))

    # results_1 = model.predict(source=img, conf=0.25)
    # results_2 = model2.predict(source=img, conf=0.25)
    # for r in results_1:
    #     # print(r.boxes)
    #     # print(r.boxes.cls)
    #     # print(r.boxes.xyxy)
    #     # print(r.boxes.conf)
    #     # print(r.names)
    #     annotated_frame = r.plot()
    #     cv2.imshow("annotated_frame", annotated_frame)
    # for r in results_2:
    #     # print(r.boxes)
    #     # print(r.boxes.cls)
    #     # print(r.boxes.xyxy)
    #     # print(r.boxes.conf)
    #     # print(r.names)
    #     annotated_frame = r.plot()
    #     cv2.imshow("annotated_frame2", annotated_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
