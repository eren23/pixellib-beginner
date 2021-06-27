from pixellib.instance import instance_segmentation
import cv2
from model import model_path

segmentation_model = instance_segmentation()

#Path for the model
segmentation_model.load_model(model_path)

cap = cv2.VideoCapture(2)

while cap.isOpened():

    success, frame = cap.read()

    seg = segmentation_model.segmentFrame(frame, show_bboxes=True)
    image = seg[1]


    cv2.imshow("Segmentation", image)
    
    if cv2.waitKey(1) &0xFF ==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

