import cv2
import mediapipe as mp
import argparse
import os

def process_image(img,face_detection):
    H,W,_=img.shape
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out=face_detection.process(img_rgb)
    
    if out.detections is not None:
        for detection in out.detections:
            locationdata=detection.location_data
            bbox=locationdata.relative_bounding_box

            x1,y1,w,h=bbox.xmin,bbox.ymin,bbox.width,bbox.height

            x1=int(x1*W)
            y1=int(y1*H)
            w=int(w*W)
            h=int(h*H)

            # blur faces
            img[y1:y1+h,x1:x1+w,:]=cv2.blur(img[y1:y1+h,x1:x1+w,:],(20,20))
    return img

output_dir='./output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
args=argparse.ArgumentParser()
args.add_argument("--mode",default='webcam')
args.add_argument("--filePath",default='Face Anonymizer/testVideo.mp4')
args=args.parse_args()

# read image
# detect faces
mp_face_detection=mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
    if args.mode in ["image"]:
        img=cv2.imread(args.filePath)
        
        img=process_image(img,face_detection)
        cv2.imwrite(os.path.join(output_dir,'output.png'),img)
    
    elif args.mode in ['video']:

        cap=cv2.VideoCapture(args.filePath)
        ret,frame=cap.read()

        output_video=cv2.VideoWriter(os.path.join(output_dir,'output.mp4'),
                                     cv2.VideoWriter_fourcc(*'MP4V'),
                                     25,
                                     (frame.shape[1],frame.shape[0]))

        while ret:
            frame=process_image(frame,face_detection)
            output_video.write(frame)
            ret,frame=cap.read()
            
        cap.release()
        output_video.release()
    
    elif args.mode in ['webcam']:

        cap=cv2.VideoCapture(0)
        ret,frame=cap.read()

        while ret:
            frame=process_image(frame,face_detection)
            cv2.imshow('frame',frame)
            cv2.waitKey(25)
            ret,frame=cap.read()

        cap.release()


