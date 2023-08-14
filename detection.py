import time
import cv2
import numpy as np
from argparse import ArgumentParser
from threading import Thread
import threading


class ThreadWithReturnValue_1(Thread):
    
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(self._args, **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
class ThreadWithReturnValue_2(Thread):
    
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(self._args[0], self._args[1], **self._kwargs)
    def join(self, *args): 
        Thread.join(self, *args)
        return self._return

# MODEL CLASS
class Model:
   
   def model(image):
      net = cv2.dnn.readNetFromDarknet("yolo_custom_tiny_LPD_v.cfg", "yolo_custom_tiny_LPD_v_best.weights")
      blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 320), swapRB=True, crop=False)

      net.setInput(blob)
      output_layers = net.getUnconnectedOutLayersNames()
      layer_outputs = net.forward(output_layers)

      return layer_outputs


# POSTPROCESS CLASS
class PostProcess:

   def postProcess(layer_outputs, image):
      classes = ['Car', 'Bus', 'Motorbike', 'Cycle', 'Truck', 'Autorickshaw', 'Rickshaw', 'Van', 'Minitruck']
      boxes = []
      confidences = []
      class_ids = []

      for output in layer_outputs:
         for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Adjust confidence threshold as needed
               center_x = int(detection[0] * image.shape[1])
               center_y = int(detection[1] * image.shape[0])
               width = int(detection[2] * image.shape[1])
               height = int(detection[3] * image.shape[0])
      
               x = int(center_x - width / 2)
               y = int(center_y - height / 2)
               # print(x, y, width, height, classes[class_id], confidence)
               boxes.append([x, y, width, height])
               confidences.append(float(confidence))
               class_ids.append(class_id)

               indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2)  

               for i in indices:
                  box = boxes[i]
                  x, y, w, h = box
                  class_id = class_ids[i]
                  label = classes[class_id]
                  confidence = confidences[i]
                  # print(label, " " , confidence)
                  # Draw bounding box and label on the image
                  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                  # text = f"{label}: {confidence:.2f}"
                  text = f"{label}"
                  cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      return image

class Detection :

   def vehicle_detection(input_file, camara_id, lock):
      thread_id = threading.get_native_id()

      print(f"Camara Id : {camara_id}   Thread id : {thread_id} start!!!")
      start_time = time.time()
      output_file = f"{camara_id}_{thread_id}.avi"
      try:
         lock.acquire()
         video = cv2.VideoCapture(input_file)
         lock.release()
         output_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 320))
         frame_count = 0
         frame_rate = 15
         while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % frame_rate == 0:

               image = cv2.resize(frame, (640, 320), interpolation=cv2.INTER_LANCZOS4)

               twrv_1 = ThreadWithReturnValue_1(target=Model.model, args=(image))
               twrv_1.start()
               layer_outputs = twrv_1.join()
               print("camara ", camara_id)
               twrv_2 = ThreadWithReturnValue_2(target=PostProcess.postProcess, args=(layer_outputs, image))
               twrv_2.start()
               image = twrv_2.join()

               # Write processed frame to output video
               lock.acquire()
               output_video.write(image)
               lock.release()
   
      except Exception as e:
         print(f"Exception occurred", e)
      finally:
        # Release resources
        video.release()
        output_video.release()
        cv2.destroyAllWindows()
   
      print(f"complete in {time.time() - start_time}")
      print(f"{camara_id}_{thread_id}.avi saved !!!")
