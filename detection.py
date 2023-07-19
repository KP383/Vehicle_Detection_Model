import time
import cv2
import numpy as np
import threading

class Detection:

   def vehicle_detection(input_file, camara_id, lock):
      thread_id = threading.get_native_id()

      print(f"Camara Id : {camara_id}   Thread id : {thread_id} start!!!")
      start_time = time.time()
      output_file = f"{camara_id}_{thread_id}.avi"
      try:
         net = cv2.dnn.readNetFromDarknet("yolo_custom_tiny_LPD_v.cfg", "yolo_custom_tiny_LPD_v_best.weights")
         classes = ['Car', 'Bus', 'Motorbike', 'Cycle', 'Truck', 'Autorickshaw', 'Rickshaw', 'Van', 'Minitruck']

         lock.acquire()
         # print("point 1")
         video = cv2.VideoCapture(input_file)
         # print(video.get(cv2.CAP_PROP_FPS))
         lock.release()
         fourcc = cv2.VideoWriter_fourcc(*'XVID')
         output_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 320))
         frame_count = 0
         frame_rate = 10
         while video.isOpened():
            
            # print("point 2")
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1

            if frame_count % frame_rate == 0:

               image = cv2.resize(frame, (640, 320), interpolation=cv2.INTER_LANCZOS4)
               blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 320), swapRB=True, crop=False)

               net.setInput(blob)
               output_layers = net.getUnconnectedOutLayersNames()
               layer_outputs = net.forward(output_layers)

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

               # indices.sort()
               # print(f"{id}", indices)

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

               # Write processed frame to output video
               lock.acquire()
               output_video.write(image)
               lock.release()
               # print("point 3")
               # Display the processed frame
               lock.acquire()
               cv2.imshow('Object Detection', image)
               if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
               lock.release()
            
      except Exception as e:
         print(f"Exception occurred", e)
      finally:
        # Release resources
        video.release()
      #   output_video.release()
        cv2.destroyAllWindows()
   
      print(f"complete in {time.time() - start_time}")
      print(f"{camara_id}_{thread_id}.avi saved !!!")
