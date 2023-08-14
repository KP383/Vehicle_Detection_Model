from flask import Flask
from detection import Detection
import threading

app = Flask("Model")

def create_thread(camara_id, video_file):
   lock = threading.Lock()
   thread_id = threading.get_native_id()
   thread = threading.Thread(target=Detection.vehicle_detection, args=(video_file, thread_id, camara_id, lock), daemon=True)
   thread.start()
   thread.join()
   return thread_id

@app.route('/camara/<id>/<video_file>')
def start_new_instance(id, video_file):
   lock = threading.Lock()
   thread = threading.Thread(target=Detection.vehicle_detection, args=(video_file, id, lock), daemon=True)
   thread.start()
   thread.join()
   return (f"Camara Id : {id}   Thread id : {thread} complete!!!")



if __name__ == '__main__':
	app.run('127.0.0.1', 5000)

