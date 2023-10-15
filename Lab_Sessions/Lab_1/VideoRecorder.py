import cv2 as cv
import sys
import os
import threading
import queue
import sounddevice as sd
import soundfile as sf

class VideoRecorder(object):
    def __init__(self, video_file_name):   
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW )
        for i in range(0,10):
            (self.status, self.frame) = self.cap.read() # Warmup
        
        self.video_file_name = video_file_name + '.mp4'
        self.frame_width = 640
        self.frame_height = 480
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width);
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height); 
        
        self.codec = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        
        self.audio_thread = AudioRecorder()
        audiofilename = 'audio'
        if os.path.exists(audiofilename+'.wav'):
            os.remove(audiofilename+'.wav')
        
        initial = False

        frames = []
        while True:
            (self.status, self.frame) = self.cap.read()
            if initial == False:
                self.audio_thread.start(audiofilename, './')
                initial = True
            
            cv.imshow('Display', self.frame)
            frames.append(self.frame)

            # Press q to stop recording
            key = cv.waitKey(1)
            if key == ord('q'):
                cv.destroyWindow('Display')
                break
            
        self.cap.release()
        self.audio_thread.stop()
        
        self.output_video = cv.VideoWriter(self.video_file_name, self.codec, 30, (self.frame_width, self.frame_height))
        for frame in frames:
            self.output_video.write(frame)
        self.output_video.release()

        
        outputfilename = "output.mp4"
        cmd = f"ffmpeg -y -i {audiofilename}.wav -i videofile.mp4 -c:v copy -c:a aac -strict experimental {outputfilename}"
        os.system(cmd)
        
        sys.exit(0)
    
class AudioRecorder():

    def __init__(self):
        self.open = True
        self.channels = 1
        self.q = queue.Queue()
        device_info = sd.query_devices(0, 'input')
        self.samplerate = int(device_info['default_samplerate'])

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def record(self):
        with sf.SoundFile(self.file_name, mode='x', samplerate=self.samplerate,
                      channels=self.channels) as file:
            with sd.InputStream(samplerate=self.samplerate,
                                channels=self.channels, callback=self.callback):

                while(self.open == True):
                    file.write(self.q.get())

    def stop(self):
        self.open = False

    def start(self, file_name, file_dir):
        self.open = True
        self.file_name = '{}/{}.wav'.format(file_dir, file_name)
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()
        
if __name__ == '__main__':
    rec = VideoRecorder('videofile')