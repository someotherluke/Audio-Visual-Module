# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 09:03:12 2023

@author: yad23rju
"""

import cv2 as cv
import sys
import os
import threading
import queue
import sounddevice as sd
import soundfile as sf

import random as rnd

class VideoRecorder(object):
    def __init__(self, video_file_name, output_name):

        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)

        for i in range(0, 10):
            (self.status, self.frame) = self.cap.read()  # Warmup

        self.video_file_name = video_file_name + '.mp4'
        self.frame_width = 640
        self.frame_height = 480
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width);
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height);

        self.codec = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

        self.audio_thread = AudioRecorder()
        audiofilename = output_name
        if os.path.exists(audiofilename + '.wav'):
            os.remove(audiofilename + '.wav')

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
                self.cap.release()
                self.audio_thread.stop()
                break



        self.output_video = cv.VideoWriter(self.video_file_name, self.codec, 30, (self.frame_width, self.frame_height))
        for frame in frames:
            self.output_video.write(frame)
        self.output_video.release()

        self.output_name = output_name + ".mp4"
        cmd = f"ffmpeg -y -i {audiofilename}.wav -i {self.video_file_name} -c:v copy -c:a aac -strict experimental {self.output_name}"
        os.system(cmd)

        #sys.exit(0)


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
                while (self.open == True):
                    file.write(self.q.get())

    def stop(self):
        self.open = False

    def start(self, file_name, file_dir):
        self.open = True
        self.file_name = '{}/{}.wav'.format(file_dir, file_name)
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()


def record_audio():
    speaker = 'D'
    #SETS THE NUMBER OF RECODINGS
    recording_number = 1
    name_list = ["Ben", "Charlie", "Chiedozie", "Carlos","El","Ethan",
                 "Francesca","Jack","Jake","James","Lindon","Marc","Nischal",
                 "Robin","Ryan","Sam","Seth","William","Bonney","Yubo"]
    rnd.shuffle(name_list)
    for name in name_list:

        for file_number in range(0,recording_number):
            file_number = str(file_number)
            #Separate with underscores so it's easy to split
            file_name = name + '_' + speaker + '_' + file_number # Will the order they're in matter?
            print(file_name)
            rec = VideoRecorder(file_name, file_name)
            #TODO: QUESTIONS: BEST RESOLUTION? Using 1920 by 1080 makes it really laggy
            
if __name__ == '__main__':
    record_audio()