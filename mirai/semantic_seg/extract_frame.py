from moviepy.editor import VideoFileClip
import sys

t = sys.argv[1]
clip = VideoFileClip("./driving.mp4")
clip.save_frame("extracted-"+t+".jpg", t=float(t))
