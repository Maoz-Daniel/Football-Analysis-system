# this file is going to have the utilities to read in the video and the the video
import cv2 # for reading in the video

def read_video(video_path): # this function is going to read in the video (code from copilot)
    capture = cv2.VideoCapture(video_path) # this is going to read in the video
    frames = [] 
    while True:
        ret, frame = capture.read() # read in the frame
        if not ret:
            break
        frames.append(frame) # append the frame to the frames list
    return frames

def save_video(output_video_frames, output_video_path): # this function is going to save the video
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # this is the codec
    out= cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1],output_video_frames[0].shape[0])) #the shape is the x and y of the frame 
    for frame in output_video_frames:
        out.write(frame)
    out.release() # release the video