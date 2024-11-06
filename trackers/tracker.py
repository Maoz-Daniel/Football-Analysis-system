
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import numpy as np
import pandas as pd
import cv2
sys.path.append('../')
from utils import get_width_of_bbox, get_center_of_bbox, get_foot_position

class Tracker:
    def __init__(self, model_path): # model_path is the path to the YOLO model
        self.model = YOLO(model_path) # load the model
        self.tracker=sv.ByteTrack() # initialize the tracker
         
    def add_position_to_track(self, tracks):

        for object, object_tracks in tracks.items():
            if object == 'ball':
                for frame_num, track in enumerate(object_tracks):
                    track['position'] = None  # אתחול של 'position' לכדורים
            else:  # שחקנים ושופטים (שיהיו מילונים)
                for frame_num, track in enumerate(object_tracks):
                    for track_id, track_info in track.items():
                        track_info['position'] = None  # אתחול של 'position' לשחקנים ושופטים

        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_data in track.items():
                    if isinstance(track_data, dict):
                        bbox=track_data['bbox']
                    if isinstance(track_data, list):
                        bbox=track_data
                       

                    if object== 'ball':
                        position=get_center_of_bbox(bbox) # get the center of the bounding box
                    else:
                        position=get_foot_position(bbox) # get the foot position of the bounding box

                    if object== 'players' or object== 'referees':
                      tracks[object][frame_num][track_id]['position']=position # add the position to the track
                    if object== 'ball':
                        tracks[object][frame_num]['position']=position # add the position to the track
                        
               

    def interpolate_positions(self,ball_positions):
        # Convert the list into a list of bbox values or empty lists
        positions = [x.get('bbox', []) if isinstance(x, dict) and 'bbox' in x else [] for x in ball_positions]

        # Create a DataFrame from the list of positions
        df_positions = pd.DataFrame(positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate the positions
        df_positions = df_positions.interpolate(method='linear')
        df_positions = df_positions.bfill()

        # Convert the DataFrame back to the original format
        result = []
        for pos in df_positions.values.tolist():
            result.append({'bbox': pos})

        # Return the result in the original format
        return result
    def detect_frames(self, frames): 
        
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1) # get the detections
            detections+=detections_batch # append the detections to the list
            #break
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):


        if read_from_stub and stub_path is not None and os.path.exists(stub_path): #if the tracks are saved in a file, read from the file
            with open(stub_path, 'rb') as f:
               tracks= pickle.load(f) # load the tracks and return them
            return tracks
            
        detections=self.detect_frames(frames) # get the detections

        tracks={ "players": [], "referees": [], "ball": []} # initialize the tracks
        

        for frame_num, detection in enumerate(detections):
            class_names=detection.names # get the class names 
            class_names_inv={v: k for k, v in class_names.items()} # get the class names in the required format

            #convert the detections to the format required by the tracker
            detection_supervision=sv.Detections.from_ultralytics(detection)
            

            #convert goalkeeper to player because goalkeeper is the same for us as player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id]=="goalkeeper":
                    detection_supervision.class_id[object_ind]=class_names_inv["player"]
            
            #track the objects
            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({}) 
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox=frame_detection[0].tolist() # get the bounding box
                class_id=frame_detection[3] # get the class id
                track_id=frame_detection[4] # get the track id

                if class_id==class_names_inv['player']:
                    tracks["players"][frame_num][track_id]={ "bbox": bbox }

                if class_id==class_names_inv['referee']:
                    tracks["referees"][frame_num][track_id]={ "bbox": bbox }

            for frame_detection in detection_supervision:
                bbox=frame_detection[0].tolist() 
                class_id=frame_detection[3]

                if class_id==class_names_inv['ball']:
                    tracks["ball"][frame_num][1]={ "bbox": bbox } # the ball is always 1
                    #print that ensure we detect the ball, print the data of the ball
                    print(f"frame_num: {frame_num}, bbox: {bbox}, class_id: {class_id}, class_name: {class_names[class_id]}")


                    

        if stub_path is not None: # save the tracks to a file
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f) 
                pickle.dump(tracks, f) 

        
        return tracks
        

            #print(detection_with_tracks)
    def draw_ellipse(self,frame, bbox, color, track_id=None):
       y2= int(bbox[3]) # get the y coordinate of the bottom of the bounding box
       x_center, _=get_center_of_bbox(bbox) # get the center of the bounding box
       width=get_width_of_bbox(bbox) # get the width of the bounding box

       cv2.ellipse(frame,center=(x_center,y2), axes=(int(width),int(0.35*width)), angle=0.0, startAngle=-45, endAngle=235, color=color, thickness=2)

       rectangle_width=40
       rectangle_height=20
       x1_rectangle=x_center-rectangle_width//2
       x2_rectangle=x_center+rectangle_width//2
       y1_rectangle=(y2-rectangle_height//2)+15
       y2_rectangle=(y2+rectangle_height//2)+15

       if(track_id is not None):
           cv2.rectangle(frame, (int(x1_rectangle), int(y1_rectangle)), (int(x2_rectangle), int(y2_rectangle)), color, cv2.FILLED)
           x1_text=x1_rectangle+12
           if track_id>99:
                x1_text-=10

           cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rectangle+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        

       return frame


    def draw_triangle(self,frame, bbox, color):
        y= int(bbox[1]) # get the y coordinate of the top of the bounding box
        x,_=get_center_of_bbox(bbox)
        triangle_points=np.array([[[x,y],[x-10,y-20],[x+10,y-20]]], np.int32) # get the points of the triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) # draw the triangle
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2) # draw the triangle
        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        #draw a semi transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255, 255, 255), -1) 
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame=team_ball_control[:frame_num+1] # get the team ball control till the current frame

        #get the number of time each team had the ball
        team1_num_frame=team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0] # get the number of frames team 1 had the ball
        team2_num_frame=team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0] # get the number of frames team 2 had the ball

        team1=team1_num_frame/(team1_num_frame+team2_num_frame) # get the percentage of time team 1 had the ball
        team2=team2_num_frame/(team1_num_frame+team2_num_frame) # get the percentage of time team 2 had the ball

        cv2.putText(frame, f"Team 1 Ball Control: {team1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3) # draw the percentage of time team 1 had the ball
        cv2.putText(frame, f"Team 2 Ball Control: {team2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3) # draw the percentage of time team 2 had the ball

        return frame



    def draw_annotations(self, video_frames, tracks, team_ball_control):

        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # copy the frame to avoid modifying the original frame

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num] 

            # draw players
            for track_id, player in player_dict.items():
                color=player.get("team color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)  # draw the player
                
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))  # draw the ball

            #draw referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255), track_id)  # draw the player  

            #draw ball
            
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball,(0,255,0))

            #draw team ball control
            frame=self.draw_team_ball_control(frame, frame_num, team_ball_control) 
                
            
            # After processing the frame, append it to output_frames
            output_frames.append(frame)  # Append the frame to the list of output frames

        return output_frames

            
        