from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer    
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    # read in the video
    video_frames=read_video('input_videos/08fd33_4.mp4') 
    

    #initialize the tracker
    tracker=Tracker('models/best (1).pt')

    #tracks=tracker.get_object_tracks(video_frames)


    tracks=tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl') # get the tracks 

    # add the position to the tracks
    tracker.add_position_to_track(tracks) 

    #camera movement estimation
    camera_movement_estimator=CameraMovementEstimator(video_frames[0])
    camera_movement_per_frme=camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stubs.pkl') # get the camera movement

    # adjust the positions of the tracks based on the camera movement
    camera_movement_estimator.adjust_posotoins_to_tracks(tracks, camera_movement_per_frme) 

    #view transformation
    view_transformer=ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks) # add the transformed position to the tracks


    #interpolate ball positions using interpolate_positions method
    tracks['ball']=tracker.interpolate_positions(tracks['ball']) # interpolate the ball positions
   

    #speed and distance estimation
    speed_and_distance_estimator=SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks) # add the speed and distance to the tracks

    #assign player teams
    team_assigner=TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0]) # assign team colors
 
    for frame_number, player_truck in enumerate(tracks['players']):
        for player_id, track in player_truck.items():
            team=team_assigner.get_player_team(video_frames[frame_number], track['bbox'], player_id)
            tracks['players'][frame_number][player_id]['team']=team # assign the team to the player
            tracks['players'][frame_number][player_id]['team color']=team_assigner.team_colors[team] # assign the team to the player

    #assign ball to player
    player_ball_assigner=PlayerBallAssigner()
    team_ball_control= []
    for frame_number, player_truck in enumerate(tracks['players']):
        ball_bbox=tracks['ball'][frame_number]['bbox'] #!!!!!!!!!!
        assigned_player=player_ball_assigner.assign_ball_to_player(player_truck, ball_bbox)

        if assigned_player!=-1:
            tracks['players'][frame_number][assigned_player]['has_ball']=True # assign the ball to the player
            team_ball_control.append(tracks['players'][frame_number][assigned_player]['team']) 
        else:
            team_ball_control.append(team_ball_control[-1]) # if no player has the ball, assign the ball to the same team as the previous frame

    team_ball_control=np.array(team_ball_control)
        

    #draw output video
    ## draw object tracks
    output_video_frames=tracker.draw_annotations(video_frames, tracks,team_ball_control)

   ## draw camera movement
    output_video_frames=camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frme)

    # draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    # save the video
    save_video(output_video_frames, 'output_videos/output_video.avi') 

 

if __name__ == "__main__":
    main()