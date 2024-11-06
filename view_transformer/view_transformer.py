import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        court_width=68 #the real width of the court in meters
        court_length=23.32 #after a calculate we take the 3 first parts of the court height

        self.pixel_verticies=np.array( [[110,1035], [265,275], [910,260], [1640,915]]) # the pixel verticies of the court
        self.target_verticies = np.array([[0,court_width], [0,0], [court_length, 0], [court_length, court_width]]) # the target verticies of the court, after the transformation

        self.pixel_verticies=self.pixel_verticies.astype(np.float32) # convert the pixel verticies to float32
        self.target_verticies=self.target_verticies.astype(np.float32) # convert the target verticies to float32

        self.perspective_transformer= cv2.getPerspectiveTransform(self.pixel_verticies, self.target_verticies) # get the perspective transformer

    def transform_point(self,point):
        if isinstance(point, np.ndarray) and point.ndim == 1 and point.size == 2:       
         p = (int(point[0]),int(point[1]))
        else:
            return None
        is_inside = cv2.pointPolygonTest(self.pixel_verticies,p,False) >= 0 
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)
        return tranform_point.reshape(-1,2)

    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_data in track.items():
                     
                    if object== 'players' or object=='referees':
                        position = track_data['position_adjusted']
                        position = np.array(position)

                        if len (position)==2 or len (position)==4:
                            position_trasnformed = self.transform_point(position)
                            if position_trasnformed is not None:
                                position_trasnformed = position_trasnformed.squeeze().tolist()
                            tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed
                    
                    if object=='ball':
                        position = track_data
                        position = np.array(position)
                        if  position is not None:
                            position_trasnformed = self.transform_point(position)  
                            if position_trasnformed is not None:                    
                                 position_trasnformed = position_trasnformed.squeeze().tolist()
                        tracks[object][frame_num]['position_transformed'] = position_trasnformed
     