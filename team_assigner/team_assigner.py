from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors={}
        self.player_team={} 


    def get_clustering_model(self,image):

        # change the image into 2d array
        image_2d=image.reshape(-1,3) # -1 means that the value is inferred from the length of the array and remaining dimensions, 3 is the number RGB values

        # perform kmeans clustering
        kmeans=KMeans(n_clusters=2, init="k-means++",n_init=1) # n_clusters is the number of clusters, init is the method of initialization, n_init is the number of times the algorithm will be run with different centroid seeds
        kmeans.fit(image_2d) # fit the model

        return kmeans


    def get_player_color(self,frame,bbox): #same code as in the notebook
        image=frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # get the image of the player

        top_half=image[0:int(image.shape[0]/2),:] # get the top half of the image

        #get cluster
        kmeans=self.get_clustering_model(top_half)

        #get clusters labels for each frame
        labels=kmeans.labels_

        #reshape the labels into the image shape
        clustered_image=labels.reshape(top_half.shape[0],top_half.shape[1]) # reshape the labels into the image shape

        corner_clusters=[clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]] # get the clusters of the corners
        non_player_cluster=max(set(corner_clusters), key=corner_clusters.count) # get the cluster that is not the player, the number that appears the most in the corners

        player_cluster=1-non_player_cluster # get the cluster that is the player

        player_color=kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame,player_detection):

        player_colors=[]

        for player_id, detection in player_detection.items():
         bbox = detection["bbox"]
         player_color = self.get_player_color(frame, bbox)
         player_colors.append(player_color)
         
        kmeans= KMeans(n_clusters=2, init="k-means++", n_init=10) # initialize the kmeans with the player colors 
        kmeans.fit(player_colors) # fit the model

        self.kmeans=kmeans # save the kmeans model

        self.team_colors[1]=kmeans.cluster_centers_[0] # get the first cluster center
        self.team_colors[2]=kmeans.cluster_centers_[1] # get the second cluster center

    def get_player_team(self,frame,player_bbox,player_id):

        if(player_id in self.player_team):
            return self.player_team[player_id]
        player_color=self.get_player_color(frame, player_bbox)

        team_id=self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id=team_id+1 # to be 1 or 2

        if player_id==116: # if the player is the goalkeeper ()
            team_id=1

        if player_id==238: # if the player is the goalkeeper ()
            team_id=2
        self.player_team[player_id]=team_id

        return team_id
        


        