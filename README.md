# StarTrackerNavigation

Implements image processing techniques to process input video
Utilises Ultralytic YOLOv5 model to create a model trained on identifying constellations at different rotations and elevations
Identified constellations have the 2D image coordinates of the stars extracted and matched with 3D coordinates from data
Utilises PnP algorithm to calculate camera coordinates based on 2D and 3D coordinates
Coordinates and predictes coordinates are visualised along with camera pose
