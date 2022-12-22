# Track 3D-Objects Over Time

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?


### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 


### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?


### 4. Can you think of ways to improve your tracking results in the future?

This is a writeup submission for sensor fusion course  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : 3D Object Detection (Midterm). 

## 3D Object detection

We have used the [Waymo Open Dataset's](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) real-world data and used 3d point cloud for lidar based object detection. 

The steps included were -

- Map the ranges channel to 8 bit and view the range /intensity image (ID_S1_EX1)
- Use the Open3D library to display a 3d rendering of 10 lidar point clouds (ID_S1_EX2)
- Create Birds Eye View perspective (BEV) of the point cloud, assign lidar intensity values to BEV, normalize the heightmap of each BEV (ID_S2_EX1,ID_S2_EX2,ID_S2_EX3)
- Use [YOLO repository](https://review.udacity.com/github.com/maudzung/SFA3D) and add parameters to setup fpn resnet model(ID_S3_EX1)
- Convert BEV coordinates into pixel coordinates and convert model output to generate bounding boxes (ID_S3_EX2)
- Compute intersection over union, treat detections as objects if IOU exceeds threshold (ID_S4_EX1)
- Compute false positives and false negatives, precision and recall(ID_S4_EX2,ID_S4_EX3)


The project can be run by executing

```
python loop_over_dataset.py
```

All training was done in the workspace provided by Udacity


## Step-1: Compute Lidar point cloud from Range Image

In this step, we first previewing the range image and then convert range and intensity channels to 8 bit format. Next, we use the openCV library to stack the range and intensity channel vertically to visualize the image.

- Convert "range" channel to 8 bit
- Convert "intensity" channel to 8 bit
- Stack up range and intensity channels vertically in openCV

```
# step 1 : extract lidar data and range image for the roof-mounted lidar
lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]

# step 2 : extract the range and the intensity channel from the range image
ri = dataset_pb2.MatrixFloat()
ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
ri = np.array(ri.data).reshape(ri.shape.dims)

# step 3 : set values <0 to zero
ri[ri<0]=0.0

# step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
ri_range = ri[:,:,0]
ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range)) 
img_range = ri_range.astype(np.uint8)

# step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
ri_intensity = ri[:,:,1]
percentile_1, percentile_99 = percentile(ri_intensity,1), percentile(ri_intensity,99)
ri_intensity = 255 * np.clip(ri_intensity,percentile_1,percentile_99)/percentile_99 
img_intensity = ri_intensity.astype(np.uint8)

# step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
img_range_intensity = np.vstack((img_range,img_intensity))
img_range_intensity = img_range_intensity.astype(np.uint8)
```

Using the following settings
```
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
show_only_frames = [0, 1]
exec_data = []
exec_detection = []
exec_tracking = []
exec_visualization = ['show_range_image']
```


