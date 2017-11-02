import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from pcl_helper import *
from rospy_message_converter import message_converter

#from sklearn.neighbors.kde import KernelDensity


def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized


def compute_color_histograms(cloud, using_hsv=True):
    cloud = ros_to_pcl(cloud)
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.006
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud = vox.filter()
    cloud = pcl_to_ros(cloud)

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])
        

    # TODO: Compute histograms
	nbins = 32
	bins_range = (0, 256)
	ch_1 = np.histogram(channel_1_vals, bins=nbins, range=bins_range)
	ch_2 = np.histogram(channel_2_vals, bins=nbins, range=bins_range)
	ch_3 = np.histogram(channel_3_vals, bins=nbins, range=bins_range)

 #    # TODO: Concatenate and normalize the histograms
    ch_features = np.concatenate((ch_1[0], ch_2[0], ch_3[0])).astype(np.float64)    


    # Replace normed_features with your feature vector
    normed_features = ch_features / np.sum(ch_features) 
    return normed_features 


def compute_normal_histograms(normal_cloud):

    # Compute histograms for the clusters
    point_colors_list = []
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    #normal_cloud changed to normal_cloud_vox
    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])
       


    # TODO: Compute histograms of normal values (just like with color)
	nbins = 32
	#Note this range can be -1, 1 or 0, 256. -1, 1 seems to work better with bayes
	bins_range = (0, 256)
	x = np.histogram(norm_x_vals, bins=nbins, range=bins_range)
	y = np.histogram(norm_y_vals, bins=nbins, range=bins_range)
	z = np.histogram(norm_z_vals, bins=nbins, range=bins_range)

    # TODO: Concatenate and normalize the histograms
    xyz_features = np.concatenate((x[0], y[0], z[0])).astype(np.float64)

    # Replace normed_features with your feature vector
    normed_features = xyz_features / np.sum(xyz_features)

    return normed_features
