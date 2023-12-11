#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def pc_sub():
    print("Came to sub func")
    rospy.Subscriber('/royale_cam0/point_cloud', PointCloud2, callback_fun, queue_size=1)
    rospy.spin()

def callback_fun(msg):
    points = np.array(list(pc2.read_points(msg, skip_nans=True)))

    # Convert numpy array to PCL PointCloud
    cloud = pcl.PCL.
    cloud.from_array(points.astype(np.float32))

    # Perform segmentation for a hemisphere
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_SPHERE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    inliers, coefficients = seg.segment()

    # Extract inliers (hemisphere) from the original cloud
    extracted_cloud = cloud.extract(inliers, negative=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(extracted_cloud.to_array()[:, 0], extracted_cloud.to_array()[:, 1], extracted_cloud.to_array()[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect((np.ptp(extracted_cloud.to_array()[:, 0])),
                      (np.ptp(extracted_cloud.to_array()[:, 1])),
                      (np.ptp(extracted_cloud.to_array()[:, 2])))

    ax.set_xlim(np.min(extracted_cloud.to_array()[:, 0]), np.max(extracted_cloud.to_array()[:, 0]))
    ax.set_ylim(np.min(extracted_cloud.to_array()[:, 1]), np.max(extracted_cloud.to_array()[:, 1]))
    ax.set_zlim(np.min(extracted_cloud.to_array()[:, 2]), np.max(extracted_cloud.to_array()[:, 2]))

    plt.title('Segmented Curve')
    plt.show()

if __name__ == '__main__':
    try:
        rospy.init_node("point_cloud_subscriber", anonymous=True)
        pc_sub()
    except rospy.ROSInterruptException:
        pass
