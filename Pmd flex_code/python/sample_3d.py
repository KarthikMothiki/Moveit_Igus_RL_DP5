#!/usr/bin/python3

# Copyright (C) 2020 pmdtechnologies ag
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
# PARTICULAR PURPOSE.

"""This sample shows how to visualize the 3D data.

It uses Open3D (http://www.open3d.org/) to display the point cloud.
"""

import argparse
try:
    from roypypack import roypy  # package installation
except ImportError:
    import roypy  # local installation
import time
import queue
from sample_camera_info import print_camera_info
from roypy_sample_utils import CameraOpener, add_camera_opener_options, select_use_case
from roypy_platform_utils import PlatformHelper

import numpy as np
import open3d as o3d

class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.queue = q
        self.figSetup = False
        self.firstTime = True
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def onNewData(self, data):
        pc = data.npoints ()
        
        #only select the three columns we're interested in
        px = pc[:,:,0]
        py = pc[:,:,1]
        pz = pc[:,:,2]
        stack1 = np.stack([px,py,pz], axis=-1)
        stack2 = stack1.reshape(-1, 3)
        
        self.queue.put(stack2)

    def paint (self, data):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """
        
        #filter out points at distance 0
        data = data[np.all(data != 0, axis=1)]
        
        vec3d = o3d.utility.Vector3dVector(data)
        
        # add the pointcloud on the first run
        if(self.firstTime):
        
            self.pointcloud = o3d.geometry.PointCloud(vec3d)
            self.vis.add_geometry(self.pointcloud)
            
            #rotate virtual camera to front of camera
            vc = self.vis.get_view_control()
            vc.set_front([0.,0.,-1.])
            
            self.firstTime = False
        
        self.pointcloud.points = vec3d

        result = self.vis.update_geometry(self.pointcloud)
        self.vis.poll_events()
        self.vis.update_renderer()

def main ():
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser (usage = __doc__)
    add_camera_opener_options (parser)
    parser.add_argument ("--minutes", type=int, default=60, help="duration to capture data in minutes")
    options = parser.parse_args()
    opener = CameraOpener (options)
    cam = opener.open_camera () 

    print_camera_info (cam)
    print("isConnected", cam.isConnected())
    print("getFrameRate", cam.getFrameRate())

    curUseCase = select_use_case(cam)

    try:
        # retrieve the interface that is available for recordings
        replay = cam.asReplay()
        print ("Using a recording")
        print ("Framecount : ", replay.frameCount())
        print ("File version : ", replay.getFileVersion())
    except SystemError:
        print ("Using a live camera")
    
    # we will use this queue to synchronize the callback with the main
    # thread, as drawing should happen in the main thread
    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)

    print ("Setting use case : " + curUseCase)
    cam.setUseCase(curUseCase)

    cam.startCapture()
    # create a loop that will run for the given amount of time (default 60 minutes)
    process_event_queue(q, l, options.minutes)
    cam.stopCapture()

def process_event_queue(q, painter, minutes_to_display):
    # Calculate the end time (current time + minutes_to_display)
    t_end = time.time() + minutes_to_display * 60

    # Use a flag to keep track of whether the user initiated a close event
    window_closed = False

    while time.time() < t_end and not window_closed:
        try:
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range(0, len(q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            break
        else:
            # Update the painter with the new item
            painter.paint(item)

        # Check if the window is still open (events are still being processed)
        window_closed = not painter.vis.poll_events()
        painter.vis.update_renderer()

    # Close the visualization window once the loop is done
    painter.vis.destroy_window()

if (__name__ == "__main__"):
    main()
