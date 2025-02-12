#!/usr/bin/python3

# Copyright (C) 2023 Infineon Technologies & pmdtechnologies ag
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
# PARTICULAR PURPOSE.

"""This sample shows how to export an rrf file to PLY format.
"""

import argparse
try:
    from roypypack import roypy  # package installation
except ImportError:
    import roypy  # local installation
import time
import queue
import sys

from roypy_sample_utils import CameraOpener, add_camera_opener_options
from roypy_platform_utils import PlatformHelper


class MyListener(roypy.IDepthDataListener):
    def __init__(self, num_frames, output, frame_to_export):
        super(MyListener, self).__init__()
        self.maxFrameIdx = num_frames-1
        self.outputName = output
        self.frameToExport = frame_to_export
        self.curFrameIdx = -1  # incremented before first use
        self.done = False
        if frame_to_export < 0:
            self.extract_only_one_frame = False
        else:
            self.extract_only_one_frame = True

    def onNewData(self, data):
        self.curFrameIdx += 1

        if self.extract_only_one_frame and not (self.curFrameIdx == self.frameToExport):
            return

        numPoints = data.getNumPoints()

        # Open file using the frame number defined as index (starting from 0) as part of the name.
        filename = str(self.outputName) + str(self.curFrameIdx) + ".ply"
        ply_file = open(filename, "w+")

        # Define the PLY header.
        ply_file.write('ply\n')
        ply_file.write('format ascii 1.0\n')
        ply_file.write('comment generated by the sample_export_ply.py script ' + roypy.getVersionString() + '\n')
        ply_file.write('element vertex ' + str(numPoints) + '\n')
        ply_file.write('property float x\n')
        ply_file.write('property float y\n')
        ply_file.write('property float z\n')
        ply_file.write('property uchar red\n')
        ply_file.write('property uchar green\n')
        ply_file.write('property uchar blue\n')
        ply_file.write('element face 0\n')
        ply_file.write('property list uchar int vertex_index\n')
        ply_file.write('end_header\n')

        # Find out the amplitude difference of the current point set.
        minAmp = 65535
        maxAmp = 0

        for i in range(numPoints):
            if data.getDepthConfidence(i) > 0:
                if data.getGrayValue(i) < minAmp:
                    minAmp = data.getGrayValue(i)
                elif data.getGrayValue(i) > maxAmp:
                    maxAmp = data.getGrayValue(i)

        rangeDiff = maxAmp - minAmp

        # We don't want to divide by zero if we have no amplitude difference.
        if rangeDiff == 0:
            rangeDiff = 1

        # Write rounded coordinates and normalized amplitudes of each point to the PLY file.
        for i in range(numPoints):
            pixelColor = 0
            if data.getDepthConfidence(i) > 0:
                # Color the points in the point cloud according to the amplitude
                pixelColor = ((data.getGrayValue(i) - minAmp) / rangeDiff) * 255.0

            # Round and remove trailing zeroes
            xString = str(round(data.getX(i), 6))
            xString = xString.rstrip('0').rstrip('.')

            yString = str(round(data.getY(i), 6))
            yString = yString.rstrip('0').rstrip('.')

            zString = str(round(data.getZ(i), 5))
            zString = zString.rstrip('0').rstrip('.')

            ply_file.write(xString + " " + yString + " " + zString + " ")

            # Make sure the color is rgb (0-255)
            pixelColor = int(pixelColor)
            ply_file.write(str(pixelColor) + " " + str(pixelColor) + " " + str(pixelColor) + "\n")

        ply_file.close()

        if self.extract_only_one_frame and (self.curFrameIdx == self.frameToExport):
            self.done = True
            return
        elif self.curFrameIdx == self.maxFrameIdx:
            self.done = True


def main():
    platformHelper = PlatformHelper()

    # Set the available arguments
    parser = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser)
    parser.add_argument("--frame", type=int, default=-1, help="index of frame (starting from 0) to be exported to PLY.")
    parser.add_argument("--output", type=str, default="outputFrame", help="name (base) of the output file(s)")
    options = parser.parse_args()

    if options.rrf is None:
        parser.print_help()
        sys.exit(1)

    opener = CameraOpener(options)

    try:
        cam = opener.open_camera()
    except:
        print("Could not open camera recording interface.")
        sys.exit(1)

    # Retrieve the interface that is available for recordings
    replay = cam.asReplay()
    print("Frame count: ", replay.frameCount())
    print("File version: ", replay.getFileVersion())

    frameToExport = options.frame
    if frameToExport >= 0:
        if frameToExport < replay.frameCount():
            print("Exporting frame: ", frameToExport)
        else:
            print("Cannot export frame with index", frameToExport,
                  "because it is not in range of [0,", replay.frameCount()-1, "].")
            sys.exit(1)
    else:
        print("Exporting all frames.")

    # Make sure the recording is sending the data on maximum speed and doesn't repeat
    replay.useTimestamps(False)
    replay.loop(False)

    listener = MyListener(replay.frameCount(), options.output, frameToExport)

    cam.registerDataListener(listener)
    cam.startCapture()

    # Wait for onNewData to be called for each frame
    while not listener.done:
        time.sleep(0.5)
    cam.stopCapture()

    print("Done")


def process_event_queue(q, writer, replay):
    while True:
        try:
            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range(0, len(q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            break
        else:
            print(str(replay.currentFrame()))
            writer.write(item)


if __name__ == "__main__":
    main()
