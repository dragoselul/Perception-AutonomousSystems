# Final Project Description
This project targets an autonomous driving scenario under an intelligent autonomous systems
perspective. The scenario considers a dynamic environment where vehicles, pedestrians, and other
typical urban entities can coexist without risk. Such a dynamic and unstructured environment can
result in occlusions in the sensors of the vision-based autonomous driving systems. For this project,
detection and tracking of entities of interest is performed with a stereo camera that is fixed on the car,
looking forward. Objects in the environment in front of the car need to be tracked in 3D, even if
occlusions happen, and be classified in their respective object type.

You are provided with:

• Three data sequences that contain:

    • Sequence 1: rectified and raw stereo pair images of pedestrians and cyclists, including
    ground truth (bounding boxes, depth).

    • Sequence 2: rectified and raw stereo pair images of pedestrians, cyclists and cars with
    occlusion, including ground truth (bounding boxes, depth).

    • Sequence 3: rectified and raw stereo pair images of pedestrians, cyclists and cars with
    occlusion, without ground truth.

• Calibration of the stereo camera system.

• An image pair of calibration patterns captured by the stereo camera system.


## Project Goals
• Detect objects in the environment and track them in 3D, even under occlusions (use the
rectified images).

• Train a machine learning system that can classify unseen images into the 3 classes
(pedestrians, cyclists, and cars) based either on 2D or 3D data.

    Use the web or/and capture your own images to create your training set. The image
    sequences 1 and 2 provided with the project will constitute your validation set and
    the sequence 3 your testing set (use the rectified images for this task).

• Calibrate the stereo camera system (use the provided calibration pattern images) and rectify
the raw stereo pairs. How does your rectification compare to the provided one? How close did
you get to the provided calibration?

## Project Report
Each group of students will submit a report about their Final Project by the end of the course.
Submission will take place by uploading on DTU Learn. The report needs to be 10±2 pages and include
a link to a video demonstrating the group's main outcomes.
Approval of report (based on a pass/fail evaluation) is mandatory for the group members to participate
in the exam. The report will account for 10% of the final grade.
