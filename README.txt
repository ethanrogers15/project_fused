Project FUSED README

Fusion-Based Utilization and Synthesis of Efficient Detections

Check out our website here for background information about the project: 
https://sites.google.com/view/projectfused

Point of Contact: Ethan Rogers, ethan.c.rogers@gmail.com

This repository contains a public-facing version of the virtual environment 
that was used to develop the detection fusion algorithm for the project. 

Other repositories for the ROS2 environment are located here:
https://github.com/ethanrogers15/project_fused_ros (more presentable version)
https://github.com/ethanrogers15/project_fused_ros_original (original environment)

To run this virtual environment, you will need Docker Desktop and VSCode with
the Remote Development & Dev Containers extensions. The virtual environments
for Project FUSED were built on a Windows laptop, but we used WSL with Ubuntu
to interface with the ROS environment. So, we recommend making sure that WSL 
is set up with Ubuntu if you are using a Windows laptop. 

After cloning the repository to a directory in your file system, opening the 
directory in VSCode will result in a prompt to "build" the environment as a 
Dev Container. After starting the build, it will take 10-20 minutes to complete
the process - some of the packages take a long time to install!

Once the environment is built, make sure that the environment's Python
interpreter is in use when selecting a Python file, and you should be good to
go!

All of the testing data that was recorded and analysis data that was randomly
picked is located in the 'data' directory. The 'models' directory contains the
three object detection models used in the fusion workflow. The 'src' directory
contains all of the scripts written during the development and analysis of the 
algorithm. Note that any Python scripts located in the 'other' directory within
'src' should be moved up one level to 'src' if you wish to run those files.

The two main files are 'workflow.py' and 'workflow_analysis.py'. The
'workflow.py' file runs the algorithm on a certain scenario from the 'data'
directory and outputs new images with detections from the fusion workflow along
with a .mp4 video stringing the images together. The 'workflow_analysis.py'
file uses the fusion workflow on the analysis dataset in the 'data' directory
to evaluate the algorithm's performance and compare it with the performance of
individual models on their sensors. Precision & recall are calculated, yielding
PR Curves and Average Precision results.