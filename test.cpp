#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Initialize VideoCapture object to open the default camera
    VideoCapture capture(0, cv::CAP_V4L2);
;
    
    // Set the frame width and height
    capture.set(CAP_PROP_FRAME_WIDTH, 2560);
    capture.set(CAP_PROP_FRAME_HEIGHT, 960);
    
    // Check if the camera opened successfully
    if (!capture.isOpened()) {
        cout << "Could not open the input video" << endl;
        return -1;
    }
    
    // Set the video stream format to MJPG
    capture.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    
    // Create a window to display the video
    namedWindow("Video", WINDOW_AUTOSIZE);
    
    // Loop to capture and display each frame
    while (true) {
        Mat frame;
        capture >> frame; // Capture a new frame
        
        // Check if the frame is empty (end of video stream)
        if (frame.empty()) {
            break;
        }
        
        imshow("Video", frame); // Display the frame
        
        // Wait for 30 milliseconds and break the loop if a key is pressed
        if (waitKey(30) >= 0) {
            break;
        }
    }
    
    // Release the VideoCapture object and close all OpenCV windows
    capture.release();
    destroyAllWindows();
    
    return 0;
}
