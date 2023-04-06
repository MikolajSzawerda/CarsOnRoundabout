#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <iostream>
using namespace cv;
using namespace std;

#define DILATE_MASK_SIZE 7
#define BLUR_SIZE 5
#define ERODE_MASK_SIZE 9

void cutObjectsByMask(Mat& frame, Mat const& mask){
    vector<Mat> channels(3);
    split(frame, channels);
    for(auto channel:channels)
        bitwise_and(channel, mask, channel);
    merge(channels, frame);
}

void processMask(Mat& mask){
    dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(DILATE_MASK_SIZE, DILATE_MASK_SIZE)));
    erode(mask, mask, getStructuringElement(MORPH_RECT, Size(ERODE_MASK_SIZE, ERODE_MASK_SIZE)));
    dilate(mask, mask, getStructuringElement(MORPH_RECT, Size(5, 5)));
    threshold(mask, mask, 150, 255, THRESH_BINARY);
}


int main() {
    VideoCapture capture("../movies/synth2/512.mp4");
    if(!capture.isOpened()){
        cout<< "Cannot open" <<endl;
        return -1;
    }
    Mat frame, preprocessed, mask;
    Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorMOG2();
    for(;;){
        capture >> frame;
        if(frame.empty()) break;
        GaussianBlur(frame, preprocessed, Size(BLUR_SIZE, BLUR_SIZE), 0);

        backSub->apply(preprocessed, mask);
        processMask(mask);
        cutObjectsByMask(frame, mask);
        imshow("Frame", frame);
        imshow("Mask", mask);

        int keyboard = waitKey(30);
        if(keyboard=='q') break;
    }
    return 0;
}
