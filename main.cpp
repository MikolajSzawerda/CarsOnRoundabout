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
#define OBJECT_MIN_SIZE 200

typedef vector<vector<Point> > Contours;

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

void get_contours(Contours& contours, Mat const& mask, Point offset){
    findContours(mask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    contours.erase(remove_if(contours.begin(), contours.end(), [&](auto const& contour) {
        return contourArea(contour) < OBJECT_MIN_SIZE;
    }), contours.end());
    for(auto& c:contours){
        for(auto& p:c){
            p+=offset;
        }
    }
}

void draw_objects_info(Contours const& contours, Mat& frame){
    String str = "# of objects: " + to_string(contours.size());
    putText(frame, str, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0, 255), 1);
}

vector<Point> getObjectsMiddlePoint(Contours const& contours){
    vector<Point> points(contours.size());
    vector<Point> approx;
    for(int i=0;i<contours.size(); i++){
        approxPolyDP(contours[i], approx, 3, true);
        Rect box = boundingRect(approx);
        points[i] = (box.tl()+box.br())/2.0;
    }
    return points;
}

void draw_middle_points(Mat const& frame, vector<Point> const& middlePoints){
    int i=0;
    for(auto const& middlePoint: middlePoints){
        circle(frame, middlePoint, 3, Scalar(0, 0, 255), FILLED);
        printf("Object nr. %d X: %d Y: %d\n", i, middlePoint.x, middlePoint.y);
    }
}


int main() {
    VideoCapture capture("../movies/synth2/522.mp4");
    if(!capture.isOpened()){
        cout<< "Cannot open" <<endl;
        return -1;
    }
    Mat frame, preprocessed, mask;
    Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorMOG2();
    Contours contours;
    capture >> frame;
    Rect roi = selectROI(frame);
    for(;;){
        capture >> frame;
        if(frame.empty()) break;
        Mat image = frame(roi);
        GaussianBlur(image, preprocessed, Size(BLUR_SIZE, BLUR_SIZE), 0);

        backSub->apply(preprocessed, mask);
        processMask(mask);

        get_contours(contours, mask, roi.tl());
        drawContours(frame, contours, -1, Scalar(0, 255, 0));
        draw_objects_info(contours, frame);

        vector<Point> objectsCoor = getObjectsMiddlePoint(contours);
        draw_middle_points(frame, objectsCoor);

        imshow("Frame", frame);
        imshow("Mask", mask);

        int keyboard = waitKey(30);
        if(keyboard=='q') break;
    }
    return 0;
}
