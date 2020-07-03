

#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <map>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void draw_match(Mat& img1, Ptr<SURF>& detector, int index,  bool extended = false){
   
    time_t start, end;
    start = clock();    
    vector<KeyPoint>key_points_img1;
    detector->detect(img1, key_points_img1);
    
    Mat descriptor_img1;
    detector->compute(img1, key_points_img1, descriptor_img1);
    end = clock();
    double compute_time = (end - start) * 1.0 / CLOCKS_PER_SEC;

    cout<<index<<","<<key_points_img1.size()<<","<<compute_time<<endl;
    return ;
  
  
}

bool compare_hessian(const KeyPoint &kp1, const KeyPoint &kp2) {
    return kp1.response > kp2.response;
}

void draw_match_double(Mat& img1,
                Ptr<SURF> &detector1,
                int index,
                int max_size = 400,
                int min_size = 200,
                int max_hessian = 800,
                int min_hessian = 5,
                bool extended = false){
    time_t start, end;

    start = clock();
                    
    vector<KeyPoint>key_points_img1;

 
    detector1->detect(img1, key_points_img1);

    int hessian = detector1->getHessianThreshold();
    while ((key_points_img1.size() < min_size || key_points_img1.size() > max_size) ){
        if ( key_points_img1.size() < min_size ){
            if (hessian <= min_hessian)
                break;
            hessian = (int)(hessian/1.2);
            detector1->setHessianThreshold(hessian);
            break;
        }
        else
        {
            sort(key_points_img1.begin(), key_points_img1.end(), compare_hessian);
            key_points_img1.resize(max_size);
            hessian = (int)(hessian * 1.2);  
            detector1->setHessianThreshold(hessian);
            break;
        }
    }

    Mat descriptor_img1;

    detector1->compute(img1, key_points_img1, descriptor_img1);
    end = clock();
    double compute_time = (end - start) * 1.0 / CLOCKS_PER_SEC;

    cout<<index<<","<<key_points_img1.size()<<","<<compute_time<<endl;
    
  
}

int main()
{
	
	VideoCapture cp;
	Mat frame;
    cp.open("/home/jiaming/Documents/research/videos/test_video.mp4");
    Ptr<SURF> detector = SURF::create(400);
    int index = 0;
    while (1)
	{
		cp >> frame;
        if(frame.empty())
            break;
        cout<< frame.size <<endl;
        //draw_match(frame, detector, index);
        //draw_match_double(frame, detector, index);
        index++;
        waitKey(30);
	}
 
	return 0;

}
