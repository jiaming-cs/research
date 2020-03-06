#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric> //accumulate
#include <cmath>
#include <climits> //INT_MAX
#include <fstream>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;




Mat get_hist(Mat img_hsv, Point center, int r = 3, int bins = 60){
    
    Mat g_kernal_x = getGaussianKernel(2*r, 2);
    Mat g_kernal_y = getGaussianKernel(2*r, 2);
    Mat g_kernal_t = g_kernal_x * g_kernal_y.t();
    Mat g_kernal = g_kernal_t(Rect(r, r, r, r));
    normalize(g_kernal, g_kernal, 1, 100, NORM_MINMAX);
    g_kernal.convertTo(g_kernal, CV_8UC1);
    //cout<<g_kernal<<endl;
    double span = 360 / bins;
    double temp [bins] = {};
    //cout<<format(g_kernal, Formatter::FMT_PYTHON)<<endl;
    //cout<<img_hsv.size<<endl;
    int i_start = max((int)center.y - r, 0);
    int i_end = min((int)center.y + r, (int)img_hsv.rows);
    int j_start = max((int)center.x - r, 0);
    int j_end = min((int)center.x + r, (int)img_hsv.cols);
    for (int i = i_start; i < i_end; i++){
        for (int j = j_start; j < j_end; j++){
            int dy = abs(i - center.y);
            int dx = abs(j - center.x);
            if ( dx*dx + dy*dy > r*r)
                continue;
            //cout<<(int)img_hsv.at<uchar>(i, j)<<endl;
            
            int h = img_hsv.at<uchar>(i, j);
            //cout<<h<<endl;
            //cout<<(int)g_kernal.at<uchar>(dx, dy)<<endl;
            temp[(int)(h / span)] += (int)g_kernal.at<uchar>(dy, dx);
        }
    }
    
  
    int total = accumulate(temp, temp+bins, 0);
    Mat hist(1, bins, CV_64FC1, temp);
    hist = hist / total;
    //cout<<"1"<<endl;
    return hist;
}




Mat addPad(Mat img){
    int length = max(img.cols, img.rows);
    Mat shiftImg = Mat::zeros(2 * length, 2 * length, img.type());
    Mat ROI = shiftImg(Rect(length - img.cols / 2, length - img.rows / 2, img.cols, img.rows));
    img.copyTo(ROI);
    return shiftImg;
}

Point2f getPosition(Point2f p, Mat affineMat){
    double x = p.x * affineMat.at<double>(0, 0) + p.y * affineMat.at<double>(0, 1) + affineMat.at<double>(0, 2);
    double y = p.x * affineMat.at<double>(1, 0) + p.y * affineMat.at<double>(1, 1) + affineMat.at<double>(1, 2);
    return Point2f(x, y);
}


Mat rotateImg(Mat img, Mat affineMat, Point2f center){
    
    
    Mat out;
    //cout<<format(affineMat, Formatter::FMT_PYTHON)<<endl;
    warpAffine(img, out, affineMat, img.size() , INTER_LINEAR, BORDER_TRANSPARENT); 
    return out;
    
}

bool isCorrect(Point2f p1, Point2f p2, double d = 10){
    return d > sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

Mat get_hue(Mat img){
    vector<Mat> chanles;
    split(img, chanles);
    Mat out = chanles[0].clone();
    return out;
}

vector<vector<DMatch>> my_knn_matcher(Mat img1, Mat img2, Mat des1, Mat des2, vector<KeyPoint> kp1, vector<KeyPoint> kp2, double weight){
    Mat hue_img1;
    Mat hue_img2;
    cvtColor(img1, img1, COLOR_BGR2HSV_FULL);
    cvtColor(img2, img2, COLOR_BGR2HSV_FULL);
    Mat img1_h = get_hue(img1);
    Mat img2_h = get_hue(img2);
    
    for (int i=0; i<kp1.size(); i++){
        hue_img1.push_back(get_hist(img1_h, kp1[i].pt, kp1[i].size));
        //hue_img1.push_back(get_hist(img1_h, kp1[i].pt));
    }
    for (int i=0; i<kp2.size(); i++){
        hue_img2.push_back(get_hist(img2_h, kp2[i].pt, kp2[i].size));
    }
    //cout<<format(hue_img1.row(0), Formatter::FMT_PYTHON)<<endl;
    //cout<<format(hue_img2.row(0), Formatter::FMT_PYTHON)<<endl;
    
    vector<vector<DMatch>> knn_matches;
    for (int i=0; i<kp1.size(); i++){
        vector<DMatch> d_match(2);
        DMatch d1;
        DMatch d2;
        d1.distance = INT_MAX * 1.0 -1;
        d2.distance = INT_MAX * 1.0;
        d1.queryIdx = i;
        d2.queryIdx = i;
        for (int j=0; j<kp2.size(); j++){
            double norm_distance = norm(des1.row(i), des2.row(j), NORM_L2);
            double hue_distance = hue_img1.row(i).dot(hue_img2.row(j));
            //cout<<format(hue_img1.row(i), Formatter::FMT_PYTHON)<<endl;
            //cout<<format(hue_img2.row(j), Formatter::FMT_PYTHON)<<endl;
            double weighted_distance = (1 - weight) * norm_distance - (weight * norm_distance * hue_distance);
            if (weighted_distance < d1.distance){
                d1.distance = weighted_distance;
                d1.trainIdx = j;
            }
            else if (weighted_distance < d2.distance){
                d2.distance = weighted_distance;
                d2.trainIdx = j;
            }
        }
        d_match[0] = d1;
        d_match[1] = d2;
        knn_matches.push_back(d_match);
    }

    return knn_matches;
}


Mat draw_match(Mat img1, Mat img2, Mat affineMat, int matchNum = 50, bool extended = false){
   
    int minHessian=600;
    Ptr<SURF>  detector = SURF::create(minHessian, 4, 3, extended);
    vector<KeyPoint>key_points_img1, key_points_img2;
    detector->detect(img1, key_points_img1);
    detector->detect(img2, key_points_img2);
 
    
    Mat descriptor_img1, descriptor_img2;
    detector->compute(img1, key_points_img1, descriptor_img1);
    detector->compute(img2, key_points_img2, descriptor_img2);
    
    
    BFMatcher matcher;
    vector<vector<DMatch>>matches;
    matcher.knnMatch(descriptor_img1, descriptor_img2, matches, 2);
    
    vector<DMatch> goodMatches;

    double ratio = 0.8;

    for(int i = 0; i < matches.size(); i++)
    {
        
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            goodMatches.push_back(matches[i][0]);
        }

    }
    
    sort(goodMatches.begin(), goodMatches.end());
    Mat img_matches = Mat::zeros(img1.rows, img1.cols + img2.cols, img1.type());
    Mat ROI = img_matches(Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(ROI);
    ROI = img_matches(Rect(img1.cols, 0, img2.cols, img2.rows));
    img2.copyTo(ROI);

    Point2f p1, p2, position;
    Point2f shift = Point2f(img1.cols, 0);

    int good = 0;

    for(unsigned int i = 0; i < min(matchNum, (int)goodMatches.size()); ++i)
    {
        p1 = key_points_img1[goodMatches[i].queryIdx].pt;
        p2 = key_points_img2[goodMatches[i].trainIdx].pt;
        position = getPosition(p1, affineMat);
        if (isCorrect(p2, position))
        {
            line(img_matches, p1, p2+shift, Scalar(0, 255, 0));
            good++;
        }
            
        else
        {
            line(img_matches, p1, p2+shift, Scalar(0, 0, 255));
        }
            

    }

    double accuracy = good * 1.0 / min(matchNum, (int)goodMatches.size());
    string ac = "Accuracy: ";
    cout<< "Normal Accuracy:" << accuracy <<endl;
    putText(img_matches, ac + to_string(accuracy), Point(50, 50), HersheyFonts::FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255));
    pair<Mat, double> out = pair<Mat, double>(img_matches, accuracy);

   return img_matches;
}

Mat draw_match_hsv(Mat img1, Mat img2, Mat affineMat, double weight, int matchNum = 50, bool extended = false){
   
    int minHessian=600;
    Ptr<SURF>  detector = SURF::create(minHessian, 4, 3, extended);
    vector<KeyPoint>key_points_img1, key_points_img2;
    detector->detect(img1, key_points_img1);
    detector->detect(img2, key_points_img2);
 
    
    Mat descriptor_img1, descriptor_img2;
    detector->compute(img1, key_points_img1, descriptor_img1);
    detector->compute(img2, key_points_img2, descriptor_img2);
    
    
    vector<vector<DMatch>>matches;
    matches = my_knn_matcher(img1, img2, descriptor_img1, descriptor_img2, key_points_img1, key_points_img2, weight);
    
    vector<DMatch> goodMatches;

    double ratio = 0.8;

    for(int i = 0; i < matches.size(); i++)
    {
        
        if (matches[i][0].distance < ratio * matches[i][1].distance)
        {
            goodMatches.push_back(matches[i][0]);
        }

    }
    
    sort(goodMatches.begin(), goodMatches.end());

    
   
 
    Mat img_matches = Mat::zeros(img1.rows, img1.cols + img2.cols, img1.type());
    Mat ROI = img_matches(Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(ROI);
    ROI = img_matches(Rect(img1.cols, 0, img2.cols, img2.rows));
    img2.copyTo(ROI);

    Point2f p1, p2, position;
    Point2f shift = Point2f(img1.cols, 0);

    int good = 0;


    for(unsigned int i = 0; i < min(matchNum, (int)goodMatches.size()); ++i)
    {
        p1 = key_points_img1[goodMatches[i].queryIdx].pt;
        p2 = key_points_img2[goodMatches[i].trainIdx].pt;
        position = getPosition(p1, affineMat);
        if (isCorrect(p2, position))
        {
            line(img_matches, p1, p2+shift, Scalar(0, 255, 0));
            good++;
        }
            
        else
        {
            line(img_matches, p1, p2+shift, Scalar(0, 0, 255));
        }
            

    }

    double accuracy = good * 1.0 / min(matchNum, (int)goodMatches.size());
    //cout<<accuracy<<endl;
    string ac = "Accuracy: ";
    
    cout<< "HSV Accuracy: " << accuracy <<endl;
    putText(img_matches, ac + to_string(accuracy), Point(50, 50), HersheyFonts::FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255));
    pair<Mat, double> out = pair<Mat, double>(img_matches, accuracy);

   return img_matches;
}

Mat read_h(string file_name){
    Mat out = Mat::zeros(2, 3, CV_64FC1);
    ifstream in(file_name);
    double n;
    for (int i = 0; i<2 ; i++){
        for (int j = 0; j < 3; j++){
            in >> n;
            out.at<double>(i, j) = n;
        }
    }
    //cout<<format(out, Formatter::FMT_PYTHON)<<endl;
    return out;

}

//Usage: <Img> <Angle> <Scale> <MatchNum>

int main(int argc, char const *argv[])
{

    if (argc != 7){
        cout<<"Usage: <ImgFoder> <ImgName1> <ImgName2> <Hname> <Weight> <MatchNum>"<<endl;
        exit(0);
    }

    string base_folder = "/home/jiaming/Documents/research/dataset/";
    string img_folder = argv[1];
    string img_name1 = argv[2];
    string img_name2 = argv[3];
    string aff_name = argv[4];
    double weight = strtod(argv[5], NULL);
    int matchNum = strtol(argv[6], NULL, 10);

    Mat img1 = imread(base_folder + img_folder + "/" + img_name1 + ".ppm", IMREAD_COLOR);
    Mat img2 = imread(base_folder + img_folder + "/" + img_name2 + ".ppm", IMREAD_COLOR);
    //imshow("img2", img2);
    //waitKey(0);
    Mat affine_matrix = read_h(base_folder+img_folder+"/"+aff_name);
    Mat out_normal = draw_match(img1, img2, affine_matrix, matchNum);
    Mat out_hsv = draw_match_hsv(img1, img2, affine_matrix, weight, matchNum);
    
    imshow("Normal", out_normal);
    imshow("HSV", out_hsv);
    waitKey(0);
    return 0;
}

