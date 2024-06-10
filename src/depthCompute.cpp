#include "depthCompute.h"

using namespace std;
using namespace cv;

/*---------------检测与匹配--------------*/
depthCompute::depthCompute(int x, int y)
{
    this->x = x;
    this->y = y;
}

void depthCompute::De_distortion(cv::Mat &src, cv::Mat &dst)//引用，即在原内存上操作
{

//在原图上进行裁减 引用
    // Mat left(src, Rect(1280, 0, 1280, 960)); // 左上角的column值，左上角的row值，矩形column方向上的宽，矩形row方向上的宽）
    // Mat right(src, Rect(0, 0, 1280, 960));   // x y w h
Mat region(src, Rect(0, 0, 1280, 720)); // Define a region in the source image

Mat left = region.clone();  // Create a copy of the defined region for 'left'
Mat right = region.clone(); // Create another copy of the same region for 'right'

//openCV 立体矫正的api
    cv::Mat R1, R2, Q;             // 左,右摄像机旋转矩阵；左右相机投影矩阵；重投影矩阵
    cv::Rect validRoiL, validRoiR; // 用于标记remap图像中包含有效像素的矩形区域。
    cv::Size imageSize(1280, 960);
    //去畸变和立体矫正同时进行
    stereoRectify(cameraMatrixL, distCoeffsL,
                  cameraMatrixR, distCoeffsR,
                  imageSize, R, T,
                  R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, -1,
                  imageSize, &validRoiL, &validRoiR);
    // alpha取值为 0 时，OpenCV对校正后图像缩放和平移，使remap图像只显示有效像素（即去除不规则的边角区域），适用于机器人避障导航等应用；
    // 为1时，remap图像将显示所有原图像中包含的像素，该取值适用于畸变系数极少的高端摄像头；
    // 在0-1之间时，OpenCV按对应比例保留原图像的边角区域像素。
    // 为-1时，OpenCV进行default sacling.
//映射
    cv::Mat lmap1, lmap2;
    cv::Mat rmap1, rmap2;
    initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, imageSize, CV_16SC2, lmap1, lmap2); // 计算左相机视图矫正映射
    initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, imageSize, CV_16SC2, rmap1, rmap2); // 计算右相机视图矫正映射

    cv::Mat imgL, imgR;
    // remap(left, imgL, lmap1, lmap2, CV_INTER_AREA); // 左校正
    // remap(right, imgR, rmap1, rmap2, CV_INTER_AREA); // 右校正
    //速度可以减少 2ms
    remap(left, imgL, lmap1, lmap2, INTER_NEAREST); // 左校正
    remap(right, imgR, rmap1, rmap2, INTER_NEAREST); // 右校正
    
    // cv::hconcat(imgR, imgL, dst);  // 1.5ms  合并Mat便于同时显示

    // std::cout<< "P1" << P1 << std::endl;
    // std::cout<< P1.at<double>(0,0) << " " << P1.at<double>(0,1) << " " << P1.at<double>(0,2) << " " << P1.at<double>(0,3) << std::endl;
    // std::cout<< P1.at<double>(1,0) << " " << P1.at<double>(1,1) << " " << P1.at<double>(1,2) << " " << P1.at<double>(1,3) << std::endl;
    // std::cout<< P1.at<double>(2,0) << " " << P1.at<double>(2,1) << " " << P1.at<double>(2,2) << " " << P1.at<double>(2,3) << std::endl;
    // std::cout<< "P2" << P2 << std::endl;
    // std::cout<< P2.at<double>(0,0) << " " << P2.at<double>(0,1) << " " << P2.at<double>(0,2) << " " << P2.at<double>(0,3) << std::endl;
    // std::cout<< P2.at<double>(1,0) << " " << P2.at<double>(1,1) << " " << P2.at<double>(1,2) << " " << P2.at<double>(1,3) << std::endl;
    // std::cout<< P2.at<double>(2,0) << " " << P2.at<double>(2,1) << " " << P2.at<double>(2,2) << " " << P2.at<double>(2,3) << std::endl;
    // std::cout<< "R1" << R1 << std::endl;
    // std::cout<< "R2" << R2 << std::endl;

    this->imgDistortedL = imgL;
    this->imgDistortedR = imgR;
    // ORB_demo(imgL, imgR, dst); // imgL和R 没有线
}

void depthCompute::depth(cv::Mat& dst)
{

    float disparity = 0.0;
    // 像素点
    if(x>0 && y>0 && xr>0 && yr>0)
    {
        double Z;//按照双目的计算流程先要找到Z,才能找到实验所需的Y；

        // 求立体校正后,像素坐标系到成像平面坐标系的放缩系数(左相机)，双目匹配的代码
        double al = P1.at<double>(0, 0) / 3.6; // fx / f
        double ar = P2.at<double>(0, 0) / 3.6; // fx / f
        double bl = P1.at<double>(1, 1) / 3.6; // fy / f
        // double br = cameraMatrixR.at<double>(1, 1) / 3.6焦距; // fy / f

        int ul = x;
        int ur = xr;

        // 求立体校正后,像素坐标转成像平面坐标并计算视差
        disparity = ((ul - P1.at<double>(0, 2)) / al) - ((ur - P2.at<double>(0, 2)) / ar);
        /*P1.at<double>(0, 2) 是 OpenCV 中访问矩阵元素的一种方式。这里的 P1 是一个 cv::Mat 对象，
而 .at<double>(0, 2) 用于访问位于矩阵第 0 行第 2 列的元素，并假设该元素的数据类型是 double。*/
        // circle(dst, Point(x, y), 10, Scalar(0, 0, 255));//BGR

        Z = 3.6 * abs(T.at<double>(0)) / disparity;//求立体校正后,相机坐标系下的Z，平移矩阵，基线，必须为正数
        // 求立体校正后,相机坐标系下的Y
        Y = Z * ((y - P1.at<double>(1, 2)) / bl) / 3.6;

        computeCompress();

        ostringstream oss;
        oss << " " << Y << "mm" ;
        string str(oss.str());
        putText(dst, str, Point(x, y), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 4, 8);
        cout << "立体校正后, 左相机坐标系下的Y:"<< Y << "mm" << endl;

        x=0, y=0, xr=0, yr=0;
    }else{
        //pass
    }
}

int depthCompute::compressDepth()
{
    return this->Y;
}

void depthCompute::get_compressBaseline(){//手先放在这里不动获取一个基准；用来算按压次数的；
    //baseline:双目相机使用  按压基准

    compressNumber = 0;
    
    this->compressBaseline = this->Y;
    std::cout << "按压基准(mm) : " << this->Y << std::endl;
};

void depthCompute::computeCompress(){
    //深度,次数和频率

    if(compressTrack.size() == 0){
        // std::cout << "compressTrack.size() == 0 compressTrack.size() == 0" << std::endl;
        compressTrack.push(this->Y);
        return;
    }

    if(compressTrack.back() < compressBaseline && this->Y > compressBaseline){
        ++compressNumber;
        
        currentTime = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch());
        if(compressNumber>1){
            auto durationalTime = currentTime - lastTime;
            compressFrequency = 60 * 1000 / static_cast<int16_t>(durationalTime.count());//times/min
        }
        lastTime = currentTime;
    }
        
    if(compressTrack.size()==50){
        compressTrack.pop();
    }
    
    compressTrack.push(this->Y);
};

std::pair<int16_t, int16_t> depthCompute::getCompress(){
    return std::make_pair(compressNumber, compressFrequency);
}

void depthCompute::clear_compress(){
    compressNumber = 0;
    compressFrequency = 0;
    compressTrack = std::queue<int32_t>();
}