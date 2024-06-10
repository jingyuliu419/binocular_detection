#include <iostream>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include "yolov5s.h"
#include "HRNet-W32.h"
#include "depthCompute.h"
#include "blackContours.h"//颜色区域分割的找mark中心的头文件
// #include "HoughLines.h"//如果想用直线检测的找mark中心的头文件的话就去掉注释
// #include "Shi-Tomasi_rectCenter.h"
#include <thread>
#include <condition_variable>
using namespace std;
using namespace cv;

// 通过全局变量获取鼠标点击点的坐标
int globalX;
int globalY;

//创建文件流
std::ofstream file("../res/depth.txt", std::ios::app);

/*相对于一开始的版本，主要是修改了基线，所以只是修改了参数R 和 T*/
void onMouse(int event, int x, int y, int flags, void *userdata)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        globalX = x;
        globalY = y;
        cout << " the Pixel of onMouse: (" << x << "," << y << ")" << endl;
    }
}

void captureThread(cv::VideoCapture &cap, std::vector<cv::Mat> &frames, std::condition_variable &capCon, std::mutex &mtx, bool &stop, bool &capThreadfinished)
{//因为图片解码慢，所以用多线程；
    while(!stop)
    {
        if(frames.empty())
        {
            //只在里面的帧被取走以后进行读取
            double t9 = getTickCount();
            cv::Mat frame;
            cap >> frame;// 14ms cap采集到的被压缩了的图像数据 的解码
            // cap.read(frame);
            // cap.grab();
            // cap.retrieve(frame);
            double t10 = getTickCount();
            std::cout << "capture >> frame 用时:" << (t10 - t9) * 1000 / getTickFrequency() << "ms" << std::endl;

            std::lock_guard<std::mutex> lock(mtx);//资源上锁
            frames.push_back(frame);//push_back,vector的尾部添加一个新元素；
            capCon.notify_one();//解锁一个线程，如果有多个，则未知哪个线程执行
        }
        else{
            std::unique_lock<std::mutex> lock(mtx);//资源上锁
            capCon.wait(lock, [&]{ return frames.empty(); });//Wait until notified
        }
    }
    //notify
    std::lock_guard<std::mutex> lock(mtx);//资源上锁
    capThreadfinished = true;//条件变量
    capCon.notify_one();////解锁一个线程，如果有多个，则未知哪个线程执行
}

void depthThread(depthCompute &depthCompute, YOLO &yolo_model, YOLO &yolo_model_r, int &thFinishedCount, std::condition_variable &orbCon, std::mutex &mtx)
{
    double t3 = getTickCount();

    // 取出mark点中心点在左右相机中的坐标
    //左图
    for (size_t i = 0; i < yolo_model.indices.size(); ++i)
    {
        int index = yolo_model.indices[i];
        if (yolo_model.class_names[yolo_model.classIds[index]] == "mark")
        {
            FindRectCenter findRectCenter;

            //防止越界
            int x = yolo_model.boxes[index].tl().x;
            int y = yolo_model.boxes[index].tl().y;
            int w = yolo_model.boxes[index].br().x - yolo_model.boxes[index].tl().x;
            int h = yolo_model.boxes[index].br().y - yolo_model.boxes[index].tl().y;

            cv::Mat mark(depthCompute.imgDistortedL, Rect(x, y, w, h));//取出mark的图像
            cv::Point rectCenter = findRectCenter(mark);//找到取出的mark图像的中心点
            depthCompute.x = yolo_model.boxes[index].tl().x + rectCenter.x;//找到取出mark图像中心点的位置后加上全图的位置信息
            depthCompute.y = yolo_model.boxes[index].tl().y + rectCenter.y;
            // depthCompute.x = 0.5 * (yolo_model.boxes[index].tl().x + yolo_model.boxes[index].br().x);
            // depthCompute.y = 0.5 * (yolo_model.boxes[index].tl().y + yolo_model.boxes[index].br().y);
            break;
        }
    }
    //右图
    for (size_t i = 0; i < yolo_model_r.indices.size(); ++i)
    {
        int index = yolo_model_r.indices[i];
        if (yolo_model_r.class_names[yolo_model_r.classIds[index]] == "mark")
        {
            FindRectCenter findRectCenter;
            int x = yolo_model_r.boxes[index].tl().x;
            int y = yolo_model_r.boxes[index].tl().y;
            int w = yolo_model_r.boxes[index].br().x - yolo_model_r.boxes[index].tl().x;
            int h = yolo_model_r.boxes[index].br().y - yolo_model_r.boxes[index].tl().y;
            cv::Mat mark(depthCompute.imgDistortedR, Rect(x, y, w, h));
            cv::Point rectCenter = findRectCenter(mark);
            depthCompute.xr = yolo_model_r.boxes[index].tl().x + rectCenter.x;
            depthCompute.yr = yolo_model_r.boxes[index].tl().y + rectCenter.y;
            // depthCompute.xr = 0.5 * (yolo_model_r.boxes[index].tl().x + yolo_model_r.boxes[index].br().x);
            // depthCompute.yr = 0.5 * (yolo_model_r.boxes[index].tl().y + yolo_model_r.boxes[index].br().y);
            break;
        }
    }
    std::cout << "findRectCenter" << std::endl;
    depthCompute.depth(depthCompute.imgDistortedL); // 计算深度

    //保存深度数据，保存到depth.txt
    if(file.is_open())
    {
        struct timespec currentTime;
        clock_gettime(CLOCK_REALTIME, &currentTime);
        // 时间转换成当前时间，精确到毫秒
        std::time_t seconds = currentTime.tv_sec;
        std::time_t nanoseconds = currentTime.tv_nsec;

        file << (long long)seconds*1000000 + nanoseconds/1000 << "\t" << depthCompute.compressDepth() << std::endl;//compressDepth私有变量，这样写为了保证安全；
    }

    double t4 = getTickCount();
    std::cout << "depth用时:" << (t4 - t3) * 1000 / getTickFrequency() << "ms" << std::endl;
    
    std::lock_guard<std::mutex> lock(mtx);//加锁
    thFinishedCount++;
    orbCon.notify_one();//
}

void HRNetThread(HRNet &HRNet_model, depthCompute &depthCompute, int &thFinishedCount, std::condition_variable &orbCon, std::mutex &mtx)
{
    std::lock_guard<std::mutex> lock(mtx);//资源上锁
    double t5 = getTickCount();
    cv::Mat resizedimg = HRNet_model.detect(depthCompute.imgDistortedL); // 推理图片放入私有变量，返回PaddedResized的图片
    HRNet_model.coordinateTrans(depthCompute.imgDistortedL);
    // HRNet_model.draw(depthCompute.imgDistortedL);
    // HRNet_model.clear();
    double t6 = getTickCount();
    std::cout << "HENET用时:" << (t6 - t5) * 1000 / getTickFrequency() << "ms" << std::endl;

    thFinishedCount++;
    orbCon.notify_one();//解锁一个线程，如果有多个，则未知哪个线程执行
}

void YOLOThread(YOLO &yolo_model_r, depthCompute &depthCompute, bool &yolo_rThreadfinished, std::condition_variable &yolo_rCon, std::mutex &mtx)
{
    double t1 = getTickCount();
    //检测右相机
    cv::Mat resizedimg = yolo_model_r.detect(depthCompute.imgDistortedR); // 推理图片放入私有变量，返回PaddedResized的图片
    yolo_model_r.coordinateTrans(depthCompute.imgDistortedR);
    double t2 = getTickCount();

    std::lock_guard<std::mutex> lock(mtx);//资源上锁，当对象超出范围时，std::cout互斥锁mtx会自动解锁，从而确保正确的资源管理并防止死锁。
    yolo_rThreadfinished = true;;
    yolo_rCon.notify_one();//解锁一个线程，如果有多个，则未知哪个线程执行
    std::cout << "yolo_model_r用时:" << (t2 - t1) * 1000 / getTickFrequency() << "ms" << std::endl;
}

int main()
{
    // std::cout<<"cv::cuda::getCudaEnabledDeviceCount():"<<cv::cuda::getCudaEnabledDeviceCount()<<std::endl;
    // cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR); // 只输出错误日志
    depthCompute depthCompute(globalX, globalY);//YOLO初始化
    HRNet HRNet_model;

    Net_config yolo_nets = {0.5, 0.7, 0.5};//置信度阈值，IOU NMS阈值，目标置信度阈值

    YOLO yolo_model(yolo_nets);
    YOLO yolo_model_r(yolo_nets);//检测右相机图像

    VideoCapture capture(0, cv::CAP_V4L2);
    // cout << "Backend : " << capture.getBackendName() << endl;
    capture.set(CAP_PROP_FRAME_WIDTH, 2560);
    capture.set(CAP_PROP_FRAME_HEIGHT, 960);
    // capture.set(CAP_PROP_FRAME_WIDTH, 1280);
    // capture.set(CAP_PROP_FRAME_HEIGHT, 480);
    if (!capture.isOpened())
    {
        cout << "Could not open the input video " << endl;
        return -1;
    }
    capture.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G')); // 视频流格式，只有用这个才能达到60帧

    VideoWriter outputVideo;
    string outputVideoPath = "../outputVideo.avi";//这个存视频不太好用，因为多线程存视频有冲突，所以保存图片了；
    outputVideo.open(outputVideoPath, CV_FOURCC('M', 'J', 'P', 'G'), 20.0, cv::Size(1280, 960), true);
    if (!outputVideo.isOpened())
    {
        cout << "Could not writer the video " << endl;
        return -1;
    }
    cv::Mat src, dst; // source  de_distortion
    int key = 0;
    const int numsThread = 1;//子线程HRNet和depth开启的数量(ORB弃用了)
    std::thread depthTh;//创建线程
    std::condition_variable orbCon;//创建一个 std::condition_variable 对象。条件变量用于阻塞一个或多个线程，直到某个线程修改线程间的共享变量，并通过condition_variable通知其余阻塞线程。从而使得已阻塞的线程可以继续处理后续的操作。
    std::mutex mtx;//创建一个互斥锁 std::mutex 对象，用来保护共享资源的访问。
    std::vector<cv::Mat> frames;

    std::condition_variable capCon;////创建一个 std::condition_variable 对象。
    //条件变量用于阻塞一个或多个线程，直到某个线程修改线程间的共享变量，并通过condition_variable通知其余阻塞线程。从而使得已阻塞的线程可以继续处理后续的操作。
    bool stop = false;
    bool capThreadfinished = false;
    //视频解码的一个线程
    std::thread capThread(captureThread, std::ref(capture), std::ref(frames), std::ref(capCon), std::ref(mtx), std::ref(stop), std::ref(capThreadfinished));
    //创建并启动了一个新的线程，这些是传递给 captureThread 函数的参数。std::ref 用于传递这些变量的引用，而不是值。这样，captureThread 函数可以在它们的原始位置上直接操作这些变量。
    capThread.detach();//当线程启动后，一定要在和线程相关联的thread销毁前，确定以何种方式等待线程执行结束。
/*detach方式，启动的线程自主在后台运行，当前的代码继续往下执行，不等待新线程结束。
join方式，等待启动的线程完成，才会继续往下执行。*/

    // 统计average_fps = sum_fps / sum_num
    double sum_fps = 0;
    double sum_num = 0;
    int pre_num = 0;
    int lastdep = 0;
    int currentdep = 0;
    int baseline = 200;
    while(!stop && capture.isOpened())
    {
        if(!frames.empty())
        {
            int thFinishedCount = 0; //判断完成的子线程数，
            double t1 = getTickCount();
            /*使用 cv::getTickCount 和 cv::getTickFrequency 可以高效地测量代码的执行时间。*/

            cv::Mat src = frames.front();
            /*front 操作用于获取或修改 std::vector 中的第一个元素。
使用 front 操作时，向量必须非空。
通过 front 操作，可以方便地访问和修改向量中的第一个元素。*/
            frames.erase(frames.begin());//有效地删除了向量 frames 中的第一个元素，并自动调整了向量的大小以反映这一变化。
            capCon.notify_one();//解锁一个线程，如果有多个，则未知哪个线程执行

            double t7 = getTickCount();
            std::cout<<src.cols<<" "<<src.rows<<std::endl;
            depthCompute.De_distortion(src, dst); // depthCompute类的声明中设置畸变参数
            
            double t8 = getTickCount();
            std::cout << "De_distortion用时:" << (t8 - t7) * 1000 / getTickFrequency() << "ms" << std::endl;

            // HRNet的模块
            // std::thread hrnetTh(HRNetThread, std::ref(HRNet_model), std::ref(depthCompute), std::ref(thFinishedCount), std::ref(orbCon), std::ref(mtx));
            // hrnetTh.detach();

            // 对右相机进行目标检测
            std::condition_variable yolo_rCon;////创建一个 std::condition_variable 对象。条件变量用于阻塞一个或多个线程，直到某个线程修改线程间的共享变量，并通过condition_variable通知其余阻塞线程。从而使得已阻塞的线程可以继续处理后续的操作。
            bool yolo_rThreadfinished = false;
            std::thread yolo_r_Th(YOLOThread, std::ref(yolo_model_r), std::ref(depthCompute), std::ref(yolo_rThreadfinished), std::ref(yolo_rCon), std::ref(mtx));//创建线程
            //创建并启动了一个新的线程，这些是传递给 captureThread 函数的参数。std::ref 用于传递这些变量的引用，而不是值。这样，captureThread 函数可以在它们的原始位置上直接操作这些变量。
            yolo_r_Th.detach();
            //当线程启动后，一定要在和线程相关联的thread销毁前，确定以何种方式等待线程执行结束。
/*detach方式，启动的线程自主在后台运行，当前的代码继续往下执行，不等待新线程结束。
join方式，等待启动的线程完成，才会继续往下执行。*/

            // 对左相机进行目标检测
            double t3 = getTickCount();
            cv::Mat resizedimg = yolo_model.detect(depthCompute.imgDistortedL);//目标检测
            yolo_model.coordinateTrans(depthCompute.imgDistortedL);
            double t4 = getTickCount();
            std::cout << "YOLO用时:" << (t4 - t3) * 1000 / getTickFrequency() << "ms" << std::endl;

            //等待右相机检测完成
            std::unique_lock<std::mutex> lock_yolo(mtx);//加锁，可以独占地管理一个互斥锁的所有权，
            yolo_rCon.wait(lock_yolo, [&]{ return yolo_rThreadfinished; });//主线程在条件变量上等待，直到 yolo_rThreadfinished 变为 true。
            lock_yolo.unlock();//解锁
            

            bool mark = false;
            if(yolo_model.indices.size()>0 && yolo_model_r.indices.size()>0) {//yolo_model.indices目标框，都要大于0

                //前10次用来设置按压基准
                if(pre_num < 10)
                    ++pre_num;
                else
                    depthCompute.get_compressBaseline();//按压基准
                //创建并开启深度的一个线程
                // 取出mark点中心点在左右相机中的坐标
                depthTh = std::thread(depthThread, std::ref(depthCompute), std::ref(yolo_model), std::ref(yolo_model_r), std::ref(thFinishedCount), std::ref(orbCon), std::ref(mtx));
                depthTh.detach();//detach方式，启动的线程自主在后台运行，当前的代码继续往下执行，不等待新线程结束。
                mark = true;
            }

            if (mark == false)
            {
                // 如果没有进入过深度测量线程
                std::cout << "none of depthead" << std::endl;
                thFinishedCount++;//要根据这个数字来了解线程的数量，所有线程结束了才能结束程序；
            }

            // 主线程等待子线程完成
            std::cout << "wait thread - - - - - " << std::endl;
            std::unique_lock<std::mutex> lock(mtx);//加锁
            orbCon.wait(lock, [&]
                        { return thFinishedCount == numsThread; });//lambda 表达式
                        /*用于检查条件变量等待的条件是否满足。
                        它捕获当前的上下文（即捕获外部作用域中的变量 thFinishedCount 
                        和 numsThread），并返回一个布尔值。条件变量将反复调用这个 lambda 表达式
                        ，直到它返回 true。*/

            // HRNet_model.draw(depthCompute.imgDistortedL);
            // HRNet_model.clear();

            yolo_model.draw(depthCompute.imgDistortedL);
            yolo_model.clear();
            yolo_model_r.draw(depthCompute.imgDistortedR);
            yolo_model_r.clear();

            namedWindow("ORB_demo");
            cv::Mat imgDistorted;
            cv::hconcat(depthCompute.imgDistortedL, depthCompute.imgDistortedR, imgDistorted);
            // cv::line(imgDistorted, cv::Point(0, 480), cv::Point(2560, 480), cv::Scalar(0,255,255));
            imshow("ORB_demo", imgDistorted); // 1.5ms左右

            currentdep = depthCompute.compressDepth();
            baseline = 215;//如果过了这个baseline就截一张图，后续可以离线分析按压姿势；
            if(lastdep > baseline && currentdep < baseline){
                // struct timespec currentTime;
                // clock_gettime(CLOCK_REALTIME, &currentTime);
                // // 时间转换成当前时间，精确到毫秒
                // std::time_t seconds = currentTime.tv_sec;
                // std::time_t nanoseconds = currentTime.tv_nsec;
                // long long posturetime = (long long)seconds*1000000 + nanoseconds/1000;
                // std::string filename = "../res/posture/ljy/wrong/" + std::to_string(posturetime) + ".jpg";
                // imwrite(filename, depthCompute.imgDistortedL);
            }
            lastdep = currentdep;

            // outputVideo<<depthCompute.imgDistortedL;
            // outputVideo<<imgDistorted;
            // imwrite("../HRNetimg.jpg", resizedimg);
            // setMouseCallback("ORB_demo", onMouse, nullptr);

            // char key_board = waitKey(10);
            key = waitKey(1);
            if (key == 's')
            {
                cout << key << endl;
                stop = true;
                std::unique_lock<std::mutex> lock_cap(mtx);
                capCon.wait(lock_cap, [&] { return capThreadfinished;} );//这里经常卡死 学习学习解决
                //lambda表达式 引用捕获
                cout << "capThreadfinished" << capThreadfinished << endl;
                return 0;
            }

            double t2 = getTickCount();
            double t = (t2 - t1) * 1000 / getTickFrequency();
            cout << "总用时：" << t << "ms" << endl;
            cout << "Frames per second: " << 1000 / t << endl;
            //显示帧率，前十张不算，因为模型加载需要时间
            if(pre_num < 10)
                ++pre_num;
            else{
                sum_fps += 1000 / t;
                ++sum_num;
                cout << "Frames per second(average): " << sum_fps / sum_num << endl;

                // struct timespec currentTime;
                // clock_gettime(CLOCK_REALTIME, &currentTime);
                // std::time_t seconds = currentTime.tv_sec;
                // std::time_t nanoseconds = currentTime.tv_nsec;
                // file << (long long)seconds*1000000 + nanoseconds/1000 << "\t" << t << std::endl;
            }
        }else{
            //wait frames get data
            // std::cout<<"frames is empty"<<std::endl;
            std::unique_lock<std::mutex> lock_cap(mtx);
            capCon.wait(lock_cap, [&]{ return !frames.empty(); });
        }
        // double fps = capture.get(CV_CAP_PROP_FPS);
        // cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;
        cout << "-------------分割线---------------" << endl;
    }

    capture.release();
    destroyAllWindows();
    return 0;
}