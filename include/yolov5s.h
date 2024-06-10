#ifndef yolov5s_h
#define yolov5s_h

#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
using namespace std;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float iouThreshold;  // iouThreshold of Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
};

class YOLO
{
public:
	YOLO(Net_config config);
	cv::Mat detect(cv::Mat& frame);
	void coordinateTrans(cv::Mat& srcimg);
	void draw(cv::Mat& frame);
	void clear();

	//��������
	std::vector<std::string> class_names;//���ģ����Ԥ��������
	std::vector<cv::Rect> boxes;//����һ������һ��cv::Rect�����vector
	std::vector<float> confs;
	std::vector<int> classIds;
	std::vector<int> indices;//indices�������boxes���±�
private:

#ifdef _WIN32
	const wchar_t* model_path = L"weights/yolov5s.onnx";
#else
	// const char* model_path = "../weights/yolov5s.onnx";
	const char* model_path = "../weights/mark.onnx";
	// const char* model_path = "../weights/staticMark.onnx";
	// const char* model_path = "../weights/motorMark.onnx";
	// const char* model_path = "../weights/motorMark20240109.onnx";
#endif

	std::string classesFile = "../res/mark.txt";
	// std::string classesFile = "../res/class.names";//COCO
	// std::string classesFile = "../res/myclass.txt";
	//����������������ʹ�õ���־��¼״̬,ʹ���κ����� Onnxruntime ����֮ǰ���봴��һ�� Env
	//ָ��Ҫ��ʾ����־��Ϣ�����������
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolov5s-5.0");
	Ort::SessionOptions session_options;//Options object used when creating a new Session object.
	Ort::Session session{ nullptr };

	vector<string> input_name;
	vector<string> output_name;
	std::vector<const char*> input_node_names;
	std::vector<const char*> output_node_names;
	vector<vector<int64_t>> input_node_dims; // >=1 inputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
	vector<float> input_tensor_values;//PaddedResize���ͼƬ���������ع�һ��Ȼ��flatten���ͼƬ����

	int64_t inpBatch = 1;//����ֻ�ǳ�ʼ�����������ֵ�в�ͬ�����Զ��޸�
	int64_t inpChannel = 3;
	int64_t inpWidth = 640;
	int64_t inpHeight = 640;
	float confThreshold = 0; // Confidence threshold
	float iouThreshold = 0;  // iouThreshold of Non-maximum suppression threshold
	float objThreshold = 0;  //Object Confidence threshold
	const bool keep_ratio = true;//  PaddedResize(���ֳ�����) / DirectResize

	cv::Mat resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left);
	void Pixel_normalize(cv::Mat& img);//resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
};

#endif