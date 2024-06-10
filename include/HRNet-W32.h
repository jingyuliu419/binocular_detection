#ifndef HRNet_h
#define HRNet_h

#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
using namespace std;

class HRNet
{
public:
	HRNet();
	cv::Mat detect(cv::Mat& frame);
	void coordinateTrans(cv::Mat& srcimg);
	void draw(cv::Mat& frame);
	void clear();
	std::pair<int, int> Keypoint(const int& i);
private:

#ifdef _WIN32
	const wchar_t* model_path = L"weights/sim_HRNet-W32.onnx";
#else
	const char* model_path = "../weights/sim_HRNet-W32.onnx";
#endif

	// std::string classesFile = "class.names";//COCO
	std::string classesFile = "../res/keypoints.txt";

	//����������������ʹ�õ���־��¼״̬,ʹ���κ����� Onnxruntime ����֮ǰ���봴��һ�� Env
	//ָ��Ҫ��ʾ����־��Ϣ�����������
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "HRNet-W32");
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
	int64_t inpHeight = 256;
	int64_t inpWidth = 192;
	
	std::vector<float> scores;//17����ķ���
	std::vector<pair<int, int>> keypoints;//17�����λ��
	const bool keep_ratio = true;//  PaddedResize(���ֳ�����) / DirectResize

	//��������
	std::vector<std::string> class_names;//���ģ����Ԥ��������
	std::vector<int> classIds;

	cv::Mat resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left);
	void Pixel_normalize(cv::Mat& img);//resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
};

#endif