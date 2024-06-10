#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include "HRNet-W32.h"
using namespace std;

HRNet::HRNet()
{
	// this->session_options.SetInterOpNumThreads(100);//When running a single node operation, ex.add, this sets the maximum number of threads to use.
	this->session_options.SetIntraOpNumThreads(100);//When running a single node operation, ex.add, this sets the maximum number of threads to use.
	//ͼ�Ż��ȼ���Set the optimization level to apply when loading a graph.
	this->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	// OrtCUDAProviderOptions cuda_options;
	// cuda_options.device_id = 0;
	// cuda_options.gpu_mem_limit = SIZE_MAX;
	// cuda_options.has_user_compute_stream = 1;
	// session_options.AppendExecutionProvider_CUDA(cuda_options);

	OrtTensorRTProviderOptions trt_options{};
	trt_options.device_id = 0;
	trt_options.has_user_compute_stream = 1;
	trt_options.trt_max_partition_iterations = 1000;
	trt_options.trt_min_subgraph_size = 1;
	trt_options.trt_engine_decryption_enable = false;
	trt_options.trt_dla_core = 0;
	trt_options.trt_dla_enable = 0;
	trt_options.trt_fp16_enable = 1;
	trt_options.trt_int8_enable = 0;
	trt_options.trt_max_workspace_size = 2147483648;
	trt_options.trt_int8_use_native_calibration_table = 1;
	trt_options.trt_engine_cache_enable = 1;
	trt_options.trt_engine_cache_path = "./trtcache";
	trt_options.trt_dump_subgraphs = 1;
	session_options.AppendExecutionProvider_TensorRT(trt_options);

	std::ifstream ifs(classesFile.c_str());
	std::string line;
	while (getline(ifs, line)) class_names.push_back(line);

	session = Ort::Session(env, model_path, session_options);
	//to allocate memory for the copy of the name(ort_session->GetInputNameAllocated) returned
	Ort::AllocatorWithDefaultOptions allocator;

	// print number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	size_t num_output_nodes = session.GetOutputCount();

	for (size_t i = 0; i < num_input_nodes; i++)
	{
		//input_names.push_back(ort_session->GetInputName(i, allocator));//Ort::Session->GetInputName�Ǳ������ĳ�Ա����
		//�ýӿڲ���һ��ָ�룬��ָ�������ָ���ķ������ͷŲ��Ҿ�����й©�������ⰲȫ��

		Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(i, allocator);
		//AllocatedStringPtr �� unique_ptr��ӵ���� OrtAllocators ������ַ������ڷ�Χ����ʱ�ͷ�����

		//unique_ptr.get()����ȡ��ǰ unique_ptr ָ���ڲ���������ָͨ�롣����ָ����Զ��ͷţ�����char* ת string ʵ�����
		input_name.push_back(input_name_Ptr.get());//char* ת string "images"
		input_node_names.push_back(input_name[i].data());//"images" char * __ptr64
		Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();//Get OrtTensorTypeAndShapeInfo from an OrtTypeInfo.
		auto input_dims = input_tensor_info.GetShape();//Uses GetDimensionsCount & GetDimensions to return a std::vector of the shape.
		 //input_dims��std::vector<__int64, class std::allocator<__int64> > 
		input_node_dims.push_back(input_dims);//input_node_dims[0] = vector<unsigned_int64>{1, 3, 640, 640}
	}
	for (size_t i = 0; i < num_output_nodes; i++)
	{
		//output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::AllocatedStringPtr output_name_Ptr = session.GetOutputNameAllocated(i, allocator);
		output_name.push_back(output_name_Ptr.get());
		output_node_names.push_back((char*)output_name[i].data());//"output0",�˴��ַ�������input�������⣬��ʾ�ַ���Ч

		//������Ort::Session::Run�����᷵�ص���Ϣ��������Բ�Ҫ
		//Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
		//auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		//auto output_dims = output_tensor_info.GetShape();
		//output_node_dims.push_back(output_dims);//output_node_dims[0] = vector<unsigned_int64>{1, 25200, 85}
	}

	//input_node_names = { "images" };//����ı�����ָ���ͷţ��˴����ø�ֵ�����
	//output_node_names = { "output0" };

	//�˴��Ǳ���resize������noamalization����
	this->inpBatch = input_node_dims[0][0];//1
	this->inpChannel = input_node_dims[0][1];//3
	this->inpHeight = input_node_dims[0][2];//256
	this->inpWidth = input_node_dims[0][3];//192
}

cv::Mat HRNet::resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	float newhw = float( *newh/(*neww) );
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		//hw>1 ������ң������������
		//���ֽϴ�H / W���Ĳ��䣬�ȱ�����С��һ��
		if (hw_scale > newhw) {
			*newh = this->inpHeight;
			*neww = int(float(this->inpHeight) / hw_scale);
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);//resizeͼƬ
			*left = int((this->inpWidth - *neww) * 0.5);//��Ҫ���Ŀ��ȣ����ҶԳ�
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, cv::BORDER_REPLICATE, 114);//���ͼƬ,ֻ��114����Scalar��114��0��0��
		}
		else {
			*newh = (int)this->inpWidth * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114);
		}
	}
	else {
		//Padded Resize��ԭͼ�����ȷ���Ҫ�󣬻��߲�Padded Resize
		resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

void HRNet::Pixel_normalize(cv::Mat& img)
{
	//img.convertTo(img, CV_32F);
	size_t row = img.rows;
	size_t col = img.cols;
	size_t channel = img.channels();
	//this->input_tensor_values.resize(size_t(row) * size_t(col) * size_t(channel));//�ѱ�input_tensor_size�Ĺ��ߴ���
	for (size_t c = 0; c < channel; c++)
	{
		for (size_t i = 0; i < row; i++)
		{
			for (size_t j = 0; j < col; j++)
			{
				//OpenCV����imread()��ȡͼƬʱ����ɫ��˳����BGR
				float pix = img.ptr<uchar>(i)[j * channel + channel - 1 - c];//ȡ ��cͨ��  ��i�� ��j�� ������ֵ��RGB˳��
				this->input_tensor_values[c * row * col + i * col + size_t(j)] = pix / 255.0;//���ع�һ��������flatten�ľ���
			}
		}
	}
}

cv::Mat HRNet::detect(cv::Mat& srcimg)
{

	size_t input_tensor_size = size_t(inpChannel * inpWidth * inpHeight);//3 * 640 * 640
	this->input_tensor_values.resize(input_tensor_size);

	int newh = 0, neww = 0, padh = 0, padw = 0;
	cv::Mat dstimg = this->resize_image(srcimg, &newh, &neww, &padh, &padw);//Padded resize��ģ������Ҫ��ĳߴ�
	this->Pixel_normalize(dstimg);//��һ��,����ת����float32��ֵ��������Ϊflatten������input_tensor_values

	// create input tensor object from data values
	//std::vector<int64_t> input_node_dims = { this->inpBatch, this->inpChannel, this->inpHeight, this->inpWidth };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, this->input_tensor_values.data(), input_tensor_size, input_node_dims[0].data(), input_node_dims[0].size());

	std::vector<Ort::Value> ort_inputs;
	ort_inputs.push_back(std::move(input_tensor));
	// score model & input tensor, get back output tensor
	std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{ nullptr }, this->input_node_names.data(), ort_inputs.data(), this->input_node_names.size(), this->output_node_names.data(), output_node_names.size());

	// Get pointer to output tensor float values,[0]����Ϊֻ��һ�������vectorֻ��һ��Ԫ��
	const float* rawOutput = output_tensors[0].GetTensorData<float>();
	//generate proposals
	std::vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();//std::vector of the shape.{1, 17, 64, 48}
	size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();// 1*17*64*48
	std::vector<float> output(rawOutput, rawOutput + count);//��������ָ�뵽βָ��ָ���1*17*64*48��ֵ������output

	// first 5 elements are box[4] and obj confidence
	int numClasses = (int)outputShape[1];//����� 17
	int elementsInBatch = (int)(outputShape[2] * outputShape[3]);//һ��Batch��������� 64 * 48
	scores.resize(numClasses);
	keypoints.resize(numClasses);
	
	int i = 0;
	//ע����17*64*48 ������һ��һ�����ŵ�
	for (auto it = output.begin(); it != output.begin() + count; it++,i++)
	{
		int k = i / elementsInBatch;//�ڼ�����
		if (scores[k] < *it)
		{
			scores[k] = *it;
			keypoints[k].first = (i % elementsInBatch) % outputShape[3];
			keypoints[k].second = (i % elementsInBatch) / outputShape[3];//�ڼ�������
		}
	}
	return dstimg;
}

void HRNet::coordinateTrans(cv::Mat& srcimg)
{
	//检测到的目标的举矩形框坐标对应到原图,相对于srcimg左上角的坐标
	int srch = srcimg.rows, srcw = srcimg.cols;
	int64_t resizedh = this->inpHeight;
	int64_t resizedw = this->inpWidth;
	float newhw = float( resizedh/(resizedw) );
	if (this->keep_ratio && 3*srch != 4*srcw) {
		float hw_scale = (float)srch / srcw;
		//hw>256:192
		if (hw_scale > newhw) {
			resizedh = this->inpHeight;
			resizedw = int(this->inpHeight / hw_scale);
			int left = int((this->inpWidth - resizedw) * 0.5);

			size_t sz = keypoints.size();
			float scale = float(srch) / float(resizedh);
			for (size_t i = 0; i < sz; ++i)
			{
				// cout<<"keypoints["<<i<<"].first:"<<keypoints[i].first<<endl;
				keypoints[i].first = scale * (4 * keypoints[i].first - left);
				// cout<<"keypoints["<<i<<"].first:"<<keypoints[i].first<<endl;
				keypoints[i].second = scale * 4 * keypoints[i].second;
			}
		}
		else {
			//hw小于256:192
			resizedh = (int)this->inpWidth * hw_scale;
			resizedw = this->inpWidth;
			int top = int((this->inpHeight - resizedh) * 0.5);	
			size_t sz = keypoints.size();
			float scale = float(srcw) / float(resizedw);
			for (size_t i = 0; i < sz; ++i)
			{
				keypoints[i].first = scale * 4 * keypoints[i].first;
				keypoints[i].second = scale *(4 * keypoints[i].second - top); 
			}
		}
	}
	else {
		//高宽比符合256 ; 192  直接放缩
		size_t sz = keypoints.size();
		float scale = float(srch) / float(resizedh);
			cout<<"srch:"<<srch<<endl;
			cout<<"resizedh:"<<resizedh<<endl;
			cout<<"scale:"<<scale<<endl;
		for (size_t i = 0; i < sz; ++i)
		{
			keypoints[i].first = scale * 4 * keypoints[i].first;
			keypoints[i].second = scale * 4 * keypoints[i].second;
		}
	}
}

void HRNet::draw(cv::Mat& dstimg) {
	if(dstimg.rows == 256 && dstimg.cols == 192)
	{
		//rng(12345)����α����������������,12345����������ӣ����԰�ϵͳʱ����Ϊ���������ʵ�������: RNG rng((unsigned)time(NULL))
		cv::RNG rng((unsigned)time(NULL));
		size_t sz = keypoints.size();
		for (size_t i = 0; i < sz; ++i)
		{

			int colorR = rng.uniform(0, 255);
			int colorG = rng.uniform(0, 255);
			int colorB = rng.uniform(0, 255);

			//��ֱ��std::to_string(),����İ취����ֻ������λС��
			float score = round(scores[i] * 100) / 100;//��λС��֮������ֱ�Ϊ0
			std::ostringstream oss;//����ɾ���������ַ����������
			oss << score;//������oss.str()����
			cv::circle(dstimg, cv::Point(keypoints[i].first, keypoints[i].second),2, cv::Scalar(colorR, colorG, colorB));
			putText(dstimg, class_names[i] + " " + oss.str(), cv::Point(keypoints[i].first, keypoints[i].second), cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(colorR, colorG, colorB), 0.25);
			line(dstimg, cv::Point(keypoints[5].first, keypoints[5].second), cv::Point(keypoints[6].first, keypoints[6].second), cv::Scalar(colorR, colorG, colorB), 3);//左右肩
			line(dstimg, cv::Point(keypoints[5].first, keypoints[5].second), cv::Point(keypoints[7].first, keypoints[7].second), cv::Scalar(colorR, colorG, colorB), 3);//左肩肘
			line(dstimg, cv::Point(keypoints[6].first, keypoints[6].second), cv::Point(keypoints[8].first, keypoints[8].second), cv::Scalar(colorR, colorG, colorB), 3);//右肩肘
			line(dstimg, cv::Point(keypoints[7].first, keypoints[7].second), cv::Point(keypoints[9].first, keypoints[9].second), cv::Scalar(colorR, colorG, colorB), 3);//左肘腕
			// line(dstimg, cv::Point(keypoints[5].first, keypoints[5].second), cv::Point(keypoints[11].first, keypoints[11].second), cv::Scalar(colorR, colorG, colorB), 3);//左肩胯
			// line(dstimg, cv::Point(keypoints[6].first, keypoints[6].second), cv::Point(keypoints[12].first, keypoints[12].second), cv::Scalar(colorR, colorG, colorB), 3);//右肩胯
			// line(dstimg, cv::Point(keypoints[11].first, keypoints[11].second), cv::Point(keypoints[12].first, keypoints[12].second), cv::Scalar(colorR, colorG, colorB), 3);//左右胯
		}
	}
	else
	{
		//rng(12345)����α����������������,12345����������ӣ����԰�ϵͳʱ����Ϊ���������ʵ�������: RNG rng((unsigned)time(NULL))
		cv::RNG rng((unsigned)time(NULL));
		size_t sz = keypoints.size();
		for (size_t i = 0; i < sz; ++i)
		{
			int colorR = rng.uniform(0, 255);
			int colorG = rng.uniform(0, 255);
			int colorB = rng.uniform(0, 255);

			//��ֱ��std::to_string(),����İ취����ֻ������λС��
			float score = round(scores[i] * 100) / 100;//��λС��֮������ֱ�Ϊ0
			std::ostringstream oss;//����ɾ���������ַ����������
			oss << score;//������oss.str()����
			cv::circle(dstimg, cv::Point(keypoints[i].first, keypoints[i].second),2, cv::Scalar(colorR, colorG, colorB), 3);
			putText(dstimg, class_names[i] + " " + oss.str(), cv::Point(keypoints[i].first, keypoints[i].second), cv::FONT_HERSHEY_SIMPLEX, 0.25, cv::Scalar(colorR, colorG, colorB), 0.25);
			line(dstimg, cv::Point(keypoints[5].first, keypoints[5].second), cv::Point(keypoints[6].first, keypoints[6].second), cv::Scalar(colorR, colorG, colorB), 3);//左右肩膀
			line(dstimg, cv::Point(keypoints[5].first, keypoints[5].second), cv::Point(keypoints[7].first, keypoints[7].second), cv::Scalar(colorR, colorG, colorB), 3);//左肩肘
			line(dstimg, cv::Point(keypoints[6].first, keypoints[6].second), cv::Point(keypoints[8].first, keypoints[8].second), cv::Scalar(colorR, colorG, colorB), 3);//右肩肘
			line(dstimg, cv::Point(keypoints[7].first, keypoints[7].second), cv::Point(keypoints[9].first, keypoints[9].second), cv::Scalar(colorR, colorG, colorB), 3);//左肘腕
			line(dstimg, cv::Point(keypoints[8].first, keypoints[8].second), cv::Point(keypoints[10].first, keypoints[10].second), cv::Scalar(colorR, colorG, colorB), 3);//右肘腕
			// line(dstimg, cv::Point(keypoints[5].first, keypoints[5].second), cv::Point(keypoints[11].first, keypoints[11].second), cv::Scalar(colorR, colorG, colorB), 3);//左肩胯
			// line(dstimg, cv::Point(keypoints[6].first, keypoints[6].second), cv::Point(keypoints[12].first, keypoints[12].second), cv::Scalar(colorR, colorG, colorB), 3);//右肩胯
			// line(dstimg, cv::Point(keypoints[11].first, keypoints[11].second), cv::Point(keypoints[12].first, keypoints[12].second), cv::Scalar(colorR, colorG, colorB), 3);//左右胯
		}
	}	
}

void HRNet::clear()
{
	//清理上一张图片的结果
	scores.clear();
}

std::pair<int, int> HRNet::Keypoint(const int& i)
{
	return this->keypoints[i];
}
