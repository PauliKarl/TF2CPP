#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include<ctime>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <direct.h>
#include <io.h>

#include "tinyxml.h"


using namespace cv;
using namespace dnn;
using namespace std;

//**********************Initialize the parameters***************************//
float confThreshold = 0.5F; // Confidence threshold
float maskThreshold = 0.3F; // Mask threshold

vector<string> classes;
vector<Scalar> colors = { Scalar(255, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0) };

//**************************��������**************************//
// Draw the predicted bounding box
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask);

// Postprocess the neural network's output for each frame
void postprocess(Mat& img_copy, Mat& frame, const vector<Mat>& outs, string save_dir, string szFname, string img_name);


//********************************************main()***********************************//
int main()
{
	//**********Load names of classes********************//
	string classesFile = "G:/tf_obj/labels.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	//*************Load the colors***********************//
	//string colorsFile = "./mask_rcnn_inception_v2_coco_2018_01_28/colors.txt";
	//ifstream colorFptr(colorsFile.c_str());
	//while (getline(colorFptr, line))
	//{
	//	char* pEnd;
	//	double r, g, b;
	//	r = strtod(line.c_str(), &pEnd);
	//	g = strtod(pEnd, NULL);
	//	b = strtod(pEnd, NULL);
	//	Scalar color = Scalar(r, g, b, 255.0);
	//	colors.push_back(Scalar(r, g, b, 255.0));
	//}

	// Give the configuration and weight files for the model
	String textGraph = "G:/tf_obj/training/output_inference/graph.pbtxt";
	String modelWeights = "G:/tf_obj/training/output_inference/frozen_inference_graph.pb";

	// Load the network
	Net net = readNetFromTensorflow(modelWeights, textGraph);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open a video file or an image file or a camera stream.
	string str, outputFile;
	//VideoWriter video;
	Mat frame, blob;


	clock_t startTime = clock();


	//***************���������ļ�·��******************//
	const char *file, *save_dir;
	file = "G:/qt_test/icon.png";

	//�����ļ�·���ֽ�
	char szDrive[5];   //������
	char szDir[50];       //·����
	char szFname[50];   //�ļ���
	char szExt[10];       //��׺��
	_splitpath_s(file, szDrive, szDir, szFname, szExt); //�ֽ�·��

	//��ȡ����ͼƬ
	frame = imread(file);

	//��������ͼ��ĸ���
	Mat img_copy;
	frame.copyTo(img_copy);


	//***********���ý���ļ��洢·��**************//
	save_dir = "G:/qt_test/";


	//***************Stop the program if reached end of video*********************//
	if (frame.empty())
	{
		cout << "Done processing !!!" << endl;
		cout << "Output file is stored as " << outputFile << endl;
		waitKey(3000);
	}
	//*************************Create a 4D blob from a frame.**********************//
	//blobFromImage(frame, blob, 1.0, Size(512, 512), Scalar(), true, false);
	blobFromImage(frame, blob);
	//std::cout << blob.size << std::endl;

	//**********Sets the input to the network****************//
	net.setInput(blob);

	//*****************Runs the forward pass to get output from the output layers********//
	std::vector<String> outNames(2);
	outNames[0] = "detection_out_final";
	outNames[1] = "detection_masks";
	vector<Mat> outs;
	net.forward(outs, outNames);

	//***********Extract the bounding box and mask for each of the detected objects*************//
	postprocess(img_copy, frame, outs, save_dir, szFname, szExt);

	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	//string label = format("Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms", t);
	//putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

	//**********Write the frame with the detection boxes****************//
	Mat detectedFrame;
	frame.convertTo(detectedFrame, CV_8U);
	clock_t endTime = clock();
	cout << "����������ʱ��" << double(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	
	//***************************Create a window ��ʾframe*****************************//
	//static const string kWinName = "Deep learning object detection in OpenCV";
	//namedWindow(kWinName, WINDOW_NORMAL);
	//imshow(kWinName, frame);
	//waitKey(0);

	return 0;
}

// For each frame, extract the bounding box and mask for each detected object
void postprocess(Mat& img_copy, Mat& frame, const vector<Mat>& outs, string save_dir, string szFname, string szExt)
{	
	//************parameters****************************//
	//	img_copy: ����ͼƬ�ĸ��������ڲü����Ŀ��		//
	//	frame: ����ͼ�����ڼ������չʾ				//
	//	outs: mask_rcnn����������������boxes ��masks	//
	//	save_dir: ����ļ������Ŀ¼					//
	//	szFname: ����ͼ����ļ���						//
	//	szExt: ����ͼ��ĺ�׺������".png"				//
	//**************************************************//

	//********************����xml�ļ��洢·��************************//
	char xml_filename[40];
	string format_rls = ".xml";
	string strn;
	strn = save_dir;
	strn += szFname;
	strn += format_rls;
	strcpy_s(xml_filename, strn.c_str());



	Mat outDetections = outs[0];
	Mat outMasks = outs[1];

	//******************Output size of masks is NxCxHxW ****************//
	// N - number of detected boxes
	// C - number of classes (excluding background)
	// HxW - segmentation shape
	const int numDetections = outDetections.size[2];
	const int numClasses = outMasks.size[1];

	//cout << numDetections << endl;

	outDetections = outDetections.reshape(1, (int)outDetections.total() / 7);

	//****************************����xml�ļ�*****************************//	
	TiXmlDocument *writeDoc = new TiXmlDocument; //xml�ĵ�ָ��

	//�ĵ���ʽ����
	TiXmlDeclaration *decl = new TiXmlDeclaration("1.0", "UTF-8", "yes");
	writeDoc->LinkEndChild(decl); //д���ĵ�

	int n = numDetections;	//���ڵ����

	TiXmlElement *RootElement = new TiXmlElement("Annotaion");//��Ԫ��
	RootElement->SetAttribute("obj_num", n); //���ԣ�Ŀ���ܸ���
	writeDoc->LinkEndChild(RootElement);

	TiXmlElement *imgnameElement = new TiXmlElement("filename");//��Ԫ�أ�ͼ���ļ���
	RootElement->LinkEndChild(imgnameElement);

	// ͼ������ȡ
	char img_name[50];
	string str_img;
	str_img = szFname;
	str_img += szExt;
	strcpy_s(img_name, str_img.c_str());
	TiXmlText *img_nameContent = new TiXmlText(img_name);
	imgnameElement->LinkEndChild(img_nameContent);


	//*********************************����Ŀ����ȡ******************************//
	for (int i = 0; i < numDetections; ++i)
	{
		float score = outDetections.at<float>(i, 2);
		if (score > confThreshold)
		{
			//Extract the bounding box�� ���ο�
			int classId = static_cast<int>(outDetections.at<float>(i, 1));
			int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
			int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
			int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
			int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

			left = max(0, min(left, frame.cols - 1));
			top = max(0, min(top, frame.rows - 1));
			right = max(0, min(right, frame.cols - 1));
			bottom = max(0, min(bottom, frame.rows - 1));
			Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

			int height = bottom - top;
			int width = right - left;

			cout << "obj_id" << i << endl;

			//****Ŀ����������Ƭ����*********//
			String obj_save_dir = save_dir + "ships/";
			if (_access(obj_save_dir.c_str(), 0) == -1)
				_mkdir(obj_save_dir.c_str());
			String filename = obj_save_dir + "ObjectId_" + to_string(static_cast<int>(i+1)) + ".jpg";
			imwrite(filename, img_copy(box));

			//Extract the mask for the object
			Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));

			//*************Draw bounding box, colorize and show the mask on the image******************//
			//drawBox(frame, classId, score, box, objectMask);


			//************************�������Ϣд��xml�ļ�*******************************//
			TiXmlElement *ObjElement = new TiXmlElement("object");//object
			//��������
			//ObjElement->SetAttribute("class", "A");
			ObjElement->SetAttribute("id", i + 1);
			//ObjElement->SetAttribute("flag", (i + 1) * 10);
			RootElement->LinkEndChild(ObjElement);//���ڵ�д���ĵ�

			//Ŀ������
			TiXmlElement *nameElement = new TiXmlElement("name");
			ObjElement->LinkEndChild(nameElement);

			char fnam[50];
			strcpy_s(fnam, filename.c_str());
			TiXmlText *nameContent = new TiXmlText(fnam);
			nameElement->LinkEndChild(nameContent);

			//���Ŷȷ���
			TiXmlElement *scoreElement = new TiXmlElement("score");
			ObjElement->LinkEndChild(scoreElement);

			char scr[10];
			sprintf_s(scr, "%f", score);
			TiXmlText *scoreContent = new TiXmlText(scr);
			scoreElement->LinkEndChild(scoreContent);

			//Ŀ��λ��bbox
			TiXmlElement *bboxElement = new TiXmlElement("bbox");
			ObjElement->LinkEndChild(bboxElement);

			// cx
			TiXmlElement *cxElement = new TiXmlElement("cx");
			bboxElement->LinkEndChild(cxElement);

			char cxx[10];
			sprintf_s(cxx, "%d", left + width /2 );
			TiXmlText *cxContent = new TiXmlText(cxx);
			cxElement->LinkEndChild(cxContent);

			// cy
			TiXmlElement *cyElement = new TiXmlElement("cy");
			bboxElement->LinkEndChild(cyElement);

			char cyy[10];
			sprintf_s(cyy, "%d", top + height / 2);
			TiXmlText *cyContent = new TiXmlText(cyy);
			cyElement->LinkEndChild(cyContent);

			// w
			TiXmlElement *wElement = new TiXmlElement("w");
			bboxElement->LinkEndChild(wElement);

			char ww[10];
			sprintf_s(ww, "%d", width);
			TiXmlText *wContent = new TiXmlText(ww);
			wElement->LinkEndChild(wContent);

			// h
			TiXmlElement *hElement = new TiXmlElement("h");
			bboxElement->LinkEndChild(hElement);

			char hh[10];
			sprintf_s(hh, "%d", height);
			TiXmlText *hContent = new TiXmlText(hh);
			hElement->LinkEndChild(hContent);
		}
	}
	writeDoc->SaveFile(xml_filename);
	delete writeDoc;
}


// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
	//************************parameters****************************//
	//	frame: ����ͼ������չʾ�����							//
	//	classId: Ŀ�����											//
	//	conf: Ŀ�����Ŷ�											//
	//	box: Ŀ��ľ��ο�����										//
	//	objectMask: Ŀ��mask										//
	//**************************************************************//

	//*******Draw a rectangle displaying the bounding box*****************//
	rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

	//**********Get the label for the class name and its confidence*********************//
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//******************Display the label at the top of the bounding box********************//
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	box.y = max(box.y, labelSize.height);
	rectangle(frame, Point(box.x, box.y - (int)round(1.5*labelSize.height)), Point(box.x + (int)round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

	Scalar color = colors[classId];

	//std::cout << objectMask.size<< std::endl;

	//******************Resize the mask, threshold, color and apply it on the image*************************//
	resize(objectMask, objectMask, Size(box.width, box.height));
	Mat mask = (objectMask > maskThreshold);

	Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
	coloredRoi.convertTo(coloredRoi, CV_8UC3);

	//***************Draw the contours on the image**********************//
	vector<Mat> contours;
	Mat hierarchy;
	mask.convertTo(mask, CV_8U);

	findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
	coloredRoi.copyTo(frame(box), mask);

	//***********��ȡ��ת��ͻ�ȡ�Ƕ�*****************//
	//RotatedRect rRect = minAreaRect(contours[0]);
	//std::cout << rRect.angle<<rRect.center.x << rRect.center.y<< std::endl;
}