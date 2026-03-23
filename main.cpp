#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "src/yolo/openvino_yolo11_det_inference.hpp"

namespace fs = std::filesystem;

int predict_image()
{
    std::string image_path = "../../../images/bus.jpg";
    if (!fs::exists(image_path))
    {
        std::cout << "image_path: " << image_path << "not exist" << std::endl;
        return -1;
    }

    std::string model_path = "../../../models/yolo11s.onnx";
    if (!fs::exists(model_path))
    {
        std::cout << "model_path: " << model_path << "not exist" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cout << "image_path: " << image_path << "read failed" << std::endl;
        return -1;
    }
    std::cout << "image size: " << image.size() << std::endl;
    // cv::imshow("image", image);
    // cv::waitKey(0);

    // Define the confidence and NMS thresholds
    const float confidence_threshold = 0.25;
    const float NMS_threshold = 0.5;

    // Initialize the YOLO inference with the specified model and parameters
    yolo::OpenvinoYolo11DetInference inference = {model_path, cv::Size(640, 640)};

    // Run inference on the input image
    auto detect_results = inference.RunInference(image, confidence_threshold, NMS_threshold);
    std::cout << "detect_results num = " << detect_results.size() << std::endl;
    std::cout << "detect_results:" << std::endl;
    for (const auto &detect_result : detect_results)
    {
        std::cout << "    class_id: " << detect_result.class_id
                  << ", class_name: " << detect_result.class_name
                  << ", confidence: " << detect_result.confidence
                  << ", box : [" << detect_result.left
                  << ", " << detect_result.top
                  << ", " << detect_result.right
                  << ", " << detect_result.bottom << "]" << std::endl;
    }

    cv::Mat draw_image = image.clone();
    yolo::DrawDetectedObject(draw_image, detect_results);

    // Display the image with the detections
    cv::imshow("draw_image", draw_image);
    cv::waitKey(0);

    return 0;
}


int predict_video()
{
    // 请根据实际情况修改视频路径
    std::string video_path = "../../../videos/traffic monitor.mp4";
    std::string output_path = "../../../videos/traffic monitor predict.mp4";
    std::string model_path = "../../../models/yolo11s.onnx";

    if (!fs::exists(model_path))
    {
        std::cout << "model_path: " << model_path << " not exist" << std::endl;
        return -1;
    }

    // 1. 初始化 YOLO 推理
    const float confidence_threshold = 0.25;
    const float NMS_threshold = 0.5;
    yolo::OpenvinoYolo11DetInference inference = {model_path, cv::Size(640, 640)};

    // 2. 打开输入视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cout << "Error: Could not open video file: " << video_path << std::endl;
        return -1;
    }

    // 获取视频的基本属性，用于初始化 VideoWriter
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 3. 初始化 VideoWriter (写入视频文件)
    // 这里使用 mp4v 编码器生成 .mp4 格式的视频
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened())
    {
        std::cout << "Error: Could not open VideoWriter for: " << output_path << std::endl;
        return -1;
    }

    cv::Mat frame;
    int frame_count = 0;
    std::cout << "Start processing video..." << std::endl;

    // 4. 循环读取视频帧
    while (cap.read(frame))
    {
        if (frame.empty())
        {
            break; // 视频结束
        }

        // 进行目标检测
        auto detect_results = inference.RunInference(frame, confidence_threshold, NMS_threshold);

        // 将检测框绘制到当前帧上
        yolo::DrawDetectedObject(frame, detect_results);

        // 将处理后的帧写入输出视频文件
        writer.write(frame);

        // 实时显示处理过程 (可选)
        cv::imshow("Video Inference", frame);
        // 按下 ESC 键 (ASCII: 27) 可以提前终止处理，等待时间为1ms
        if (cv::waitKey(1) == 27)
        {
            std::cout << "Video processing interrupted by user." << std::endl;
            break;
        }

        frame_count++;
        if (frame_count % 30 == 0)
        {
            std::cout << "Processed " << frame_count << " frames..." << std::endl;
        }
    }

    // 5. 释放资源
    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "Video processing complete. Output saved to: " << output_path << std::endl;
    return 0;
}


int main(int argc, char *argv[])
{
    std::cout << "argc: " << argc << std::endl;
    std::cout << "program name: " << argv[0] << std::endl;

    int res = predict_image();
    // int res = predict_video();
    if (res != 0)
        return -1;

    return 0;
}
