#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "src/yolo/openvino_yolo11_det_inference.hpp"

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    std::cout << "argc: " << argc << std::endl;
    std::cout << "program name: " << argv[0] << std::endl;

    std::string image_path = "../../../images/bus.jpg";
    if (!fs::exists(image_path))
    {
        std::cout << "image_path: " << image_path << "not exist" << std::endl;
        return -1;
    }

    std::string model_path = "../../../models/yolo11n.onnx";
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
    inference.DrawDetectedObject(draw_image, detect_results);

    // Display the image with the detections
    cv::imshow("draw_image", draw_image);
    cv::waitKey(0);

    return 0;
}
