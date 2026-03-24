#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "src/yolo/openvino_yolo11_det_inference.hpp"
#include "src/ByteTrack/BYTETracker.h"

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
    auto detect_results = inference.predict(image, confidence_threshold, NMS_threshold);
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
    yolo::draw_detected_object(draw_image, detect_results);

    // Display the image with the detections
    cv::imshow("draw_image", draw_image);
    cv::waitKey(0);

    return 0;
}

int predict_video()
{
    // 请根据实际情况修改视频路径
    std::string video_path = "../../../videos/traffic monitor-1.mp4";
    std::string output_path = "../../../videos/traffic monitor-1 predict.mp4";
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
        auto detect_results = inference.predict(frame, confidence_threshold, NMS_threshold);
        std::cout << "detect_results size: " << detect_results.size() << std::endl;

        // 将检测框绘制到当前帧上
        yolo::draw_detected_object(frame, detect_results);

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

int track_video()
{
    // 请根据实际情况修改视频路径
    std::string video_path = "../../../videos/traffic monitor-1.mp4";
    std::string output_path = "../../../videos/traffic monitor track.mp4";
    std::string model_path = "../../../models/yolo11s.onnx";

    if (!fs::exists(model_path))
    {
        std::cout << "model_path: " << model_path << " not exist" << std::endl;
        return -1;
    }

    // 1. 初始化 YOLO 推理
    const float confidence_threshold = 0.05;
    const float NMS_threshold = 0.5;
    yolo::OpenvinoYolo11DetInference inference = {model_path, cv::Size(640, 640)};

    // 2. 初始化追踪器
    // max_time_lost 死亡倒计时，目标丢失（未匹配到检测框）后，在内存中保留等待重新出现的总帧数
    int max_time_lost = 60;
    // track_high_thresh 高分界线，得分大于此值的框为“高分框”，参与第一轮常规匹配。
    float track_high_thresh = 0.3;
    // track_low_thresh 低分界线，得分在此值与高分界线之间的为“低分框”，参与第二轮遮挡修补；低于此值的框直接丢弃。
    float track_low_thresh = 0.1;
    // new_track_thresh 出生门槛，只有得分大于此值的检测框，才能被初始化为全新的追踪目标。
    float new_track_thresh = 0.3;
    // match_thresh 认亲标准，判定检测框与已有轨迹“是否为同一目标”的匹配代价容忍度（通常基于 IoU）。
    float match_thresh = 0.8;
    ByteTrack::BYTETracker tracker(
        max_time_lost,
        track_high_thresh,
        track_low_thresh,
        new_track_thresh,
        match_thresh);
    std::vector<ByteTrack::Object> track_objects;
    std::vector<ByteTrack::STrack> tracklets;
    std::vector<ByteTrack::STrack> lostTracklets;

    // 3. 打开输入视频
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

    // 4. 初始化 VideoWriter (写入视频文件)
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
        auto detect_results = inference.predict(frame, confidence_threshold, NMS_threshold);
        std::cout << "detect_results size: " << detect_results.size() << std::endl;

        // 追踪
        track_objects.clear();
        tracklets.clear();
        lostTracklets.clear();
        int target_id = 0;
        for (const auto &detect_result : detect_results)
        {
            ByteTrack::Object obj;
            obj.target_id = target_id;
            obj.class_id = detect_result.class_id;
            obj.prob = detect_result.confidence;
            obj.rect = cv::Rect_<float>(
                static_cast<float>(detect_result.left),
                static_cast<float>(detect_result.top),
                static_cast<float>(detect_result.right - detect_result.left),
                static_cast<float>(detect_result.bottom - detect_result.top));
            target_id += 1;
            track_objects.push_back(obj);
        }

        // Tracking
        tracker.update(track_objects, lostTracklets, tracklets);
        std::cout << "tracklets size: " << tracklets.size() << std::endl;
        std::cout << "lostTracklets size: " << lostTracklets.size() << std::endl;

        std::vector<yolo::YoloDetectResult> detect_results1;
        for (const auto &tracklet : tracklets)
        {
            // 创建新数据
            yolo::YoloDetectResult result;
            result.track_id = tracklet.track_id;
            result.class_id = tracklet.class_id;
            result.class_name = inference._classes[result.class_id];
            result.confidence = tracklet.score;
            result.left = tracklet.tlwh[0];
            result.top = tracklet.tlwh[1];
            result.right = tracklet.tlwh[0] + tracklet.tlwh[2];
            result.bottom = tracklet.tlwh[1] + tracklet.tlwh[3];

            detect_results1.push_back(result);
        }

        // 将检测框绘制到当前帧上
        yolo::draw_detected_object(frame, detect_results1);

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

        std::cout << std::endl;
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
    // int res = track_video();
    if (res != 0)
        return -1;

    return 0;
}
