// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef YOLO11_OPENVINO_INFER_HPP
#define YOLO11_OPENVINO_INFER_HPP

#include <string>
#include <vector>
#include <array>
#include <random>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace yolo
{

    struct YoloDetectResult
    {
        int class_id;
        std::string class_name;
        float confidence;
        int left;
        int top;
        int right;
        int bottom;
    };

    class OpenvinoInference
    {
    public:
        cv::Size2f _model_input_shape; // Input shape of the model
        cv::Size _model_output_shape;  // Output shape of the model

        ov::InferRequest _inference_request; // OpenVINO inference request
        ov::CompiledModel _compiled_model;   // OpenVINO compiled model

        std::vector<std::string> _classes{
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"};

        void InitializeModel(const std::string &model_path)
        {
            ov::Core core;                                                  // OpenVINO core object
            std::shared_ptr<ov::Model> model = core.read_model(model_path); // Read the model from file

            // If the model has dynamic shapes, reshape it to the specified input shape
            if (model->is_dynamic())
            {
                model->reshape({1, 3, static_cast<long int>(this->_model_input_shape.height), static_cast<long int>(this->_model_input_shape.width)});
            }

            // Preprocessing setup for the model
            ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
            // ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
            ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::RGB);
            // ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255, 255, 255});
            ppp.input().preprocess().convert_element_type(ov::element::f32).scale({255, 255, 255});
            ppp.input().model().set_layout("NCHW");
            ppp.output().tensor().set_element_type(ov::element::f32);
            model = ppp.build(); // Build the preprocessed model

            // Compile the model for inference
            this->_compiled_model = core.compile_model(model, "AUTO");
            this->_inference_request = this->_compiled_model.create_infer_request(); // Create inference request

            int width, height;

            // Get input shape from the model
            const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
            const ov::Shape input_shape = inputs[0].get_shape();
            height = input_shape[1];
            width = input_shape[2];
            this->_model_input_shape = cv::Size2f(width, height);

            // Get output shape from the model
            const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
            const ov::Shape output_shape = outputs[0].get_shape();
            height = output_shape[1];
            width = output_shape[2];
            this->_model_output_shape = cv::Size(width, height);
        }

        // Constructor to initialize the model with specified input shape
        OpenvinoInference(const std::string &model_path, const cv::Size model_input_shape)
        {
            this->_model_input_shape = model_input_shape;
            InitializeModel(model_path);
        }

        std::vector<YoloDetectResult> RunInference(
            cv::Mat &image,
            const float confidence_threshold = 0.25,
            const float NMS_threshold = 0.5
        )
        {
            float scale_factor = Preprocessing(image);          // Preprocess the input image
            this->_inference_request.infer();                   // Run inference
            auto detect_results = PostProcessing(
                confidence_threshold,
                NMS_threshold,
                scale_factor,
                image.size()
            ); // Postprocess the inference results
            return detect_results;
        }

        // Method to preprocess the input image
        float Preprocessing(const cv::Mat &image)
        {
            float scale_factor = std::min(
                this->_model_input_shape.width / static_cast<float>(image.cols),
                this->_model_input_shape.height / static_cast<float>(image.rows));
            int new_width = static_cast<int>(image.cols * scale_factor);
            int new_height = static_cast<int>(image.rows * scale_factor);

            cv::Mat resized_image;
            // 缩放高宽的长边为最大长度
            cv::resize(image, resized_image, {new_width, new_height}, 0, 0, cv::INTER_AREA); // Resize the image to imagech the model input shape

            std::cout << "resized_image size: " << resized_image.size() << std::endl;
            // cv::imshow("resized_image", resized_image);
            // cv::waitKey(0);

            // 填充bottom和right的长度
            int delta_w = this->_model_input_shape.width - new_width;
            int delta_h = this->_model_input_shape.height - new_height;
            cv::copyMakeBorder(
                resized_image,
                resized_image,
                0,
                delta_h,
                0,
                delta_w,
                cv::BORDER_CONSTANT,
                cv::Scalar(0, 0, 0));

            std::cout << "copyMakeBorder image size: " << resized_image.size() << std::endl;
            // cv::imshow("resized_image", resized_image);
            // cv::waitKey(0);

            ov::Tensor input_tensor = this->_inference_request.get_input_tensor();
            uint8_t *input_data = input_tensor.data<uint8_t>();
            size_t bytes_to_copy = resized_image.total() * resized_image.elemSize();
            memcpy(input_data, resized_image.data, bytes_to_copy);

            this->_inference_request.set_input_tensor(input_tensor); // Set input tensor for inference

            return scale_factor;
        }

        // Method to postprocess the inference results
        std::vector<YoloDetectResult> PostProcessing(
            const float confidence_threshold,
            const float NMS_threshold,
            const float scale_factor,
            const cv::Size original_shape
        )
        {
            std::vector<int> class_list;
            std::vector<float> confidence_list;
            std::vector<cv::Rect> box_list;
            std::vector<cv::Rect> nms_box_list; // 【新增】专门用于 NMS 的带偏移量框列表

            // 1. 获取原始的 84 x 8400 矩阵
            // 8400 列：代表模型预测出的 8400 个候选框（Anchors）。
            // 84 行：代表每个候选框的 84 个属性。前 4 个是中心点坐标和宽高 (cx,cy,w,h)，后 80 个是该框属于 80 个类别的得分。
            const float *detections = this->_inference_request.get_output_tensor().data<const float>();
            const cv::Mat detection_outputs(this->_model_output_shape, CV_32F, (float *)detections); // Create OpenCV imagerix from output tensor

            // 【新增】设定一个足够大的常量作为偏移基数 (通常 YOLO 输入是 640 或 1280，4096 绝对够用)
            const int max_wh = 4096;

            // 2. 将其转置为 8400 x 84 (行数变成 8400，列数变成 84)
            cv::Mat transposed_outputs;
            cv::transpose(detection_outputs, transposed_outputs);

            // Iterate over detections and collect class IDs, confidence scores, and bounding boxes
            // 循环 8400 次，每次处理一列（即一个预测框）
            for (int i = 0; i < transposed_outputs.rows; ++i)
            {
                // transposed_outputs.row(i)：取第 i 个预测框的所有数据（84个值）。
                // .colRange(4, transposed_outputs.cols)：跳过前 4 个位置坐标参数，截取第 4 到第 83 行。这 80 个值就是该框在 80 个分类上的概率得分。
                const cv::Mat _classesscores = transposed_outputs.row(i).colRange(4, transposed_outputs.cols);

                // 找出得分最高的值赋给 score，并把该得分的**索引（类别 ID）**赋给 class_id.y
                cv::Point class_id;
                double score;
                cv::minMaxLoc(_classesscores, nullptr, &score, nullptr, &class_id); // Find the class with the highest score

                // Check if the detection meets the confidence threshold
                if (score > confidence_threshold)
                {
                    class_list.push_back(class_id.y);
                    confidence_list.push_back(score);

                    const float cx = transposed_outputs.at<float>(i, 0);
                    const float cy = transposed_outputs.at<float>(i, 1);
                    const float w = transposed_outputs.at<float>(i, 2);
                    const float h = transposed_outputs.at<float>(i, 3);

                    // 1. 保存原始真实的预测框（用于最终结果提取和绘制）
                    cv::Rect box;
                    box.x = static_cast<int>((cx - w / 2));
                    box.y = static_cast<int>((cy - h / 2));
                    box.width = static_cast<int>(w);
                    box.height = static_cast<int>(h);
                    box_list.push_back(box);

                    // 2. 【新增】生成带偏移量的框（仅用于送入 NMS 函数）
                    cv::Rect nms_box;
                    nms_box.x = box.x + class_id.y * max_wh;
                    nms_box.y = box.y + class_id.y * max_wh;
                    nms_box.width = box.width;
                    nms_box.height = box.height;
                    nms_box_list.push_back(nms_box);
                }
            }

            // Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
            std::vector<int> NMS_result;
            // 【核心修改】这里传入的是带偏移量的 nms_box_list，而不是原来的 box_list
            cv::dnn::NMSBoxes(nms_box_list, confidence_list, confidence_threshold, NMS_threshold, NMS_result);

            std::vector<YoloDetectResult> results;
            // Collect final detections after NMS
            for (int i = 0; i < NMS_result.size(); ++i)
            {
                YoloDetectResult result;
                int id = NMS_result[i];

                result.class_id = class_list[id];
                result.class_name = this->_classes[result.class_id];
                result.confidence = confidence_list[id];

                // 【注意】这里依然使用未加偏移量的真实 box_list[id] 来还原坐标
                auto box = GetBoundingBox(box_list[id], scale_factor, original_shape);
                result.left = box[0];
                result.top = box[1];
                result.right = box[2];
                result.bottom = box[3];

                results.push_back(result);
            }
            return results;
        }

        // Method to get the bounding box in the correct scale
        std::array<int, 4> GetBoundingBox(const cv::Rect &src, const float scale_factor, const cv::Size& original_shape)
        {
            // 1. 将基于 640x640 尺度的坐标和宽高，按比例还原回原图尺度
            int left = static_cast<int>(src.x / scale_factor);
            int top = static_cast<int>(src.y / scale_factor);
            int width = static_cast<int>(src.width / scale_factor);
            int height = static_cast<int>(src.height / scale_factor);

            // 2. 构建还原后的预测框
            cv::Rect scaled_box(left, top, width, height);

            // 3. 构建代表原始图像物理边界的框
            cv::Rect image_rect(0, 0, original_shape.width, original_shape.height);

            // 4. 【核心安全机制】求交集，自动切除越界部分
            cv::Rect safe_box = scaled_box & image_rect;

            // 5. 提取安全截断后的左上角和右下角坐标
            int safe_left = safe_box.x;
            int safe_top = safe_box.y;
            int safe_right = safe_box.x + safe_box.width;
            int safe_bottom = safe_box.y + safe_box.height;

            return {safe_left, safe_top, safe_right, safe_bottom};
        }


        cv::Mat DrawDetectedObject(cv::Mat &image, const std::vector<YoloDetectResult> &detect_results)
        {
            for (const auto &result : detect_results)
            {
                const float &confidence = result.confidence;
                const int &class_id = result.class_id;

                // Generate a random color for the bounding box
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<int> dis(120, 255);
                const cv::Scalar &color = cv::Scalar(dis(gen), dis(gen), dis(gen));

                // Draw the bounding box around the detected object
                cv::rectangle(image, cv::Point(result.left, result.top), cv::Point(result.right, result.bottom), color, 2);

                // Prepare the class label and confidence text
                std::string classString = _classes[class_id] + " " + std::to_string(confidence).substr(0, 4);

                // Get the size of the text box
                cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 1, 0);
                int top;
                if (result.top > textSize.height + 5)
                    top = result.top - textSize.height - 10;
                else
                    top = result.top + textSize.height + 10;

                // xmin + t_w, ymin

                // Draw the text box
                cv::rectangle(image, {result.left, top}, {result.left + textSize.width, result.top}, color, cv::FILLED);

                // Put the class label and confidence text above the bounding box
                // ymin - 5 if ymin > t_h + 5 else ymin + t_h + 5
                if (result.top > textSize.height + 5)
                    top = result.top - 5;
                else
                    top = result.top + textSize.height + 5;
                cv::putText(image, classString, {result.left, top}, cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 1, 0);
            }
            return image;
        }
    };

} // namespace yolo

#endif // YOLO11_OPENVINO_INFER_HPP
