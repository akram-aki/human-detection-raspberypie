#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <net.h>
#include <vector>
#include <algorithm> // For std::sort
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>

// --- CONFIGURATION ---
std::string RTSP_URL = "rtsp://6868:6868@172.19.1.181:554/cam/realmonitor?channel=7&subtype=0";
const float CONFIDENCE_THRESH = 0.25f; // Lowered to debug detection
const float NMS_THRESH = 0.45f;        // How much overlap is allowed
const int TARGET_SIZE = 640;           // YOLOv8 standard size

// --- LOGGING HELPERS ---
std::string getTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    struct tm t;
    localtime_s(&t, &time);
    ss << std::put_time(&t, "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

#define LOG_INFO(fmt, ...) printf("[%s] [INFO] " fmt "\n", getTimestamp().c_str(), ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) printf("[%s] [ERROR] " fmt "\n", getTimestamp().c_str(), ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) printf("[%s] [WARN] " fmt "\n", getTimestamp().c_str(), ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) printf("[%s] [DEBUG] " fmt "\n", getTimestamp().c_str(), ##__VA_ARGS__)

// A simple structure to hold our detection results
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// --- HELPER FUNCTION: INTERSECTION OVER UNION (For NMS) ---
static inline float intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;
        while (faceobjects[j].prob < p)
            j--;
        if (i <= j)
        {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }
    if (left < j)
        qsort_descent_inplace(faceobjects, left, j);
    if (i < right)
        qsort_descent_inplace(faceobjects, i, right);
}
static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();
    const int n = (int)faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = faceobjects[i].rect.area();
    for (int i = 0; i < n; i++)
    {
        const Object &a = faceobjects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = faceobjects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

// --- CORE FUNCTION: DETECT PEOPLE ---
// This function takes a Frame and the Network, and returns a list of "People"
std::vector<Object> detect_people(const cv::Mat &bgr, ncnn::Net &yolov8)
{
    LOG_DEBUG("detect_people() called");

    int img_w = bgr.cols;
    int img_h = bgr.rows;
    LOG_DEBUG("Input image dimensions: %dx%d", img_w, img_h);

    // 1. Prepare Input (Resize & Normalize)
    LOG_DEBUG("Preparing input: resizing to %dx%d and converting BGR to RGB", TARGET_SIZE, TARGET_SIZE);
    auto prep_start = std::chrono::high_resolution_clock::now();

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, TARGET_SIZE, TARGET_SIZE);

    // YOLOv8 expects values 0.0 to 1.0 (so we divide by 255)
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(0, norm_vals);

    auto prep_end = std::chrono::high_resolution_clock::now();
    auto prep_duration = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_start);
    LOG_DEBUG("Input preparation completed in %lld microseconds", prep_duration.count());

    // 2. Run Inference
    LOG_DEBUG("Creating extractor and running inference");
    auto infer_start = std::chrono::high_resolution_clock::now();

    ncnn::Extractor ex = yolov8.create_extractor();
    ex.input("images", in); // 'images' is the input name for YOLOv8
    ncnn::Mat out;
    ex.extract("output0", out); // 'output0' is the output name

    auto infer_end = std::chrono::high_resolution_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start);
    LOG_DEBUG("Inference completed in %lld microseconds (%.2f ms)", infer_duration.count(), infer_duration.count() / 1000.0f);
    LOG_DEBUG("Output tensor dimensions: w=%d, h=%d, c=%d", out.w, out.h, out.c);

    // 3. Decode Output (The Hard Part)
    // YOLOv8 output shape is roughly [1, 84, 8400]
    // We need to iterate through all 8400 possibilities
    LOG_DEBUG("Decoding output: processing %d anchor points", out.w);
    auto decode_start = std::chrono::high_resolution_clock::now();

    std::vector<Object> proposals;

    const int num_anchors = out.w; // 8400
    // Access rows using pointers
    // Row 0=x, 1=y, 2=w, 3=h.
    // Row 4 is "Person" probability (Class 0)
    const float *cx_ptr = out.row(0);
    const float *cy_ptr = out.row(1);
    const float *w_ptr = out.row(2);
    const float *h_ptr = out.row(3);
    const float *person_score_ptr = out.row(4); // Only checking Class 0 (Person)

    int proposals_before_threshold = 0;
    float max_score_seen = 0.0f;
    for (int i = 0; i < num_anchors; i++)
    {
        float score = person_score_ptr[i];
        if (score > max_score_seen)
            max_score_seen = score;

        proposals_before_threshold++;

        if (score > CONFIDENCE_THRESH)
        {
            float cx = cx_ptr[i];
            float cy = cy_ptr[i];
            float w = w_ptr[i];
            float h = h_ptr[i];

            // Convert center-x-y to top-left-x-y
            float x0 = cx - w * 0.5f;
            float y0 = cy - h * 0.5f;

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = w;
            obj.rect.height = h;
            obj.label = 0;
            obj.prob = score;
            proposals.push_back(obj);
        }
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    auto decode_duration = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start);
    LOG_DEBUG("Decoded %d proposals from %d anchors (confidence threshold: %.2f) in %lld microseconds",
              (int)proposals.size(), proposals_before_threshold, CONFIDENCE_THRESH, decode_duration.count());
    LOG_DEBUG("Max score seen in this frame: %.4f", max_score_seen);

    // 4. Sort and Apply NMS (Remove duplicate boxes)
    LOG_DEBUG("Applying NMS (threshold: %.2f) to %d proposals", NMS_THRESH, (int)proposals.size());
    auto nms_start = std::chrono::high_resolution_clock::now();

    std::vector<int> picked;
    if (!proposals.empty())
    {
        qsort_descent_inplace(proposals, 0, (int)proposals.size() - 1);
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    }

    auto nms_end = std::chrono::high_resolution_clock::now();
    auto nms_duration = std::chrono::duration_cast<std::chrono::microseconds>(nms_end - nms_start);
    LOG_DEBUG("NMS completed: %d detections after filtering (removed %d duplicates) in %lld microseconds",
              (int)picked.size(), (int)proposals.size() - (int)picked.size(), nms_duration.count());

    LOG_DEBUG("Scaling bounding boxes from %dx%d to original image size %dx%d", TARGET_SIZE, TARGET_SIZE, img_w, img_h);
    float scale_x = (float)img_w / TARGET_SIZE;
    float scale_y = (float)img_h / TARGET_SIZE;
    LOG_DEBUG("Scale factors: x=%.3f, y=%.3f", scale_x, scale_y);

    std::vector<Object> objects;
    for (int i = 0; i < (int)picked.size(); i++)
    {
        Object obj = proposals[picked[i]];

        // Scale back to original image size
        obj.rect.x *= scale_x;
        obj.rect.y *= scale_y;
        obj.rect.width *= scale_x;
        obj.rect.height *= scale_y;

        // Fix potential out-of-bounds
        obj.rect.x = std::max<float>(obj.rect.x, 0.f);
        obj.rect.y = std::max<float>(obj.rect.y, 0.f);
        obj.rect.width = std::min<float>(obj.rect.width, (float)img_w - obj.rect.x);
        obj.rect.height = std::min<float>(obj.rect.height, (float)img_h - obj.rect.y);

        LOG_DEBUG("Detection %d: Person (%.1f%%) at [%.0f, %.0f, %.0fx%.0f]",
                  i + 1, obj.prob * 100.0f, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        objects.push_back(obj);
    }

    LOG_DEBUG("detect_people() returning %d final detections", (int)objects.size());
    return objects;
}

// --- MAIN FUNCTION ---
int main()
{
    LOG_INFO("========================================");
    LOG_INFO("Security Camera System Starting");
    LOG_INFO("========================================");

    LOG_INFO("Initializing OpenCV logging (set to WARNING level)");
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING); // Quiet logs

    LOG_INFO("Setting RTSP transport to TCP for stable connection");
    _putenv("OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp"); // Force TCP

    // 1. Load Model
    LOG_INFO("Step 1: Loading YOLOv8 NCNN Model");
    LOG_INFO("Model files: yolov8n.param, yolov8n.bin");
    LOG_INFO("Configuration: Confidence threshold=%.2f, NMS threshold=%.2f, Target size=%dx%d",
             CONFIDENCE_THRESH, NMS_THRESH, TARGET_SIZE, TARGET_SIZE);

    auto model_load_start = std::chrono::high_resolution_clock::now();

    ncnn::Net yolov8;
    yolov8.opt.use_vulkan_compute = false;
    LOG_DEBUG("Vulkan compute disabled, using CPU");

    LOG_INFO("Loading model parameter file: yolov8n.param");
    if (yolov8.load_param("yolov8n.param") != 0)
    {
        LOG_ERROR("Failed to load yolov8n.param - file not found or invalid");
        LOG_ERROR("Please ensure yolov8n.param is in the current working directory");
        return -1;
    }
    LOG_INFO("Model parameter file loaded successfully");

    LOG_INFO("Loading model weights file: yolov8n.bin");
    if (yolov8.load_model("yolov8n.bin") != 0)
    {
        LOG_ERROR("Failed to load yolov8n.bin - file not found or invalid");
        LOG_ERROR("Please ensure yolov8n.bin is in the current working directory");
        return -1;
    }

    auto model_load_end = std::chrono::high_resolution_clock::now();
    auto model_load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(model_load_end - model_load_start);
    LOG_INFO("Model loaded successfully in %lld milliseconds", model_load_duration.count());

    // 2. Open Stream
    LOG_INFO("Step 2: Connecting to RTSP Camera Stream");
    LOG_INFO("RTSP URL: %s", RTSP_URL.c_str());

    auto stream_start = std::chrono::high_resolution_clock::now();

    cv::VideoCapture cap(RTSP_URL, cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        LOG_ERROR("Failed to open RTSP stream");
        LOG_ERROR("Possible causes:");
        LOG_ERROR("  - Network connectivity issues");
        LOG_ERROR("  - Incorrect RTSP URL or credentials");
        LOG_ERROR("  - Camera is offline or unreachable");
        LOG_ERROR("  - Firewall blocking RTSP port");
        return -1;
    }

    auto stream_end = std::chrono::high_resolution_clock::now();
    auto stream_duration = std::chrono::duration_cast<std::chrono::milliseconds>(stream_end - stream_start);
    LOG_INFO("RTSP stream connected successfully in %lld milliseconds", stream_duration.count());

    // Get stream properties
    int stream_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int stream_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double stream_fps = cap.get(cv::CAP_PROP_FPS);
    LOG_INFO("Stream properties: %dx%d @ %.2f FPS", stream_width, stream_height, stream_fps);

    LOG_INFO("========================================");
    LOG_INFO("System is LIVE! Starting detection loop");
    LOG_INFO("Press 'q' to quit");
    LOG_INFO("========================================");

    cv::Mat frame;
    int frame_count = 0;
    int total_detections = 0;

    // Performance Timer
    cv::TickMeter tm;
    auto overall_start = std::chrono::high_resolution_clock::now();

    // Detection interval management
    auto last_detection_time = std::chrono::high_resolution_clock::now();
    const auto detection_interval = std::chrono::milliseconds(1000); // 1000ms = 1 second
    bool first_run = true;
    std::vector<Object> current_objects; // Store results to persist between scans

    // Create window explicitly
    cv::namedWindow("NCNN Human Detector", cv::WINDOW_AUTOSIZE);

    while (true)
    {
        auto frame_start = std::chrono::high_resolution_clock::now();

        LOG_DEBUG("--- Frame %d ---", frame_count + 1);
        LOG_DEBUG("Capturing frame from stream");

        cap >> frame;
        if (frame.empty())
        {
            LOG_WARN("Received empty frame - stream may have ended or connection lost");
            LOG_WARN("Attempting to reconnect...");

            // Try to reopen
            cap.release();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            cap.open(RTSP_URL, cv::CAP_FFMPEG);

            if (!cap.isOpened())
            {
                LOG_ERROR("Failed to reconnect to stream - exiting");
                break;
            }
            LOG_INFO("Successfully reconnected to stream");
            continue;
        }

        auto capture_end = std::chrono::high_resolution_clock::now();
        auto capture_duration = std::chrono::duration_cast<std::chrono::microseconds>(capture_end - frame_start);
        LOG_DEBUG("Frame captured: %dx%d in %lld microseconds", frame.cols, frame.rows, capture_duration.count());

        tm.start();

        // Check if it's time to run detection
        auto now = std::chrono::high_resolution_clock::now();
        auto time_since_last_detection = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_detection_time);

        if (first_run || time_since_last_detection >= detection_interval)
        {
            auto detection_start = std::chrono::high_resolution_clock::now();

            // >>> RUN DETECTION <<<
            LOG_DEBUG("Running person detection on frame (Interval triggered: %lld ms)", time_since_last_detection.count());
            current_objects = detect_people(frame, yolov8);

            auto detection_end = std::chrono::high_resolution_clock::now();
            auto detection_duration = std::chrono::duration_cast<std::chrono::microseconds>(detection_end - detection_start);
            LOG_INFO("Frame %d: Detected %d person(s) in %.2f ms",
                     frame_count + 1, (int)current_objects.size(), detection_duration.count() / 1000.0f);

            total_detections += (int)current_objects.size();
            last_detection_time = now;
            first_run = false;
        }
        else
        {
            LOG_DEBUG("Skipping detection (Last run %lld ms ago)", time_since_last_detection.count());
        }

        tm.stop();
        frame_count++;

        // >>> DRAW RESULTS (Using last known objects) <<<
        LOG_DEBUG("Drawing %d bounding boxes on frame", (int)current_objects.size());
        auto draw_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < current_objects.size(); i++)
        {
            const auto &obj = current_objects[i];

            // Draw Red Box
            cv::rectangle(frame, obj.rect, cv::Scalar(0, 0, 255), 2);

            // Draw Label
            std::string text = "Person " + std::to_string((int)(obj.prob * 100)) + "%";

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(frame, cv::Rect(cv::Point((int)obj.rect.x, (int)(obj.rect.y - label_size.height)), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(0, 0, 255), -1);

            cv::putText(frame, text, cv::Point((int)obj.rect.x, (int)obj.rect.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            LOG_DEBUG("  Box %zu: Person %.1f%% at [%.0f, %.0f, %.0fx%.0f]",
                      i + 1, obj.prob * 100.0f, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        }

        auto draw_end = std::chrono::high_resolution_clock::now();
        auto draw_duration = std::chrono::duration_cast<std::chrono::microseconds>(draw_end - draw_start);
        LOG_DEBUG("Drawing completed in %lld microseconds", draw_duration.count());

        // Draw FPS
        cv::putText(frame, "FPS: " + std::to_string((int)tm.getFPS()), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        // Draw detection count
        if (current_objects.empty())
        {
            cv::putText(frame, "Status: Scanning...", cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        }
        else
        {
            cv::putText(frame, "Status: PERSON DETECTED!", cv::Point(10, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        // Show scan timer
        long long ms_until_scan = std::max<long long>(0, 1000 - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - last_detection_time).count());
        cv::putText(frame, "Next scan: " + std::to_string(ms_until_scan) + "ms", cv::Point(10, 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
        cv::putText(frame, "Detections (Last Scan): " + std::to_string((int)current_objects.size()), cv::Point(10, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);

        auto display_start = std::chrono::high_resolution_clock::now();
        cv::imshow("NCNN Human Detector", frame);
        auto display_end = std::chrono::high_resolution_clock::now();
        auto display_duration = std::chrono::duration_cast<std::chrono::microseconds>(display_end - display_start);
        LOG_DEBUG("Display update completed in %lld microseconds", display_duration.count());

        tm.reset();

        if (cv::waitKey(1) == 'q')
        {
            LOG_INFO("Quit key ('q') pressed by user");
            break;
        }

        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
        LOG_DEBUG("Total frame processing time: %lld microseconds (%.2f ms)",
                  frame_duration.count(), frame_duration.count() / 1000.0f);
    }

    auto overall_end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::seconds>(overall_end - overall_start);

    LOG_INFO("========================================");
    LOG_INFO("Session Summary:");
    LOG_INFO("  Total frames processed: %d", frame_count);
    LOG_INFO("  Total detections: %d", total_detections);
    LOG_INFO("  Average detections per frame: %.2f", frame_count > 0 ? (float)total_detections / frame_count : 0.0f);
    LOG_INFO("  Session duration: %lld seconds", overall_duration.count());
    LOG_INFO("========================================");
    LOG_INFO("Shutting down gracefully");

    return 0;
}