#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp> // ADDED: Logging header
#include <net.h>                         // NCNN header
#include <stdio.h>

// --- CONFIGURATION ---
// This is the URL that worked for you earlier
std::string RTSP_URL = "rtsp://6868:6868@172.19.1.181:554/cam/realmonitor?channel=7&subtype=0";

int main()
{
    // ADDED: Turn on detailed logging
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);

    printf("--- CAMERA STREAM TEST ---\n");

    // 1. Verify NCNN (Keep this to ensure your files are still safe)
    ncnn::Net yolov8;
    yolov8.opt.use_vulkan_compute = false;
    int ret = yolov8.load_param("yolov8n.param");
    if (ret == 0)
    {
        printf("[OK] NCNN Model loaded successfully.\n");
    }
    else
    {
        printf("[WARNING] Could not find yolov8n.param. (Not critical for this test, but fix it later)\n");
    }

    // 2. Connect to Camera
    printf("[*] Connecting to camera... (This might take 5-10 seconds)\n");

    // MODIFIED: Added cv::CAP_FFMPEG to ensure we see the network logs
    cv::VideoCapture cap(RTSP_URL, cv::CAP_FFMPEG);

    // 3. Check Connection
    if (!cap.isOpened())
    {
        printf("[ERROR] Could not open the RTSP stream.\n");
        printf("Possible fixes:\n");
        printf("  - Check if the IP is still 172.19.1.181\n");
        printf("  - Check if VLC player can still open this link.\n");
        return -1;
    }
    printf("[SUCCESS] Connected! Starting stream...\n");

    // 4. Stream Loop
    cv::Mat frame;
    while (true)
    {
        // Read a new frame
        cap >> frame;

        // Safety check: Did we lose the signal?
        if (frame.empty())
        {
            printf("[ERROR] Blank frame received. Stream ended or network error.\n");
            break;
        }

        // Show the video
        cv::imshow("C++ Security Feed", frame);

        // Press 'q' to quit
        // waitKey(1) means "wait 1ms". This creates the delay needed for the window to refresh.
        char key = (char)cv::waitKey(1);
        if (key == 'q')
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}