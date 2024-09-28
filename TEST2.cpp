#if defined(_WIN32)
#define NOMINMAX
#endif

#include <map>
#include <set>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <facerec/libfacerec.h>
#include "ConsoleArgumentsParser.h"

int currentY = 0;
static const std::vector<std::pair<std::string, std::string>> bone_map = {
        {"right_ankle","right_knee"},
        {"right_knee","right_hip"},
        {"left_hip","right_hip"},
        {"left_shoulder","left_hip"},
        {"right_shoulder","right_hip"},
        {"left_shoulder","right_shoulder"},
        {"left_shoulder","left_elbow"},
        {"right_shoulder","right_elbow"},
        {"left_elbow","left_wrist"},
        {"right_elbow","right_wrist"},
        {"left_eye","right_eye"},
        {"nose","left_eye"},
        {"left_knee", "left_hip"},
        {"right_ear", "right_shoulder"},
        {"left_ear", "left_shoulder"},
        {"right_eye", "right_ear"},
        {"left_eye", "left_ear"},
        {"nose", "right_eye"},
        {"left_ankle", "left_knee"}
};

void putTextWithRightExpansion(cv::Mat& img, const std::string& text, cv::Point org, int fontFace, double fontScale, cv::Scalar color,
    int thickness = 1, int lineType = cv::LINE_8, bool bottomLeftOrigin = false, int borderType = cv::BORDER_CONSTANT)
{
    int* baseLine = nullptr;
    cv::Size text_sz = cv::getTextSize(text, fontFace, fontScale, thickness, baseLine);
    auto right_border = text_sz.width + org.x - img.cols;

    if (right_border > 0)
        cv::copyMakeBorder(img, img, 0, 0, 0, right_border + 2, borderType);

    cv::putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);
}

void drawObjects(const pbio::Context& data, cv::Mat& image, std::string class_filter = std::string())
{
    const auto width = image.size[1];
    const auto heigth = image.size[0];
    for (const auto& obj : data.at("objects"))
    {
        std::string class_name = obj.at("class").getString();
        if (!class_filter.empty() && class_name.compare(class_filter))
            continue;
        const auto& rectCtx = obj.at("bbox");
        cv::Rect rect(cv::Point{ std::max(static_cast<int>(rectCtx[0].getDouble() * width), 0),
                                 std::max(static_cast<int>(rectCtx[1].getDouble() * heigth), 0) },
            cv::Point{ std::min(static_cast<int>(rectCtx[2].getDouble() * width), width),
            std::min(static_cast<int>(rectCtx[3].getDouble() * heigth), heigth) });
        cv::rectangle(image, rect, { 0, 0, 255 }, 2);
        if (class_filter.empty())
        {
            putTextWithRightExpansion(image, class_name, cv::Point{ rect.x, rect.y + 20 + currentY }, cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 1, false);
            currentY += 30;
        }
        if (class_filter == "body" && obj.contains("keypoints"))
        {


            const auto& posesCtx = obj["keypoints"];

            for (auto ptr = posesCtx.begin(); ptr != posesCtx.end(); ++ptr)
            {
                std::string key_name = ptr.key();
                auto proj = (*ptr)["proj"];
                cv::circle(image, cv::Point{
                            static_cast<int>(proj[0].getDouble() * width),
                            static_cast<int>(proj[1].getDouble() * heigth) },
                            3, cv::Scalar(0, 255, 0), -1, false);
            }

            for (auto& bone : bone_map)
            {
                std::string key1 = bone.first;
                std::string key2 = bone.second;
                int x1 = posesCtx[key1]["proj"][0].getDouble() * width;
                int y1 = posesCtx[key1]["proj"][1].getDouble() * heigth;
                int x2 = posesCtx[key2]["proj"][0].getDouble() * width;
                int y2 = posesCtx[key2]["proj"][1].getDouble() * heigth;
                cv::line(image, cv::Point{ x1, y1 }, cv::Point{ x2,y2 }, cv::Scalar(0, 255, 0), 1,
                    cv::LINE_4);
            }
        }
    }
}

void drawFaceKeypoint(const pbio::Context& data, cv::Mat& image)
{
    const auto width = image.size[1];
    const auto heigth = image.size[0];

    for (const auto& obj : data.at("objects"))
    {
        for (const auto& point : obj.at("keypoints").at("points"))
        {
            cv::circle(image, cv::Point2f(point["proj"][0].getDouble() * image.size[1], point["proj"][1].getDouble() * image.size[0]), 2, { 0, 255, 0 }, 5);
        }
    }
}

void drawEmotions(const pbio::Context& data, cv::Mat& image)
{
    const auto width = image.size[1];
    const auto heigth = image.size[0];
    for (const auto& obj : data.at("objects"))
    {
        std::string class_name = obj.at("class").getString();
        if (class_name.compare("face"))
            continue;

        const pbio::Context& emotions = obj.at("emotions");
        cv::Point text_point{ std::min(static_cast<int>(obj.at("bbox")[2].getDouble() * width), width),
                             std::max(static_cast<int>(obj.at("bbox")[1].getDouble() * heigth), 0) + 15 + currentY };
        currentY += 90;

        std::set<std::pair<double, std::string>> results;
        for (size_t i = 0; i < emotions.size(); ++i)
            results.insert({ emotions[i]["confidence"].getDouble(), emotions[i]["emotion"].getString() });
        auto top_shift = results.rbegin()->first;
        double lenght_coef = 50;
        float font_height = 12;
        for (auto emotion_iter = results.rbegin(); emotion_iter != results.rend(); ++emotion_iter)
        {
            const std::string& emotion = emotion_iter->second;
            const double confidence = emotion_iter->first;
            const float text_line_height = font_height * std::max(0.5, confidence / top_shift);
            putTextWithRightExpansion(image, emotion, text_point + cv::Point(top_shift * lenght_coef, 0), cv::FONT_HERSHEY_DUPLEX, text_line_height / (font_height * 2), cv::Scalar(0, 255, 0), 1, false);
            cv::rectangle(
                image,
                cv::Rect(text_point.x, text_point.y - text_line_height / 2, confidence * lenght_coef, text_line_height),
                emotion == "NEUTRAL" ? cv::Scalar(255, 0, 0) :
                emotion == "HAPPY" ? cv::Scalar(0, 255, 0) :
                emotion == "ANGRY" ? cv::Scalar(0, 0, 255) :
                emotion == "SURPRISED" ? cv::Scalar(0, 255, 255) :
                emotion == "DISGUSTED" ? cv::Scalar(255, 255, 0) :
                emotion == "SAD" ? cv::Scalar(255, 0, 255) :
                emotion == "SCARED" ? cv::Scalar(127, 127, 127) :
                cv::Scalar(0, 0, 0),
                -1);

            text_point.y += text_line_height;
            text_point.y += text_line_height / 3;
        }
    }
}

void drawAgeGenderMaskQuality(const pbio::Context& data, cv::Mat& image, const std::string& className)
{
    const auto width = image.size[1];
    const auto heigth = image.size[0];
    cv::Mat image_copy = image.clone();

    int objects_counter = 1;
    for (const auto& obj : data.at("objects"))
    {
        if (obj.at("class").getString().compare("face"))
            continue;
        cv::Point text_point{ std::min(static_cast<int>(obj.at("bbox")[2].getDouble() * width), width),
                             std::max(static_cast<int>(obj.at("bbox")[1].getDouble() * heigth), 0) + 15 + currentY };
        currentY += 30;
        if (!className.compare("gender"))
            putTextWithRightExpansion(image, obj.at("gender").getString(), text_point, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1, false);
        else if (!className.compare("age"))
            putTextWithRightExpansion(image, "Age: " + std::to_string(obj.at(className).getLong()), text_point, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1, false);
        else if (!className.compare("mask"))
            putTextWithRightExpansion(image, "Mask: " + std::to_string(obj.at("has_medical_mask").at("value").getBool()), text_point, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1, false);
        else if (!className.compare("quality"))
        {
            if (objects_counter <= 6)
            {
                cv::Rect face_rect(cv::Point{ std::max(static_cast<int>(obj.at("bbox")[0].getDouble() * width), 0), std::max(static_cast<int>(obj.at("bbox")[1].getDouble() * heigth), 0) },
                    cv::Point{ std::min(static_cast<int>(obj.at("bbox")[2].getDouble() * width), width), std::min(static_cast<int>(obj.at("bbox")[3].getDouble() * heigth), heigth) });
                int height_for_text_and_crop = static_cast<int>(heigth * 0.15 * objects_counter - (heigth * 0.05));

                cv::Mat face = image_copy(face_rect);
                double scale_factor = heigth / (7.0 * (face.rows));
                cv::resize(face, face, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
                cv::Rect roi = cv::Rect(width + 5, height_for_text_and_crop, face.cols, face.rows);
                putTextWithRightExpansion(image, "Result of 6 first faces:", cv::Point{ width,20 }, cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 0), 1, false);
                putTextWithRightExpansion(image, "Quality :", cv::Point{ width + 5 + roi.width, height_for_text_and_crop + 30 }, cv::FONT_HERSHEY_DUPLEX,
                    0.5, cv::Scalar(0, 255, 0), 1, false);
                putTextWithRightExpansion(image, std::to_string((int)std::round(obj["quality"]["total_score"].getDouble() * 100)),
                    cv::Point{ width + 75 + roi.width, height_for_text_and_crop + 30 }, cv::FONT_HERSHEY_DUPLEX,
                    0.5, cv::Scalar(0, 255, 0), 1, false);
                cv::Mat draw_roi = image(roi);
                face.copyTo(draw_roi);
                objects_counter++;
            }
        }
    }
}

void drawLiveness(const pbio::Context& data, cv::Mat& image)
{
    const auto width = image.size[1];
    const auto heigth = image.size[0];

    for (const auto& obj : data.at("objects"))
    {
        if (obj.at("class").getString().compare("face"))
            continue;
        cv::Point text_point{ static_cast<int>(obj.at("bbox")[2].getDouble() * width),
                             static_cast<int>(obj.at("bbox")[1].getDouble() * heigth) + currentY };
        currentY += 30;
        if (obj.contains("liveness"))
        {
            const pbio::Context& liveness = obj.at("liveness");
            putTextWithRightExpansion(image, liveness.at("value").getString(), text_point, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1, false);
            text_point.y += 12;
            putTextWithRightExpansion(image, std::to_string(liveness.at("confidence").getDouble()), text_point, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1, false);
            if (liveness.contains("info"))
            {
                text_point.y += 12;
                putTextWithRightExpansion(image, "info: " + liveness.at("info").getString(), text_point, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255), 1, false);
            }
        }
    }
}

const std::map<std::string, std::string> unitTypes
{
    {"body", "HUMAN_BODY_DETECTOR"},
    {"face", "FACE_DETECTOR"},
    {"objects", "OBJECT_DETECTOR"},
    {"face_keypoint", "FACE_FITTER"},
    {"emotions", "EMOTION_ESTIMATOR"},
    {"age", "AGE_ESTIMATOR"},
    {"gender","GENDER_ESTIMATOR"},
    {"mask", "MASK_ESTIMATOR"},
    {"liveness", "LIVENESS_ESTIMATOR"},
    {"quality", "QUALITY_ASSESSMENT_ESTIMATOR"},
};



pbio::Context createFaceDetector(pbio::FacerecService& service);
pbio::Context createFaceFitter(pbio::FacerecService& service);



int main(int argc, char** argv)
{
    std::cout << "usage: " << argv[0] <<
        " [--unit_type body|face|face_keypoint|objects|emotions|age|gender|mask|liveness|quality]" << std::endl;

#if defined(_WIN32)
    const std::string default_dll_path = "/bin/facerec.dll";
#else
    const std::string default_dll_path = "/lib/libfacerec.so";
#endif

    ConsoleArgumentsParser parser(argc, argv);

    std::vector<std::string> unit_types = parser.get_all<std::string>("--unit_type");
    const std::string sdk_dir = BASE_DIR;
    const std::string modification = parser.get<std::string>("--modification", "");
    const std::string version = parser.get<std::string>("--version", "");

    const std::string dll_path{ sdk_dir + default_dll_path };
    const std::string config_dir{ sdk_dir + "/conf/facerec" };
    const std::string license_dir{ sdk_dir + "/license" };
#if defined(_WIN32)
    const std::string lib_dir{ sdk_dir + "/bin" };
#else
    const std::string lib_dir{ sdk_dir + "/lib" };
#endif

    try
    {
        const pbio::FacerecService::Ptr service =
            pbio::FacerecService::createService(
                dll_path,
                config_dir,
                license_dir);

        std::cout << "Library version: " << service->getVersion() << std::endl << std::endl;

        cv::VideoCapture cap(0);
        cv::Mat frame;

        if (!cap.isOpened())
        {
            throw std::runtime_error("Can't open camera");
        }

        while (true)
        {
            cap >> frame;
            currentY = 0;
            if (frame.empty())
            {
                std::cerr << "Empty frame captured!" << std::endl;
                break;
            }
            pbio::Context ioData = service->createContextFromFrame(frame.data, frame.cols, frame.rows, pbio::Context::Format::FORMAT_RGB, 0);

            for (const auto& unit_type : unit_types)
            {

                auto configCtx = service->createContext();

                if (unitTypes.find(unit_type) == unitTypes.end())
                    throw pbio::Error(0x917ca17f, "unit_type not found");

                configCtx["unit_type"] = unitTypes.at(unit_type);
                configCtx["ONNXRuntime"]["library_path"] = lib_dir;
                if (!modification.empty())
                    configCtx["modification"] = modification;
                if (!version.empty())
                    configCtx["version"] = std::stoll(version);

                if (unit_type == "quality")
                    configCtx["config_name"] = "quality_assessment.xml";

                if (unit_type == "liveness")
                {
                    configCtx["config_name"] = "liveness_2d_estimator_v3.xml";
                    configCtx["sdk_path"] = sdk_dir;
                    configCtx["facerec_conf_dir"] = sdk_dir + "/conf/facerec/";
                }

                pbio::ProcessingBlock processingBlock = service->createProcessingBlock(configCtx);

                if (unit_type == "quality" || unit_type == "liveness")
                {
                    pbio::ProcessingBlock faceDetector = service->createProcessingBlock(createFaceDetector(*service));
                    pbio::ProcessingBlock faceFitter = service->createProcessingBlock(createFaceFitter(*service));

                    faceDetector(ioData);
                    faceFitter(ioData);
                    processingBlock(ioData);
                }
                else if (!unit_type.compare("emotions") || !unit_type.compare("gender") ||
                    !unit_type.compare("age") || !unit_type.compare("mask"))
                {
                    pbio::ProcessingBlock faceDetector = service->createProcessingBlock(createFaceDetector(*service));

                    faceDetector(ioData);

                    if (unit_type.compare("face_keypoint"))
                    {
                        auto fitterCtx = service->createContext();
                        fitterCtx["unit_type"] = unitTypes.at("face_keypoint");
                        pbio::ProcessingBlock fitterBlock = service->createProcessingBlock(fitterCtx);
                        fitterBlock(ioData);
                    }

                    processingBlock(ioData);
                }
                else
                {
                    if (!unit_type.compare("pose"))
                    {
                        auto modelDetectorCtx = service->createContext();
                        modelDetectorCtx["unit_type"] = unitTypes.at("body");
                        modelDetectorCtx["ONNXRuntime"]["library_path"] = lib_dir;
                        modelDetectorCtx["confidence_threshold"] = 0.4;
                        modelDetectorCtx["iou_threshold"] = 0.45;
                        pbio::ProcessingBlock  bodyDetector = service->createProcessingBlock(modelDetectorCtx);
                        bodyDetector(ioData);
                    }

                    processingBlock(ioData);
                }

                if (ioData.isNone() || !ioData.contains("objects"))
                    std::cerr << "No objects have been detected!" << std::endl;
                else if (!unit_type.compare("body") || !unit_type.compare("pose"))
                    drawObjects(ioData, frame, "body");
                else if (!unit_type.compare("face"))
                    drawObjects(ioData, frame, "face");
                else if (!unit_type.compare("face_keypoint"))
                    drawFaceKeypoint(ioData, frame);
                else if (!unit_type.compare("objects"))
                    drawObjects(ioData, frame);
                else if (!unit_type.compare("emotions"))
                    drawEmotions(ioData, frame);
                else if (!unit_type.compare("age") || !unit_type.compare("gender") || !unit_type.compare("mask") || !unit_type.compare("quality"))
                    drawAgeGenderMaskQuality(ioData, frame, unit_type);
                else if (unit_type.find("liveness") != std::string::npos)
                    drawLiveness(ioData, frame);
            }

            cv::imshow("frame", frame);
            if (cv::waitKey(1) >= 0)
                break;
        }
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "Opencv exception caught: " << e.what() << std::endl;
    }
    catch (const pbio::Error& e)
    {
        std::cerr << "facerec exception caught: '" << e.what() << "' code: " << std::hex << e.code() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "exception caught: '" << e.what() << "'" << std::endl;
    }
    return 0;
}

pbio::Context createFaceDetector(pbio::FacerecService& service)
{
    pbio::Context config = service.createContext();

    config["unit_type"] = "FACE_DETECTOR";
    config["modification"] = "ssyv";
    config["version"] = static_cast<int64_t>(2);

    return config;
}

pbio::Context createFaceFitter(pbio::FacerecService& service)
{
    pbio::Context config = service.createContext();

    config["unit_type"] = "FACE_FITTER";
    config["modification"] = "fda";

    return config;
}
