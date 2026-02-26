/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <torch/torch.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

#include <rerun.hpp>
#include <sophus/se3.hpp>
#include <vector>
#include <string>
#include <chrono>

#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
std::string calib = "";
std::string dataAssociation = "";
std::string cfg_yaml = "";
std::string save_dir = "";
std::string which_dataset = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;
bool use_gaussian_viewer = false;

int mode=0;

bool firstRosSpin=false;

using namespace dso;


void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();
}



void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations=6;
		setting_minOptIterations=1;

		setting_logStuff = false;
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}






void parseArgument(char* arg)
{
	int option;
	float foption;
	char buf[1000];


    if(1==sscanf(arg,"use_gaussian_viewer=%d",&option))
    {
        if(option==1)
        {
            use_gaussian_viewer = true;
            printf("USING Gaussian VIEWER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

	if(1==sscanf(arg,"preset=%d",&option))
	{
		settingsDefault(option);
		return;
	}


	if(1==sscanf(arg,"rec=%d",&option))
	{
		if(option==0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
		return;
	}



	if(1==sscanf(arg,"noros=%d",&option))
	{
		if(option==1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}
	if(1==sscanf(arg,"reverse=%d",&option))
	{
		if(option==1)
		{
			reverse = true;
			printf("REVERSE!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"prefetch=%d",&option))
	{
		if(option==1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
		return;
	}
	if(1==sscanf(arg,"start=%d",&option))
	{
		start = option;
		printf("START AT %d!\n",start);
		return;
	}
	if(1==sscanf(arg,"end=%d",&option))
	{
		end = option;
		printf("END AT %d!\n",start);
		return;
	}

	if(1==sscanf(arg,"files=%s",buf))
	{
		source = buf;
		printf("loading data from %s!\n", source.c_str());
		return;
	}

	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}

	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignette = buf;
		printf("loading vignette from %s!\n", vignette.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaCalib = buf;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
		return;
	}

	if(1==sscanf(arg,"rescale=%f",&foption))
	{
		rescale = foption;
		printf("RESCALE %f!\n", rescale);
		return;
	}

	if(1==sscanf(arg,"speed=%f",&foption))
	{
		playbackSpeed = foption;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
		return;
	}

	if(1==sscanf(arg,"save=%d",&option))
	{
		if(option==1)
		{
			debugSaveImages = true;
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

	if(1==sscanf(arg,"mode=%d",&option))
	{

		mode = option;
		if(option==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(option==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		}
		if(option==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
		}
		return;
	}

	if (1 == sscanf(arg, "dataassociation=%s", buf)) 
	{
		dataAssociation = buf;
		printf("ASSOCIATION LOADED FROM %s\n", dataAssociation.c_str());
        return;
	}

	if (1 == sscanf(arg, "cfg_yaml=%s", buf)) 
	{
		cfg_yaml = buf;
		printf("Load parameters for 2DGS %s\n", cfg_yaml.c_str());
        return;
	}

	if (1 == sscanf(arg, "save_dir=%s", buf)) 
	{
		save_dir = buf;
		printf("SAVE DIRECTORY FOR KF %s\n", save_dir.c_str());
        return;
	}

	if (1 == sscanf(arg, "which_dataset=%s", buf)) 
	{
		which_dataset = buf;
		printf("Which dataset? %s\n", which_dataset.c_str());
        return;
	}

	printf("could not parse argument \"%s\"!!!!\n", arg);
}


std::vector<float> getPointBGR(MinimalImageB3* img, float u, float v) {
    if (u < 0 || u >= img->w || v < 0 || v >= img->h) {
        throw std::out_of_range("Coordinates are outside the image bounds!");
    }

    int x = static_cast<int>(u);
    int y = static_cast<int>(v);

    int index = y * img->w + x;

    Eigen::Matrix<unsigned char, 3, 1>& pixel = img->data[index];

	std::vector<float> out_colors = {(float)pixel(0)/255.0f, (float)pixel(1)/255.0f, (float)pixel(2)/255.0f};

	return out_colors;
}

void savePointCloud(const std::string& filename, FrameHessian* frameHessian, bool init) 
{
	if (!init)
	{
		std::filesystem::path filePath(filename);
		auto parentDir = filePath.parent_path();
		if (!parentDir.empty() && std::filesystem::exists(parentDir)) 
		{
			std::filesystem::remove_all(parentDir);
		}
		if (!std::filesystem::exists(parentDir)) 
		{
			std::filesystem::create_directories(parentDir);
		}
	}

	std::ofstream file(filename);
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << frameHessian->pointHessians.size() << "\n";
    file << "property float idepth_scaled\n";
	file << "property float width\n";
	file << "property float height\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";

	MinimalImageB3* img = frameHessian->kf_img;

	for (PointHessian* ph : frameHessian->pointHessians) 
	{
		if (ph->idepth > 0) {
			float depth = 1.0f / ph->idepth;

			std::vector<float> bgr = getPointBGR(img, ph->u, ph->v);
			
			// float r = static_cast<float>(ph->kf_color[2]);
			// float g = static_cast<float>(ph->kf_color[1]);
			// float b = static_cast<float>(ph->kf_color[0]);

			float r = bgr[2];
			float g = bgr[1];
			float b = bgr[0];

			file << ph->idepth << " " << ph->u << " " << ph->v << " "
					<< r << " " << g << " " << b << "\n";
		}
    }
    file.close();
}


void saveAllFramesTrajectoryTUM(std::vector<FrameShell*> allFrameHistory, const std::string& filename, ImageFolderReader* reader)
{
	std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

	for(size_t i=1; i<allFrameHistory.size(); i++)
    {
        FrameShell* frame = allFrameHistory[i];

		if(frame->poseValid == false)
			continue;

        Sophus::SE3f c2w = frame->camToWorld.cast<float>();
        Eigen::Quaternionf q = c2w.unit_quaternion();
        Eigen::Vector3f t = c2w.translation();

		double timestamp;
		if (which_dataset == "tum_rgbd") {
			timestamp = reader->getTimeForEval(i);
		} else if (which_dataset == "replica") {
			timestamp = (double)frame->incoming_id;
		}

        f << std::fixed << std::setprecision(6) << timestamp << std::setprecision(9) << " " << t(0) << " " << t(1) << " " << t(2)
          << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

    f.close();
}

void saveImage(const std::string& filename, const MinimalImageB3* kf_img, bool init) 
{
	if (!init)
	{
		std::filesystem::path filePath(filename);
		auto parentDir = filePath.parent_path();
		if (!parentDir.empty() && std::filesystem::exists(parentDir)) 
		{
			std::filesystem::remove_all(parentDir);
		}
		if (!std::filesystem::exists(parentDir)) 
		{
			std::filesystem::create_directories(parentDir);
		}
	}

    cv::Mat image(kf_img->h, kf_img->w, CV_8UC3, kf_img->data);
    cv::imwrite(filename, image);
}


void saveCameraPose(const std::string& filename, const SE3& camToWorld, bool init) 
{
	if (!init)
	{
		std::filesystem::path filePath(filename);
		auto parentDir = filePath.parent_path();
		if (!parentDir.empty() && std::filesystem::exists(parentDir)) 
		{
			std::filesystem::remove_all(parentDir);
		}
		if (!std::filesystem::exists(parentDir)) 
		{
			std::filesystem::create_directories(parentDir);
		}
	}

    std::ofstream file(filename);

    Eigen::Matrix4d mat = camToWorld.matrix();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            file << mat(i, j) << " ";
        }
        file << "\n";
    }
    file.close();
}


void saveAffLight(const std::string& filename, AffLight aff_g2l, bool init)
{
	if (!init)
	{
		std::filesystem::path filePath(filename);
		auto parentDir = filePath.parent_path();
		if (!parentDir.empty() && std::filesystem::exists(parentDir)) 
		{
			std::filesystem::remove_all(parentDir);
		}
		if (!std::filesystem::exists(parentDir)) 
		{
			std::filesystem::create_directories(parentDir);
		}
	}

    std::ofstream outFile(filename);

    double a = aff_g2l.a;
    double b = aff_g2l.b;

    outFile << std::fixed << std::setprecision(6);
    outFile << "aff_g2l.a: " << a << "\n";
    outFile << "aff_g2l.b: " << b << "\n";

    outFile.close();
}



void rerunVis(const rerun::RecordingStream& rec, const std::vector<FrameHessian*>& keyframeHessians, CalibHessian& HCalib)
{
    int i = 0;
	float fx = HCalib.fxl();
	float fy = HCalib.fyl();
	float cx = HCalib.cxl();
	float cy = HCalib.cyl();

    for (FrameHessian* fh : keyframeHessians) {
		Sophus::SE3d camToWorld = fh->shell->camToWorld;
		Eigen::Vector3d translation = camToWorld.translation();
		Eigen::Quaterniond rotation = camToWorld.unit_quaternion();

		MinimalImageB3* img = fh->kf_img;
		cv::Mat imgMat(img->h, img->w, CV_8UC3, img->data);

		std::string stimestamp = "timestamp";
		rec.set_time_seconds(stimestamp, i);

		std::string entity_path = "keyframe/transform_" + std::to_string(fh->shell->incoming_id);

		rec.log(
			entity_path,
			rerun::Transform3D(
				rerun::components::Translation3D(
					translation.x(),
					translation.y(),
					translation.z()
				),
				rerun::Quaternion::from_wxyz(
					rotation.w(),
					rotation.x(),
					rotation.y(),
					rotation.z()
				)
			)
		);

		const std::array<float, 9> cam_intrinsic = {
			fx, 0.0f, cx,   
			0.0f, fy, cy,   
			0.0f, 0.0f, 1.0f
		};

		rec.log(
			entity_path,
			rerun::Pinhole(cam_intrinsic)
				.with_resolution(img->w, img->h)
		);

		int width = img->w;
		int height = img->h;

		rerun::WidthHeight width_height(width, height);
		std::vector<uint8_t> image_vector(imgMat.data, imgMat.data + (width * height * 3));

		rec.log(
			entity_path,
			rerun::Image::from_rgb24(
				rerun::Collection<uint8_t>(image_vector),
				width_height
			)
		);

		// int j = 0;
		// for (PointHessian* ph : fh->pointHessians) {
        //     if (ph->idepth > 0) {
        //         float depth = 1.0f / ph->idepth;
        //         float x = ((ph->u - cx) / fx) * depth;
        //         float y = ((ph->v - cy) / fy) * depth;
        //         float z = depth;

        //         Eigen::Vector3d point_in_camera(x, y, z);
        //         Eigen::Vector3d point_in_world = camToWorld * point_in_camera;

        //         std::string point_entity_path = "mappoints/" + std::to_string(fh->shell->incoming_id) + "/points/" + std::to_string(j);

		// 		rec.log(
		// 			point_entity_path,
		// 			rerun::Points3D({
		// 				{
		// 					static_cast<float>(point_in_world.x()), 
		// 					static_cast<float>(point_in_world.y()), 
		// 					static_cast<float>(point_in_world.z())
		// 				}
		// 			})
        //         );
        //     }
		// 	++j;
        // }
	++i;
    }
}

int main( int argc, char** argv )
{
	//setlocale(LC_ALL, "");
	for(int i=1; i<argc;i++)
		parseArgument(argv[i]);

	// hook crtl+C.
	boost::thread exThread = boost::thread(exitThread);

	ImageFolderReader* reader = new ImageFolderReader(source, calib, gammaCalib, vignette, "", mode, cfg_yaml, which_dataset);

	// TODO
	// if (mode == 1) // tum_rgbd
	if(which_dataset == "tum_rgbd")
	{
        delete reader;

		reader = new ImageFolderReader(source, calib, gammaCalib, vignette, dataAssociation, mode, cfg_yaml, which_dataset);
	}
	// if (mode == 2) // replica, scannet,
	if(which_dataset == "replica" || which_dataset == "scannet")
	{
        delete reader;

		reader = new ImageFolderReader(source, calib, gammaCalib, vignette, "", mode, cfg_yaml, which_dataset);
	}

	reader->setGlobalCalibration();

	// Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

	// Load cfg (currently, only 2DGS parameters) (TODO: whole parameters)
	std::filesystem::path cfg_path(cfg_yaml);

	if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
	{
		printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}

	int lstart=start;
	int lend = end;
	int linc = 1;
	if(reverse)
	{
		printf("REVERSE!!!!");
		lstart=end-1;
		if(lstart >= reader->getNumImages())
			lstart = reader->getNumImages()-1;
		lend = start;
		linc = -1;
	}


	// Create SLAM
	// FullSystem* fullSystem = new FullSystem();
	std::shared_ptr<FullSystem> fullSystem = std::make_shared<FullSystem>();
	fullSystem->setGammaFunction(reader->getPhotometricGamma());
	fullSystem->linearizeOperation = (playbackSpeed==0);


	// Create GaussianMapper
	std::thread training_thd;
	std::filesystem::path output_dir(save_dir);
	std::shared_ptr<GaussianMapper> pGausMapper;
	// if (mode == 1 || mode == 2) {
	pGausMapper = std::make_shared<GaussianMapper>(
		fullSystem, cfg_path, output_dir, 0, device_type);
	training_thd = std::thread(&GaussianMapper::run, pGausMapper.get());
	// }

    IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }

	std::thread viewer_thd;
	// if ((mode == 1 || mode == 2) && use_gaussian_viewer) {
	if (use_gaussian_viewer) {
		std::shared_ptr<ImGuiViewer> pViewer;
		pViewer = std::make_shared<ImGuiViewer>(fullSystem, pGausMapper);
		viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
	}

    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());


    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        for(int i=lstart;i>= 0 && i< reader->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }


        std::vector<ImageAndExposure*> preloadedImages;
        if(preload)
        {
            printf("LOADING ALL IMAGES!\n");
            for(int ii=0;ii<(int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0.0;
		int prev_gt_saved = -1;

		fullSystem->isLastimg = false;
		fullSystem->isSave = !save_dir.empty();
		bool init_nonkf = false;

        for(int ii=0;ii<(int)idsToPlay.size(); ii++)
        {
			// for demo
			// if (ii > 230) {
			// 	break;
			// }

            if(!fullSystem->initialized)	// if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];


            ImageAndExposure* img;
			MinimalImageB3* gt_img;
			MinimalImage<unsigned short>* gt_depth;
            if(preload)
                img = preloadedImages[ii];
            else
                img = reader->getImage(i);

			// for demo video
			gt_img = reader->getImageRawRGB(i);

            bool skipFrame=false;
            if(playbackSpeed!=0)
            {
                struct timeval tv_now; gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec-tv_start.tv_sec) + (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));

                if(sSinceStart < timesToPlayAt[ii])
                    usleep((int)((timesToPlayAt[ii]-sSinceStart)*1000*1000));
                else if(sSinceStart > timesToPlayAt[ii]+0.5+0.1*(ii%2))
                {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame=true;
                }
            }


			// if (ii == (int)idsToPlay.size() - 1) {
			// 	fullSystem->isLastimg = true;
			// 	std::vector<FrameHessian*> wframeHessians = fullSystem->getFrameHessians();
				// for(unsigned int i=0;i<wframeHessians.size();i++){
				// 	std::cout << wframeHessians[i]->pointHessians.size() << " " <<
				// 	wframeHessians[i]->pointHessiansMarginalized.size() << " " <<
				// 	wframeHessians[i]->pointHessiansOut.size() << " " <<
				// 	wframeHessians[i]->immaturePoints.size() << " " <<
				// 	wframeHessians[i]->shell->incoming_id << std::endl;
				// }
			// }


            if(!skipFrame) fullSystem->addActiveFrame(img, gt_img, i, mode);

			// // Only use to check convergence of tracked poses 
			// if (!save_dir.empty() && fullSystem->current_fh != nullptr){
			// 	std::string imgfile = save_dir+"/nonkf_img/"+std::to_string(i)+".png";
			// 	std::string posefile = save_dir+"/nonkf_pose/"+std::to_string(i)+".txt";
			// 	std::string afffile = save_dir+"/nonkf_aff/"+std::to_string(i)+".txt";
			// 	auto nonkf_img = reader->getImageRawRGB(i);
			// 	saveImage(imgfile, nonkf_img, init_nonkf);
			// 	saveCameraPose(posefile, fullSystem->current_fh->shell->camToWorld, init_nonkf);
			// 	saveAffLight(afffile, fullSystem->current_fh->shell->aff_g2l, init_nonkf);
			// 	init_nonkf = true;
			// }


			{	// Locked block
				std::unique_lock<std::mutex> lock(fullSystem->allKeyframeMutex);
				if (!fullSystem->allKeyframeHessians.empty())
				{
					auto lastKeyframe = fullSystem->allKeyframeHessians.back();
					if (lastKeyframe->shell->incoming_id != prev_gt_saved) {
						// std::cout << lastKeyframe->shell->incoming_id << std::endl;
						gt_img = reader->getImageRawRGB(lastKeyframe->shell->incoming_id);

						// if (mode!=0 && reader->RGBorRGBD == "RGB-D") {
						// 	gt_depth = reader->getImageRawDepth(lastKeyframe->shell->incoming_id);
						// }
						
						lastKeyframe->kf_img = gt_img;

						// lastKeyframe->kf_sparse_depth = cv::Mat::zeros(gt_img->h, gt_img->w, CV_32FC1);

						// if (!lastKeyframe->pointHessians.empty()) {
						// 	for (PointHessian* ph : lastKeyframe->pointHessians) {
						// 		ph->kf_color = getPointBGR(gt_img, ph->u, ph->v);
						// 		// sparse_depth
						// 		lastKeyframe->kf_sparse_depth.at<float>(ph->v, ph->u) = 1. / ph->idepth_scaled;
						// 	}
						// }
						// if (!lastKeyframe->pointHessiansMarginalized.empty()) {
						// 	for (PointHessian* ph : lastKeyframe->pointHessiansMarginalized) {
						// 		ph->kf_color = getPointBGR(gt_img, ph->u, ph->v);
						// 		// sparse_depth
						// 		lastKeyframe->kf_sparse_depth.at<float>(ph->v, ph->u) = 1. / ph->idepth_scaled;
						// 	}
						// }

						// lastKeyframe->exist_gt_img = true;

						fullSystem->keepKFHessians.push_back(lastKeyframe);
						prev_gt_saved = lastKeyframe->shell->incoming_id;
					}
				}
			}



            delete img;

            if(fullSystem->isLost)
            {
                    printf("LOST!!\n");
                    break;
            }

			// loop
			{
				struct timeval tv_now;
				gettimeofday(&tv_now, NULL);

				double frameDuration = 1.0/30.0; // 30 fps
				double idealTime = (ii + 1) * frameDuration;

				double timeSinceStart =
					(tv_now.tv_sec - tv_start.tv_sec) + 
					(tv_now.tv_usec - tv_start.tv_usec) / 1e6;

				double remaining = idealTime - timeSinceStart;
				if(remaining > 0)
				{
					usleep((useconds_t)(remaining * 1e6));
				}
			}
        }
        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);


        fullSystem->printResult("result.txt");


        int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        // double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
		double MilliSecondsTakenMT = ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader->getNumImages()) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)reader->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }

    });


    // if(viewer != 0)
    //     viewer->run();

	// if (mode == 1 || mode == 2){
	
	if (use_gaussian_viewer)
		viewer_thd.join();
	// }

	training_thd.join();
	runthread.join();

	if (!fullSystem->isRunMapping()) {
	// if (false) {
		std::string test = "test_viewer";
		// const auto rec = rerun::RecordingStream(test);
		// rec.spawn().exit_on_failure();

		// std::vector<FrameHessian*> keyframeHessians = fullSystem->allKeyframeHessians;

		CalibHessian HCalib;
		fullSystem->setHcalib(HCalib);

		// Save results
		if (!save_dir.empty()) {
			bool init = false;
			std::vector<FrameHessian*> saveKeyframeHessians = fullSystem->keepKFHessians;
			for (size_t i = 0; i < saveKeyframeHessians.size(); ++i) {
				FrameHessian* fh = saveKeyframeHessians[i];
				std::string pointfile = save_dir+"/map/"+std::to_string(fh->shell->incoming_id)+".ply";
				std::string imgfile = save_dir+"/img/"+std::to_string(fh->shell->incoming_id)+".png";
				std::string posefile = save_dir+"/pose/"+std::to_string(fh->shell->incoming_id)+".txt";
				std::string afffile = save_dir+"/aff/"+std::to_string(fh->shell->incoming_id)+".txt";
				savePointCloud(pointfile, fh, init);
				saveImage(imgfile, fh->kf_img, init);
				saveCameraPose(posefile, fh->shell->camToWorld, init);
				saveAffLight(afffile, fh->shell->aff_g2l, init);
				init = true;
			}
			std::cout << "DSO Keyframe Results Saved" << std::endl;
		}

		// Save All Tracked Poses
		std::string allposefile = save_dir+"/AllFramePose.txt";
		saveAllFramesTrajectoryTUM(fullSystem->allFrameHistory, allposefile, reader);
	};


	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}


	// printf("DELETE FULLSYSTEM!\n");
	// delete &fullSystem;

	printf("DELETE READER!\n");
	delete reader;

	printf("EXIT NOW!\n");
	return 0;
}
