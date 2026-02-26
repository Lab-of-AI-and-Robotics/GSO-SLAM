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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;



FullSystem::FullSystem()
{

	int retstat =0;
	if(setting_logStuff)
	{

		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);



	selectionMap = new float[wG[0]*hG[0]];

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	delete[] selectionMap;

	for(FrameShell* s : allFrameHistory)
		delete s;
	for(FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	for(FrameShell* s : allFrameHistory)
	{
		if(!s->poseValid) continue;

		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

		myfile << s->timestamp <<
			" " << s->camToWorld.translation().transpose()<<
			" " << s->camToWorld.so3().unit_quaternion().x()<<
			" " << s->camToWorld.so3().unit_quaternion().y()<<
			" " << s->camToWorld.so3().unit_quaternion().z()<<
			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
	}
	myfile.close();
}


Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);



	FrameHessian* lastF = coarseTracker->lastRef;

	AffLight aff_last_2_l = AffLight(0,0);

	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	if(allFrameHistory.size() == 2)
		for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) lastF_2_fh_tries.push_back(SE3());
	else
	{
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
			lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
			aff_last_2_l = slast->aff_g2l;
		}
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.


		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.


		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}

		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}


	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	// I'll keep track of the so-far best achieved residual for each level in achievedRes.
	// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		bool trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, aff_g2l_this,
				pyrLevelsUsed-1,
				achievedRes);	// in each level has to be at least as good as the last try.
		tryIterations++;

		if(i != 0)
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					i,
					i, pyrLevelsUsed-1,
					aff_g2l_this.a,aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					coarseTracker->lastResiduals[0],
					coarseTracker->lastResiduals[1],
					coarseTracker->lastResiduals[2],
					coarseTracker->lastResiduals[3],
					coarseTracker->lastResiduals[4]);
		}


		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
					achievedRes[i] = coarseTracker->lastResiduals[i];
			}
		}


        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

	}

	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;


	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << " "
						<< fh->shell->camToWorld.log().transpose() << " "
						<< aff_g2l.a << " "
						<< aff_g2l.b << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}


	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}


void FullSystem::optimizeRefPose(FrameHessian* fh, const std::shared_ptr<pcl::PointCloud<PointXYZIndexBW>>& ref_cloud)
{
	FrameHessian* lastF = coarseTracker->lastRef;

	AffLight aff_last_2_l = AffLight(0,0);
	aff_last_2_l = lastF->shell->aff_g2l;

	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM reference KF.

	////// Constant Motion Trial //////
	// if(allFrameHistory.size() == 1)
	// {
	// 	lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM reference KF.
	// 	std::cout << allFrameHistory[allFrameHistory.size()-1]->incoming_id << allFrameHistory.size() << std::endl;
	// }
	// else
	// {
	// 	FrameShell* last_shell = allFrameHistory[allFrameHistory.size()-1];
	// 	FrameShell* prelast_shell = allFrameHistory[allFrameHistory.size()-2];
	// 	SE3 sprelast_2_slast;
		
	// 	{	// lock on global pose consistency!
	// 		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
	// 		sprelast_2_slast = last_shell->camToWorld.inverse() * prelast_shell->camToWorld;
	// 	}
	// 	SE3 slast_2_fh = sprelast_2_slast; // assume t-2 to t-1 same with t-1 to t
		
	// 	lastF_2_fh_tries.push_back(slast_2_fh); // assume constant motion.
	// }

	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		coarseTracker->optimizeRelative(fh, lastF_2_fh_this, aff_g2l_this, pyrLevelsUsed-1, ref_cloud);

		aff_g2l = aff_g2l_this;
		lastF_2_fh = lastF_2_fh_this;
	}


	// no lock required, as fh is not used anywhere yet.
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	// std::cout << "relative matrix: " << lastF_2_fh.matrix() << std::endl;
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
}


void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	K(0,0) = Hcalib.fxl();
	K(1,1) = Hcalib.fyl();
	K(0,2) = Hcalib.cxl();
	K(1,2) = Hcalib.cyl();

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{

		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for(ImmaturePoint* ph : host->immaturePoints)
		{
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}




void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}



void FullSystem::activatePointsMT()
{

	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



	FrameHessian* newestHs = frameHessians.back();

	// make dist map.
	coarseDistanceMap->makeK(&Hcalib);
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	//coarseTracker->debugPlotDistMap("distMap");

	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);


	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		if(host == newestHs) continue;

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


		for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
		{
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
//				immature_invalid_deleted++;
				// remove point.
				delete ph;
				host->immaturePoints[i]=0;
				continue;
			}

			// can activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8
							&& ph->quality > setting_minTraceQuality
							&& (ph->idepth_max+ph->idepth_min) > 0;


			// if I cannot activate the point, skip it. Maybe also delete it.
			if(!canActivate)
			{
				// if point will be out afterwards, delete it instead.
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
//					immature_notReady_deleted++;
					delete ph;
					host->immaturePoints[i]=0;
				}
//				immature_notReady_skipped++;
				continue;
			}


			// see if we need to activate point due to distance map.
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
			{

				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				if(dist>=currentMinActDist* ph->my_type)
				{
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
			}
			else
			{
				delete ph;
				host->immaturePoints[i]=0;
			}
		}
	}


//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			ef->insertPoint(newpoint);
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		}
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			delete ph;
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}


	for(FrameHessian* host : frameHessians)
	{
		for(int i=0;i<(int)host->immaturePoints.size();i++)
		{
			if(host->immaturePoints[i]==0)
			{
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}


}






void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	//if(setting_margPointVisWindow>0)
	{
		for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
			if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

		for(int i=0; i< (int)frameHessians.size();i++)
			if(frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
	}



	//ef->setAdjointsF();
	//ef->setDeltaF(&Hcalib);
	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		for(unsigned int i=0;i<host->pointHessians.size();i++)
		{
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) continue;

			if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
			{
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				host->pointHessians[i]=0;
				flag_nores++;
			}
			else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;
				if(ph->isInlierNew())
				{
					flag_in++;
					int ngoodRes=0;
					for(PointFrameResidual* r : ph->residuals)
					{
						r->resetOOB();
						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
						r->applyRes(true);
						if(r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
                    if(ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}


				}
				else
				{
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}

				host->pointHessians[i]=0;
			}
		}


		for(int i=0;i<(int)host->pointHessians.size();i++)
		{
			if(host->pointHessians[i]==0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}

}


void FullSystem::addActiveFrame( ImageAndExposure* image, MinimalImageB3* gt_img, int id, int mode)
{

    if(isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);


	// =========================== add into allFrameHistory =========================
	FrameHessian* fh = new FrameHessian();
	FrameShell* shell = new FrameShell();
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
	fh->shell = shell;
	allFrameHistory.push_back(shell);
	// gt img, for 2DGS
	fh->kf_img = gt_img;
	fh->exist_gt_img = true;

	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib);


	if(!initialized)
	{
		// use initializer!
		if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer.
		{

			coarseInitializer->setFirst(&Hcalib, fh);
		}
		else if(coarseInitializer->trackFrame(fh, outputWrapper))	// if SNAPPED
		{

			initializeFromInitializer(fh);
			lock.unlock();
			deliverTrackedFrame(fh, true);
		}
		else
		{
			// if still initializing
			fh->shell->poseValid = false;
			delete fh;
			fh = nullptr;
		}
		if (isSave)
		{
			if (fh != nullptr){current_fh = new FrameHessian(*fh);}
			else{current_fh = nullptr;}
		}
		return;
	}
	else	// do front-end operation.
	{
		// =========================== SWAP tracking reference?. =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker; coarseTracker=coarseTracker_forNewKF; coarseTracker_forNewKF=tmp;
		}


		Vec4 tres = trackNewCoarse(fh);
		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
			isLost=true;
            return;
        }

		bool needToMakeKF = false;
		if(setting_keyframesPerSecond > 0)
		{
			needToMakeKF = allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{
			Vec2 refToFh=AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// BRIGHTNESS CHECK
			if(mode==1)
			{
				needToMakeKF = allFrameHistory.size()== 1 ||
						setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
						setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
						setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
						0.0 * setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1.4 ||
						2.5*coarseTracker->firstCoarseRMSE < tres[0];
			}
			if(mode==2)
			{
				needToMakeKF = allFrameHistory.size()== 1 ||
						setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
						setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
						setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
						1.0 * setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1.0 ||
						2.0*coarseTracker->firstCoarseRMSE < tres[0];
			}
			// Replica: 1.0, 2
			// TUM-RGBD: 0, 1.4, 2.5
		}




        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);




		lock.unlock();
		if(isSave) {current_fh = new FrameHessian(*fh);}
		deliverTrackedFrame(fh, needToMakeKF);
		return;
	}
}


SE3 FullSystem::localizeFrame( ImageAndExposure* image, FrameHessian* refFrame, const std::shared_ptr<pcl::PointCloud<PointXYZIndexBW>>& ref_cloud )
{
	// =========================== add into allFrameHistory =========================
	std::unique_ptr<FrameHessian> fh = std::make_unique<FrameHessian>();
	std::unique_ptr<FrameShell> shell = std::make_unique<FrameShell>();
	shell->camToWorld = refFrame->shell->camToWorld; 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0,0);
    shell->timestamp = image->timestamp;
    shell->incoming_id = refFrame->shell->incoming_id + 1;
	fh->shell = shell.get();
	allFrameHistory.push_back(shell.get());


	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image->exposure_time;
	refFrame->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib);

	coarseTracker->makeK(&Hcalib);
	coarseTracker->setTrackingRef(refFrame);

	optimizeRefPose(fh.get(), ref_cloud);

	// std::cout << "Initial camToWorld: " << refFrame->shell->camToWorld.matrix() << std::endl;
	// std::cout << "Updated camToWorld: " << fh->shell->camToWorld.matrix() << std::endl;

	return fh->shell->camToWorld;
}


void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{


	if(linearizeOperation)
	{
		if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );



		if(needKF) {
			makeKeyFrame(fh);
			lastKFIncomingId = fh->shell->incoming_id;
		}
		else makeNonKeyFrame(fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
}

void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();


		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeNonKeyFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
				}
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				lock.unlock();
				makeKeyFrame(fh);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);
	delete fh;
}

void FullSystem::makeKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);

	// call Gaussian Mapper
	callKFUpdateFromGS = true;
	isDoneKFUpdateFromGS = false;
	// std::cout << "Call GS Mapper" << std::endl;
	while (!isDoneKFUpdateFromGS) {
		// std::cout << "Waiting Gaussian Mapper to update DSO KFs..." << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	callKFUpdateFromGS = false;

	boost::unique_lock<boost::mutex> lock(mapMutex);

	// =========================== Flag Frames to be Marginalized. =========================
	flagFramesForMarginalization(fh);


	// =========================== add New Frame to Hessian Struct. =========================
	fh->idx = frameHessians.size();
	frameHessians.push_back(fh);
	fh->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);

	setPrecalcValues();

	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians)
		{
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
			ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0];
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
			numFwdResAdde+=1;
		}
	}




	// =========================== Activate Points (& flag for marginalization). =========================
	activatePointsMT();
	ef->makeIDX();




	// =========================== OPTIMIZE ALL =========================

	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
	float rmse = optimize(setting_maxOptIterations);





	// =========================== Figure Out if INITIALIZATION FAILED =========================
	if(allKeyFramesHistory.size() <= 4)
	{
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}



    if(isLost) return;




	// =========================== REMOVE OUTLIER =========================
	removeOutliers();




	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		coarseTracker_forNewKF->makeK(&Hcalib);
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);



        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}


	debugPlot("post Optimize");






	// =========================== (Activate-)Marginalize Points =========================
	flagPointsForRemoval();
	ef->dropPointsF();
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
	ef->marginalizePointsF();



	// =========================== add new Immature points & new residuals =========================
	makeNewTraces(fh, 0);





    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }

	
	// for(unsigned int i=0;i<frameHessians.size();i++){
	// 	std::cout << frameHessians[i]->pointHessians.size() << " " <<
	// 	frameHessians[i]->pointHessiansMarginalized.size() << " " <<
	// 	frameHessians[i]->pointHessiansOut.size() << " " <<
	// 	frameHessians[i]->immaturePoints.size() << std::endl;
	// }
	// std::cout << std::endl;
	

	// add new frozen fh, for 2DGS
	boost::unique_lock<boost::mutex> frozenKFLock(frozenMapMutex);
	FrozenFrameHessian* frozenFh = new FrozenFrameHessian();
	frozenFh->camToWorld = fh->shell->camToWorld;
	frozenFh->kfImg = fh->kf_img;
	frozenFh->incomingID = fh->shell->incoming_id;
	frameHessiansFrozen.push_back(frozenFh);

	// =========================== Marginalize Frames =========================
	ImmaturePoint* pt_dummy = new ImmaturePoint(10,10,fh,1.0f, &Hcalib);
	pt_dummy->idepth_max = 100.0f;

	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization){
			// for GS
			FrameHessian* fh_tosave = new FrameHessian();
			FrameShell* shell_tosave = new FrameShell();

			shell_tosave->camToWorld = frameHessians[i]->shell->camToWorld;
			fh_tosave->shell = shell_tosave;
			fh_tosave->shell->incoming_id = frameHessians[i]->shell->incoming_id;
			fh_tosave->shell->aff_g2l = frameHessians[i]->shell->aff_g2l;

			for (size_t j = 0; j < frameHessians[i]->pointHessians.size(); ++j) {
				PointHessian* ph = frameHessians[i]->pointHessians[j];

				PointHessian* ph_tosave = new PointHessian(pt_dummy, &Hcalib);
				ph_tosave->idepth = ph->idepth_scaled;
				ph_tosave->u = ph->u;
				ph_tosave->v = ph->v;

				fh_tosave->pointHessians.push_back(ph_tosave);
			}
			for(PointHessian* ph : frameHessians[i]->pointHessiansMarginalized)
			{
				PointHessian* ph_tosave = new PointHessian(pt_dummy, &Hcalib);
				ph_tosave->idepth = ph->idepth_scaled;
				ph_tosave->u = ph->u;
				ph_tosave->v = ph->v;

				fh_tosave->pointHessians.push_back(ph_tosave);
			}

			//// Ground Truth 이미지 input 여부
			fh_tosave->exist_gt_img = false;

			{
				std::unique_lock<std::mutex> lock(allKeyframeMutex);
				allKeyframeHessians.push_back(fh_tosave);
			}
			marginalizeFrame(frameHessians[i]);

			// remove frozenfh
			deleteOutOrder<FrozenFrameHessian>(frameHessiansFrozen, frameHessiansFrozen[i]);
			i=0;
		}

	// std::cout << "frozen/active: " << std::endl;
	// for(unsigned int i=0; i<frameHessians.size()-1; i++) {
	// 	std::cout << frameHessiansFrozen[i]->incomingID << "/" << frameHessians[i]->shell->incoming_id << std::endl;
	// }
	// std::cout << "" << std::endl;

	//// Covariance Insertion ////
	if (frameHessians.size() >= 3)
	{
		Mat33f K = Mat33f::Identity();
		K(0, 0) = Hcalib.fxl();
		K(1, 1) = Hcalib.fyl();
		K(0, 2) = Hcalib.cxl();
		K(1, 2) = Hcalib.cyl();

		int width = coarseTracker_forNewKF->w[0];
		int height = coarseTracker_forNewKF->h[0];

		// t-1 frame
		auto fh_last = frameHessians[frameHessians.size() - 2]; // host

		// Results for visualization
		std::vector<std::pair<cv::Point, Eigen::Matrix2f>> ProjectedCovariances;
		std::unordered_map<std::string, std::vector<std::pair<cv::Point, Eigen::Matrix2f>>> sigma2d_results;
		sigma2d_results["fh_last_points"] = std::vector<std::pair<cv::Point, Eigen::Matrix2f>>();

		// Load saved intensity and gradient from DSO
		Eigen::Vector3f* dI_l_last = fh_last->dIp[0];

		int count_3d_cov = 0;

		bool co_vis = true;
		for (PointHessian* ph : fh_last->pointHessians)
		{
			std::vector<Eigen::Matrix2f> cov_2ds;
			std::vector<Eigen::MatrixXf> JW_matrixes;

			float idepth = ph->idepth;
			Vec3f pixel_coords(ph->u, ph->v, 1.0f);
			Vec3f cam_point_last = K.inverse() * pixel_coords / idepth;
			Vec3f world_point = fh_last->shell->camToWorld.cast<float>() * cam_point_last;

			int u_last = std::round(ph->u);
			int v_last = std::round(ph->v);

			std::pair<int, int> cov_center_last(u_last, v_last);
			auto [cov_2d_last, valid_cov_last] = compute2DCovariance(dI_l_last, cov_center_last);
			Eigen::MatrixXf JW_last = computeCovProjection(fh_last->shell->camToWorld.cast<float>().inverse(), K, world_point);
			if (valid_cov_last)
			{
				// Eigen::MatrixXf JW_last = computeCovProjection(fh_last->shell->camToWorld.cast<float>().inverse(), K, world_point);
				cov_2ds.push_back(cov_2d_last);
				JW_matrixes.push_back(JW_last);
			}

			if (co_vis)
			{
				// Co-visible points
				for (PointFrameResidual* r : ph->residuals)
				{
					if(fh_last->shell->incoming_id > r->target->shell->incoming_id)
					{
						Vec3f cam_point_target = r->target->shell->camToWorld.cast<float>().inverse() * world_point;
						Eigen::Vector3f* dI_l_target = r->target->dIp[0];

						float Z = cam_point_target[2];
						if (Z <= 0) continue;
						
						Vec3f pixel_target = K * cam_point_target / Z;
						int u_target = std::round(pixel_target[0]);
						int v_target = std::round(pixel_target[1]);

						// Check FOV
						if (u_target >= 3 && u_target < (width - 3) && v_target >= 3 && v_target < (height - 3))
						{
							// Covariance for target
							std::pair<int, int> cov_center_target(u_target, v_target);
							auto [cov_2d_target, valid_cov_target] = compute2DCovariance(dI_l_target, cov_center_target);
							if (valid_cov_target)
							{
								Eigen::MatrixXf JW_target = computeCovProjection(r->target->shell->camToWorld.cast<float>().inverse(), K, world_point);
								cov_2ds.push_back(cov_2d_target);
								JW_matrixes.push_back(JW_target);
							}
						}
					}
				}
			}
			else
			{
				auto fh_prelast = frameHessians[frameHessians.size() - 3];
				// t-2 frame
				Vec3f cam_point_prelast = fh_prelast->shell->camToWorld.cast<float>().inverse() * world_point;
				Eigen::Vector3f* dI_l_prelast = fh_prelast->dIp[0];

				float Z = cam_point_prelast[2];
				if (Z <= 0) continue;
				
				Vec3f pixel_prelast = K * cam_point_prelast / Z;
				int u_prelast = std::round(pixel_prelast[0]);
				int v_prelast = std::round(pixel_prelast[1]);

				// Check FOV
				if (u_prelast >= 3 && u_prelast < (width - 3) && v_prelast >= 3 && v_prelast < (height - 3))
				{
					// Covariance for target
					std::pair<int, int> cov_center_prelast(u_prelast, v_prelast);
					auto [cov_2d_prelast, valid_cov_prelast] = compute2DCovariance(dI_l_prelast, cov_center_prelast);
					if (valid_cov_prelast)
					{
						Eigen::MatrixXf JW_prelast = computeCovProjection(fh_prelast->shell->camToWorld.cast<float>().inverse(), K, world_point);
						cov_2ds.push_back(cov_2d_prelast);
						JW_matrixes.push_back(JW_prelast);
					}
				}				
			}

			if (cov_2ds.size() >= 2) 
			{
				// Estimate 3D covariance for the current point
				auto [cov_3d, valid_cov_3d] = compute3DCovariance(cov_2ds, JW_matrixes);

				if (valid_cov_3d) 
				{
					count_3d_cov++;

					// Eigenvalue Decomposition of cov_3d
					Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(cov_3d);
					ph->eigenvalues = eigenSolver.eigenvalues();   // 3x1
					ph->eigenvectors = eigenSolver.eigenvectors(); // 3x3

					// // // Visualization Structures
					// Eigen::Matrix2f projectedCov_last = project3DTo2DCovariance(cov_3d, JW_last);
					// ProjectedCovariances.push_back({cv::Point(cov_center_last.first, cov_center_last.second), projectedCov_last});
					// if (valid_cov_last) {
					// 	sigma2d_results["fh_last_points"].push_back({cv::Point(cov_center_last.first, cov_center_last.second), cov_2d_last});
					// }
				} else {
					ph->eigenvalues = Eigen::Vector3f(0.0001, 0.0001, 0.0001);
					ph->eigenvectors = Eigen::Matrix3f::Identity();
				}
			} else {
				ph->eigenvalues = Eigen::Vector3f(0.0001, 0.0001, 0.0001);
				ph->eigenvectors = Eigen::Matrix3f::Identity();
			}
		}

		if (!fh_last->pointHessiansMarginalized.empty()) {
			for (PointHessian* ph : fh_last->pointHessiansMarginalized)
			{
				std::vector<Eigen::Matrix2f> cov_2ds;
				std::vector<Eigen::MatrixXf> JW_matrixes;

				float idepth = ph->idepth;
				Vec3f pixel_coords(ph->u, ph->v, 1.0f);
				Vec3f cam_point_last = K.inverse() * pixel_coords / idepth;
				Vec3f world_point = fh_last->shell->camToWorld.cast<float>() * cam_point_last;

				int u_last = std::round(ph->u);
				int v_last = std::round(ph->v);

				std::pair<int, int> cov_center_last(u_last, v_last);
				auto [cov_2d_last, valid_cov_last] = compute2DCovariance(dI_l_last, cov_center_last);
				Eigen::MatrixXf JW_last = computeCovProjection(fh_last->shell->camToWorld.cast<float>().inverse(), K, world_point);
				if (valid_cov_last)
				{
					// Eigen::MatrixXf JW_last = computeCovProjection(fh_last->shell->camToWorld.cast<float>().inverse(), K, world_point);
					cov_2ds.push_back(cov_2d_last);
					JW_matrixes.push_back(JW_last);
				}

				// if (co_vis)
				if (false)
				{
					// Co-visible points
					for (PointFrameResidual* r : ph->residuals)
					{
						if(fh_last->shell->incoming_id > r->target->shell->incoming_id)
						{
							Vec3f cam_point_target = r->target->shell->camToWorld.cast<float>().inverse() * world_point;
							Eigen::Vector3f* dI_l_target = r->target->dIp[0];

							float Z = cam_point_target[2];
							if (Z <= 0) continue;
							
							Vec3f pixel_target = K * cam_point_target / Z;
							int u_target = std::round(pixel_target[0]);
							int v_target = std::round(pixel_target[1]);

							// Check FOV
							if (u_target >= 3 && u_target < (width - 3) && v_target >= 3 && v_target < (height - 3))
							{
								// Covariance for target
								std::pair<int, int> cov_center_target(u_target, v_target);
								auto [cov_2d_target, valid_cov_target] = compute2DCovariance(dI_l_target, cov_center_target);
								if (valid_cov_target)
								{
									Eigen::MatrixXf JW_target = computeCovProjection(r->target->shell->camToWorld.cast<float>().inverse(), K, world_point);
									cov_2ds.push_back(cov_2d_target);
									JW_matrixes.push_back(JW_target);
								}
							}
						}
					}
				}
				else
				{
					auto fh_prelast = frameHessians[frameHessians.size() - 3];
					// t-2 frame
					Vec3f cam_point_prelast = fh_prelast->shell->camToWorld.cast<float>().inverse() * world_point;
					Eigen::Vector3f* dI_l_prelast = fh_prelast->dIp[0];

					float Z = cam_point_prelast[2];
					if (Z <= 0) continue;
					
					Vec3f pixel_prelast = K * cam_point_prelast / Z;
					int u_prelast = std::round(pixel_prelast[0]);
					int v_prelast = std::round(pixel_prelast[1]);

					// Check FOV
					if (u_prelast >= 3 && u_prelast < (width - 3) && v_prelast >= 3 && v_prelast < (height - 3))
					{
						// Covariance for target
						std::pair<int, int> cov_center_prelast(u_prelast, v_prelast);
						auto [cov_2d_prelast, valid_cov_prelast] = compute2DCovariance(dI_l_prelast, cov_center_prelast);
						if (valid_cov_prelast)
						{
							Eigen::MatrixXf JW_prelast = computeCovProjection(fh_prelast->shell->camToWorld.cast<float>().inverse(), K, world_point);
							cov_2ds.push_back(cov_2d_prelast);
							JW_matrixes.push_back(JW_prelast);
						}
					}				
				}

				if (cov_2ds.size() >= 2) 
				{
					// Estimate 3D covariance for the current point
					auto [cov_3d, valid_cov_3d] = compute3DCovariance(cov_2ds, JW_matrixes);

					if (valid_cov_3d) 
					{
						count_3d_cov++;

						// Eigenvalue Decomposition of cov_3d
						Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(cov_3d);
						ph->eigenvalues = eigenSolver.eigenvalues();   // 3x1
						ph->eigenvectors = eigenSolver.eigenvectors(); // 3x3

						// // // Visualization Structures
						// Eigen::Matrix2f projectedCov_last = project3DTo2DCovariance(cov_3d, JW_last);
						// ProjectedCovariances.push_back({cv::Point(cov_center_last.first, cov_center_last.second), projectedCov_last});
						// if (valid_cov_last) {
						// 	sigma2d_results["fh_last_points"].push_back({cv::Point(cov_center_last.first, cov_center_last.second), cov_2d_last});
						// }
					} else {
						ph->eigenvalues = Eigen::Vector3f(0.0001, 0.0001, 0.0001);
						ph->eigenvectors = Eigen::Matrix3f::Identity();
					}
				} else {
					ph->eigenvalues = Eigen::Vector3f(0.0001, 0.0001, 0.0001);
					ph->eigenvectors = Eigen::Matrix3f::Identity();
				}
			}
		}
		
		// std::cout << "Points #: " << fh_last->pointHessians.size() << std::endl;
		// std::cout << "3D cov #: " << count_3d_cov << std::endl;

		// // // Visualization
		// cv::Mat Image_last;
		// createImageFromIntensity(dI_l_last, width, height, Image_last);
		// std::string saveDirectory = "/home/jiung/DSO_2DGS/output";
		// std::string savePath = saveDirectory + "/last_image_with_covariances.png";
		// plotImagesWithCovariances(Image_last, sigma2d_results["fh_last_points"], ProjectedCovariances, savePath);
	}

	float fx = Hcalib.fxl();
	float fy = Hcalib.fyl();
	float cx = Hcalib.cxl();
	float cy = Hcalib.cyl();

	for(unsigned int i=0; i<frameHessians.size()-1; i++) {
		auto gt_img = frameHessians[i]->kf_img;
		Sophus::SE3d c2w = frameHessians[i]->shell->camToWorld;

		// std::cout 	<< frameHessians[i]->shell->incoming_id << "/"
		// 			<< frameHessians[i]->pointHessians.size() << "/" 
		// 			<< frameHessians[i]->pointHessiansMarginalized.size() << "/" 
		// 			<< frameHessians[i]->immaturePoints.size() << "/"
		// 			<< frameHessians[i]->pointHessians.size() + frameHessians[i]->pointHessiansMarginalized.size() + frameHessians[i]->immaturePoints.size() + frameHessians[i]->pointHessiansOut.size()
		// 			<< std::endl;

		frameHessiansFrozen[i]->kfSparseDepth = cv::Mat::zeros(gt_img->h, gt_img->w, CV_32FC1);

		// update pose
		frameHessiansFrozen[i]->camToWorld = frameHessians[i]->shell->camToWorld;

		if (i == frameHessians.size() - 2) {	// new kf + frozen kf
			FrozenFrameHessian* newFh = new FrozenFrameHessian();
			newFh->kfSparseDepth = cv::Mat::zeros(gt_img->h, gt_img->w, CV_32FC1);
			newFh->camToWorld = frameHessians[i]->shell->camToWorld;
			newFh->kfImg = frameHessians[i]->kf_img;
			newFh->incomingID = frameHessians[i]->shell->incoming_id;
			newframeHessians.push_back(newFh);
			if (!frameHessians[i]->pointHessians.empty()) {
				for (PointHessian* ph : frameHessians[i]->pointHessians) {
					// colors
					auto colors = getPointBGR(gt_img, ph->u, ph->v);
					newFh->colors.push_back(colors.at(0));
					newFh->colors.push_back(colors.at(1));
					newFh->colors.push_back(colors.at(2));

					// xyzs
					float depth = 1.0f / ph->idepth_scaled;
                    float x = ((ph->u - cx) / fx) * depth;
                    float y = ((ph->v - cy) / fy) * depth;
                    float z = depth;

					Eigen::Vector3d point_in_camera(x, y, z);
                    Eigen::Vector3d point_in_world = c2w * point_in_camera;

					newFh->pointsInWorld.push_back(point_in_world.x());
					newFh->pointsInWorld.push_back(point_in_world.y());
					newFh->pointsInWorld.push_back(point_in_world.z());

					// scales
					Eigen::Vector3f scales = ph->eigenvalues.cwiseSqrt();
					//// ascending order -> descending order
					std::vector<int> indices = {0, 1, 2};
					std::sort(indices.begin(), indices.end(), [&](int i, int j) {
						return scales(i) > scales(j);  // descending
					});
					Eigen::Vector3f scalesDesc(scales[indices[0]], scales[indices[1]], scales[indices[2]]);

					// regularize
					scalesDesc = scalesDesc / scalesDesc(0) * 0.01;
					
					newFh->scales.push_back(scalesDesc[0]);
					newFh->scales.push_back(scalesDesc[1]);
					// std::cout << scalesDesc[0] << " / " << scalesDesc[1] << " / " << scalesDesc[2] << std::endl;

					// rots
					// ascending order -> descending order
					Eigen::Matrix3f descendingEigenVectors;
					descendingEigenVectors.col(0) = ph->eigenvectors.col(indices[0]);
					descendingEigenVectors.col(1) = ph->eigenvectors.col(indices[1]);
					descendingEigenVectors.col(2) = ph->eigenvectors.col(indices[2]);
					Eigen::Quaternionf qfrommat(descendingEigenVectors);
					qfrommat.normalize();
					newFh->rots.push_back(qfrommat.w());
					newFh->rots.push_back(qfrommat.x());
					newFh->rots.push_back(qfrommat.y());
					newFh->rots.push_back(qfrommat.z());

					// debug
					//// original
					// Eigen::Vector3f original_scales = ph->eigenvalues.cwiseSqrt();
					// Eigen::Matrix3f original_vecs = ph->eigenvectors;
					// Eigen::Matrix3f original_mat = original_vecs * original_scales.asDiagonal() * original_vecs.transpose();

					// Eigen::Vector3f reverse_scales(scales.z(), scales.y(), scales.x());
					// Eigen::Matrix3f reverse_mat = descendingEigenVectors * reverse_scales.asDiagonal() * descendingEigenVectors.transpose();

					// std::cout << original_mat.matrix() << " / " << std::endl;
					// std::cout << reverse_mat.matrix() << std::endl;
					// std::cout << "" << std::endl;

					// sparse_depth
					newFh->kfSparseDepth.at<float>(ph->v, ph->u) = 1. / ph->idepth_scaled;
					frameHessiansFrozen[i]->kfSparseDepth.at<float>(ph->v, ph->u) = 1. / ph->idepth_scaled;
				}
			}
			if (!frameHessians[i]->pointHessiansMarginalized.empty()) {
			// if (false) {
				for (PointHessian* ph : frameHessians[i]->pointHessiansMarginalized) {
					// colors
					auto colors = getPointBGR(gt_img, ph->u, ph->v);
					newFh->colors.push_back(colors.at(0));
					newFh->colors.push_back(colors.at(1));
					newFh->colors.push_back(colors.at(2));

					// xyzs
					float depth = 1.0f / ph->idepth_scaled;
                    float x = ((ph->u - cx) / fx) * depth;
                    float y = ((ph->v - cy) / fy) * depth;
                    float z = depth;

					Eigen::Vector3d point_in_camera(x, y, z);
                    Eigen::Vector3d point_in_world = c2w * point_in_camera;

					newFh->pointsInWorld.push_back(point_in_world.x());
					newFh->pointsInWorld.push_back(point_in_world.y());
					newFh->pointsInWorld.push_back(point_in_world.z());

					// scales
					Eigen::Vector3f scales = ph->eigenvalues.cwiseSqrt();
					//// ascending order -> descending order
					std::vector<int> indices = {0, 1, 2};
					std::sort(indices.begin(), indices.end(), [&](int i, int j) {
						return scales(i) > scales(j);  // descending
					});
					Eigen::Vector3f scalesDesc(scales[indices[0]], scales[indices[1]], scales[indices[2]]);

					// regularize
					scalesDesc = scalesDesc / scalesDesc(0) * 0.01;
					
					newFh->scales.push_back(scalesDesc[0]);
					newFh->scales.push_back(scalesDesc[1]);
					// std::cout << scalesDesc[0] << " / " << scalesDesc[1] << " / " << scalesDesc[2] << std::endl;

					// rots
					// ascending order -> descending order
					Eigen::Matrix3f descendingEigenVectors;
					descendingEigenVectors.col(0) = ph->eigenvectors.col(indices[0]);
					descendingEigenVectors.col(1) = ph->eigenvectors.col(indices[1]);
					descendingEigenVectors.col(2) = ph->eigenvectors.col(indices[2]);
					Eigen::Quaternionf qfrommat(descendingEigenVectors);
					qfrommat.normalize();
					newFh->rots.push_back(qfrommat.w());
					newFh->rots.push_back(qfrommat.x());
					newFh->rots.push_back(qfrommat.y());
					newFh->rots.push_back(qfrommat.z());

					// sparse_depth
					newFh->kfSparseDepth.at<float>(ph->v, ph->u) = 1. / ph->idepth_scaled;
					frameHessiansFrozen[i]->kfSparseDepth.at<float>(ph->v, ph->u) = 1. / ph->idepth_scaled;
				}
			}	
		} else { 		// only frozen kf
			if (!frameHessians[i]->pointHessians.empty()) {
				for (PointHessian* ph : frameHessians[i]->pointHessians) {
					frameHessiansFrozen[i]->kfSparseDepth.at<float>(ph->v, ph->u) = 1. / ph->idepth_scaled;
				}
			}
			if (!frameHessians[i]->pointHessiansMarginalized.empty()) {
				for (PointHessian* ph : frameHessians[i]->pointHessiansMarginalized) {
					frameHessiansFrozen[i]->kfSparseDepth.at<float>(ph->v, ph->u) = 1. / ph->idepth_scaled;
				}
			}

			// // for figure
			// cv::Mat depth_cv_normalized, depth_clipped;
			// // clip depth
			// cv::threshold(frameHessiansFrozen[i]->kfSparseDepth, depth_clipped, 10, 10, cv::THRESH_TRUNC);
			// cv::threshold(depth_clipped, depth_clipped, 0, 0, cv::THRESH_TOZERO);

			// // cv::normalize(depth_clipped, depth_cv_normalized, 0, 255, cv::NORM_MINMAX);
			// // depth_cv_normalized.convertTo(depth_cv_normalized, CV_8UC1);
			// // cv::imwrite(result_depth_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".png"), depth_cv_normalized);

			// depth_clipped *= 6553.5;
			// depth_clipped.convertTo(depth_clipped, CV_16U);
			// cv::imwrite("/home/riboha/DSO_2DGS/experiments_bash/results/demo/" + (std::to_string(frameHessiansFrozen[i]->incomingID) + ".png"), depth_clipped);
		}
	}
	printLogLine();
    //printEigenValLine();
}


Eigen::Matrix2f FullSystem::project3DTo2DCovariance(const Eigen::Matrix3f& cov_3d, const Eigen::MatrixXf& T_last) {
    // Extract the first two rows of T_last
    Eigen::Matrix<float, 2, 3> J = T_last.block<2, 3>(0, 0);

    // Project 3D covariance to 2D
    Eigen::Matrix2f cov_2d_projected = J * cov_3d * J.transpose();
    return cov_2d_projected;
}


void FullSystem::drawEllipse(cv::Mat& image, const cv::Point& center, const Eigen::Matrix2f& covariance, const cv::Scalar& color, int thickness) {
    // Ensure covariance is float
    Eigen::Matrix2f covariance_f = covariance;

    // Compute eigenvalues and eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigensolver(covariance_f);
    Eigen::Vector2f eigenvalues = eigensolver.eigenvalues();
    Eigen::Matrix2f eigenvectors = eigensolver.eigenvectors();

    // Compute angle of rotation
    float angle = atan2(eigenvectors(1, 0), eigenvectors(0, 0)) * 180.0 / M_PI;

    // Clamp eigenvalues to avoid negative or invalid square roots
    float width = 2.0f * sqrt(std::max(0.0f, static_cast<float>(eigenvalues[0])));
    float height = 2.0f * sqrt(std::max(0.0f, static_cast<float>(eigenvalues[1])));

    // Ensure thickness is within valid range
    thickness = std::clamp(thickness, 1, 10);

    // Check for valid axes dimensions
    if (width <= 0 || height <= 0) {
        std::cerr << "Invalid ellipse dimensions: width = " << width << ", height = " << height << std::endl;
        return;
    }

    // Draw the ellipse
    ellipse(image, center, cv::Size(static_cast<int>(width / 2), static_cast<int>(height / 2)),
            angle, 0, 360, color, thickness);
}


void FullSystem::plotImagesWithCovariances(const std::vector<cv::Mat>& images,
                                           const std::vector<std::vector<std::pair<int, int>>>& imagePointsList,
                                           const std::unordered_map<std::string, std::vector<Eigen::Matrix2f>>& covariancesList,
                                           const std::vector<std::vector<Eigen::Matrix2f>>& allProjectedCovariances) {
    for (size_t i = 0; i < images.size(); ++i) {
        cv::Mat displayImage;
        cvtColor(images[i], displayImage, cv::COLOR_GRAY2BGR);

        const auto& points = imagePointsList[i];
        const auto& covariances = covariancesList.at(i == 0 ? "fh_last_points" : "fh_prelast_points");
        const auto& projectedCovariances = allProjectedCovariances[i];

        for (size_t j = 0; j < points.size(); ++j) {
            cv::Point center(static_cast<int>(points[j].first), static_cast<int>(points[j].second));

            // Original 2D covariance (blue)
            drawEllipse(displayImage, center, covariances[j], cv::Scalar(255, 0, 0), 2);

            // Projected 2D covariance (green)
            drawEllipse(displayImage, center, projectedCovariances[j], cv::Scalar(0, 255, 0), 2);
        }

        std::string windowName = (i == 0) ? "Last Image" : "Pre-last Image";
        cv::imshow(windowName, displayImage);
    }

    cv::waitKey(0);
}


void FullSystem::createImageFromIntensity(const Eigen::Vector3f* dI_l, int width, int height, cv::Mat& outputImage) {
    // Initialize the output image
    outputImage = cv::Mat::zeros(height, width, CV_8UC1);

    // Iterate through all pixels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = x + y * width;

            // Get the intensity from dI_l[idx][0]
            float f_xy = static_cast<float>(dI_l[idx][0]);
            uchar intensity = static_cast<uchar>(f_xy);

            // Set the pixel value
            outputImage.at<uchar>(y, x) = intensity;
        }
    }
}


std::pair<Eigen::Matrix2f, bool> FullSystem::compute2DCovariance(const Eigen::Vector3f* dI_l, const std::pair<int, int>& cov_center) {
    // Pattern offsets (equivalent to 8 points in DSO)
    std::vector<std::pair<int, int>> pattern_offsets = {
        {0, -2}, {-1, -1}, {1, -1},
        {-2, 0}, {0, 0}, {2, 0},
        {-1, 1}, {0, 2}
    };

    std::vector<Eigen::Vector2f> gradients;
    std::vector<Eigen::Vector2f> z_vectors;

	int w0 = coarseTracker_forNewKF->w[0];

    for (const auto& offset : pattern_offsets) {
        int x = cov_center.first + offset.first;
        int y = cov_center.second + offset.second;

        int idx = x + y * w0;
        float f_xy = dI_l[idx][0];
        if (f_xy == 0) continue;

        // Compute ∇f / f
        Eigen::Vector2f grad_div_f(
            dI_l[idx][1] / f_xy,
            dI_l[idx][2] / f_xy
        );
        gradients.push_back(grad_div_f);

        // Compute z vector
        Eigen::Vector2f z(x - cov_center.first, y - cov_center.second);
        z_vectors.push_back(z);
    }

	if (gradients.empty() && z_vectors.empty()) {
        std::vector<std::pair<int, int>> fallback_pattern_offsets = {
            {2, 1}, {0, 0}, {-1, 2}, {-2, 1},
            {-2, -1}, {-1, -2}, {2, -1}, {1, -2}
        };
        for (const auto& offset : fallback_pattern_offsets) {
            int x = cov_center.first + offset.first;
            int y = cov_center.second + offset.second;
            int idx = x + y * w0;
            float f_xy = dI_l[idx][0];
            if (f_xy == 0) continue;
            // Compute ∇f / f
            Eigen::Vector2f grad_div_f(
                dI_l[idx][1] / f_xy,
                dI_l[idx][2] / f_xy
            );
            gradients.push_back(grad_div_f);
            // Compute z vector
            Eigen::Vector2f z(x - cov_center.first, y - cov_center.second);
            z_vectors.push_back(z);
        }
    }

    // not enough to calculate covariance
    if (gradients.empty() || z_vectors.empty()) {
        return {Eigen::Matrix2f::Zero(), false};
    }


    // Construct Z matrix
    Eigen::MatrixXf Z(2 * z_vectors.size(), 1);
    for (size_t i = 0; i < z_vectors.size(); ++i) {
        Z(2 * i, 0) = z_vectors[i][0];
        Z(2 * i + 1, 0) = z_vectors[i][1];
    }

    // Construct G matrix
    Eigen::MatrixXf G(2 * gradients.size(), 3);
    for (size_t i = 0; i < gradients.size(); ++i) {
        float g1 = gradients[i][0];
        float g2 = gradients[i][1];

        G(2 * i, 0) = -g1;
        G(2 * i, 1) = 0;
        G(2 * i, 2) = -g2;

        G(2 * i + 1, 0) = 0;
        G(2 * i + 1, 1) = -g2;
        G(2 * i + 1, 2) = -g1;
    }

    // Solve for p = [a, b, c]
	Eigen::VectorXf p = G.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Z);
    float a = p[0], b = p[1], c = p[2];

    // Construct covariance matrix
    Eigen::Matrix2f cov;
    cov << a, c,
           c, b;

    // Check for positive semi-definiteness
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig_solver(cov);
    Eigen::Vector2f eigenvalues = eig_solver.eigenvalues();

    if (cov != Eigen::Matrix2f::Zero()) {
        if (eigenvalues.minCoeff() < 0) {
            // Fix covariance matrix
            cov += Eigen::Matrix2f::Identity() * 1e-6;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> fixed_solver(cov);
            Eigen::Vector2f fixed_eigenvalues = fixed_solver.eigenvalues();

            if (fixed_eigenvalues.minCoeff() >= 0) {
                return {cov, true};
            } else {
                Eigen::Matrix2f clipped_cov = eig_solver.eigenvectors() * 
                                              eigenvalues.cwiseMax(1e-6).asDiagonal() *
                                              eig_solver.eigenvectors().transpose();
				Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> fixed_solver(clipped_cov);
				Eigen::Vector2f fixed_eigenvalues = fixed_solver.eigenvalues();
				if (fixed_eigenvalues.minCoeff() >= 0) {
					return {clipped_cov, true};
				}
				else{
					// std::cerr << "Failed Covariance: \n" << cov << std::endl;
					return {cov, false};
				}
            }
        }
        return {cov, true};
    }
    // std::cerr << "Failed Covariance: \n" << cov << std::endl;
    return {cov, false};
}


Eigen::MatrixXf FullSystem::computeCovProjection(const Sophus::SE3f& w2c_extrinsic, const Mat33f& camera_matrix, const Eigen::Vector3f& point_world) {
    // Extract rotation matrix (R) and translation vector (t)
    Eigen::Matrix3f rotation_matrix = w2c_extrinsic.rotationMatrix();
    Eigen::Vector3f translation_vector = w2c_extrinsic.translation();

    Eigen::Vector3f point_camera = rotation_matrix * point_world + translation_vector;

    float x = point_camera(0);
    float y = point_camera(1);
    float z = point_camera(2);

    if (z <= 0) {
        throw std::runtime_error("Point is behind the camera or on the image plane.");
    }

	float fx = camera_matrix(0, 0);
	float fy = camera_matrix(1, 1);
	float cx = camera_matrix(0, 2);
	float cy = camera_matrix(1, 2);

    // Compute Jacobian matrix (J)
    Eigen::MatrixXf jacobian(2, 3);
    jacobian << fx / z, 0, -fx * x / (z * z),
                0, fy / z, -fy * y / (z * z);

    // Compute J * R
    Eigen::MatrixXf jr_matrix = jacobian * rotation_matrix;

    return jr_matrix;
}


std::pair<Eigen::Matrix3f, bool> FullSystem::compute3DCovariance(
    const std::vector<Eigen::Matrix2f>& Sigma_Is, 
    const std::vector<Eigen::MatrixXf>& JWs) 
{

    // Flatten the 2x2 covariance matrices into a combined b vector
    Eigen::VectorXf b(4 * Sigma_Is.size());
    for (size_t i = 0; i < Sigma_Is.size(); ++i) {
        Eigen::VectorXf bi = Eigen::Map<const Eigen::VectorXf>(Sigma_Is[i].data(), Sigma_Is[i].size());
        b.segment<4>(4 * i) = bi;
    }

    // Function to construct the coefficient matrix A (4x6 for each JW)
    auto constructA = [](const Eigen::MatrixXf& J) {
        Eigen::MatrixXf A(4, 6);
        Eigen::Vector3f J0 = J.row(0);
        Eigen::Vector3f J1 = J.row(1);

        A.row(0) << J0[0] * J0[0], J0[1] * J0[1], J0[2] * J0[2], J0[0] * J0[1], J0[0] * J0[2], J0[1] * J0[2];
        A.row(1) << J1[0] * J1[0], J1[1] * J1[1], J1[2] * J1[2], J1[0] * J1[1], J1[0] * J1[2], J1[1] * J1[2];
        A.row(2) << J0[0] * J1[0], J0[1] * J1[1], J0[2] * J1[2], J0[0] * J1[1] + J1[0] * J0[1], J0[0] * J1[2] + J1[0] * J0[2], J0[1] * J1[2] + J1[1] * J0[2];
        A.row(3) << J1[0] * J0[0], J1[1] * J0[1], J1[2] * J0[2], J1[0] * J0[1] + J0[0] * J1[1], J1[0] * J0[2] + J0[0] * J1[2], J1[1] * J0[2] + J0[1] * J1[2];

        return A;
    };

    // Combine A matrices for all JWs
    Eigen::MatrixXf A(4 * JWs.size(), 6);
    for (size_t i = 0; i < JWs.size(); ++i) {
        A.block<4, 6>(4 * i, 0) = constructA(JWs[i]);
    }

    // Solve for s (6x1 vector)
    Eigen::VectorXf s = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    // Reconstruct the 3D covariance matrix
    Eigen::Matrix3f Sigma_W;
    Sigma_W << s(0), s(3) / 2, s(4) / 2,
               s(3) / 2, s(1), s(5) / 2,
               s(4) / 2, s(5) / 2, s(2);

    // Check for positive semi-definiteness
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig_solver(Sigma_W);
    Eigen::Vector3f eigenvalues = eig_solver.eigenvalues();

    if (Sigma_W != Eigen::Matrix3f::Zero()) {
        if (eigenvalues.minCoeff() < 0) {
            // Fix covariance matrix
            Sigma_W += Eigen::Matrix3f::Identity() * 1e-6;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> fixed_solver(Sigma_W);
            Eigen::Vector3f fixed_eigenvalues = fixed_solver.eigenvalues();

            if (fixed_eigenvalues.minCoeff() >= 0) {
                return {Sigma_W, true};
            } else {
                Eigen::Matrix3f clipped_cov = eig_solver.eigenvectors() * 
                                              eigenvalues.cwiseMax(1e-6).asDiagonal() *
                                              eig_solver.eigenvectors().transpose();
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> fixed_solver(clipped_cov);
                Eigen::Vector3f fixed_eigenvalues = fixed_solver.eigenvalues();
                if (fixed_eigenvalues.minCoeff() >= 0) {
                    return {clipped_cov, true};
                } else {
                    return {Sigma_W, false};
                }
            }
        }
        return {Sigma_W, true};
    }
    return {Sigma_W, false};
}


void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
	firstFrame->idx = frameHessians.size();
	frameHessians.push_back(firstFrame);
	firstFrame->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(firstFrame->shell);
	ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues();

	// add new frozen fh, for 2DGS
	{
		boost::unique_lock<boost::mutex> frozenKFLock(frozenMapMutex);
		FrozenFrameHessian* frozenFh = new FrozenFrameHessian();
		frozenFh->camToWorld = firstFrame->shell->camToWorld;
		frozenFh->kfImg = firstFrame->kf_img;
		frozenFh->incomingID = firstFrame->shell->incoming_id;
		frameHessiansFrozen.push_back(frozenFh);
	}

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

	firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);


	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		sumID += coarseInitializer->points[0][i].iR;
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);

	// randomly sub-select the points I need.
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		if(rand()/(float)RAND_MAX > keepPercentage) continue;

		Pnt* point = coarseInitializer->points[0]+i;
		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);

		if(!std::isfinite(pt->energyTH)) { delete pt; continue; }


		pt->idepth_max=pt->idepth_min=1;
		PointHessian* ph = new PointHessian(pt, &Hcalib);
		delete pt;
		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

		ph->setIdepthScaled(point->iR*rescaleFactor);
		ph->setIdepthZero(ph->idepth);
		ph->hasDepthPrior=true;
		ph->setPointStatus(PointHessian::ACTIVE);

		firstFrame->pointHessians.push_back(ph);
		ef->insertPoint(ph);
	}



	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;


	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->camToWorld = SE3();
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	initialized=true;
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
	pixelSelector->allowFast = true;
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

	newFrame->pointHessians.reserve(numPointsTotal*1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);


	for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
	for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
	{
		int i = x+y*wG[0];
		if(selectionMap[i]==0) continue;

		ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
		if(!std::isfinite(impt->energyTH)) delete impt;
		else newFrame->immaturePoints.push_back(impt);

	}
	//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}



void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size());
		for(unsigned int i=0;i<frameHessians.size();i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
	}

	ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l().a << " "  <<
				frameHessians.back()->aff_g2l().b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}


}



void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}

std::vector<float> FullSystem::getPointBGR(MinimalImageB3* img, float u, float v) {
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



}
