/************************************************************************
*                                                                       *
* TAPSim - an atom probe data simulation program                        *
*                                                                       *
* Copyright (C) 2011 Christian Oberdorfer                               *
*                                                                       *
* This program is free software: you can redistribute it and/or modify  *
* it under the terms of the GNU General Public License as published by  *
* the Free Software Foundation, either version 3 of the License, or any *
* any later version.                                                    *
*                                                                       *
* This program is distributed in the hope that it will be useful,       *
* but WITHOUT ANY WARRANTY; without even the implied warranty of        *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
* GNU General Public License for more details.                          *
*                                                                       *
* You should have received a copy of the GNU General Public License     *
* along with this program.  If not, see 'http://www.gnu.org/licenses'   *
*                                                                       *
************************************************************************/ 

#include "process.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include <queue>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <random>

#include "../vector/vector.h"

#include "commands.h"
#include "trajectory_3d.h"
#include "file_io.h"
#include "logResults.h"
#include "logTrajectory.h"
#include "logGrid.h"
#include "logSurface.h"
#include "info.h"
#include "debug.h"

//#define VERBOSE

void Process::initialization(const char* configFile, const char* nodeFile, System_3d* system)
{
	time_t startTime;

	info::begin() << "Reading configuration from file \"" << configFile << "\"" << std::endl;
	File_Io::readConfig(configFile,&system->configTable);

	info::begin() << "\t-> temperature: " << system->configTable.temperature() << "K" << std::endl;
	info::begin() << "\t-> node-types / id-values:" << std::endl;
	const std::list<Configuration::NodeId> ids = system->configTable.ids();
	for (std::list<Configuration::NodeId>::const_iterator i = ids.begin(); i != ids.end(); i++)
		info::begin() << "\t\t - " << system->configTable[*i].name() << ": #" << i->toValue() << std::endl;
	info::begin() << "\t-> total number of types: " << ids.size() << std::endl;
	
	info::begin() << "Reading node data from file \"" << nodeFile << "\"" << std::endl;

	{
		int withPotentials;
		File_Io::readGeomGrid(nodeFile,&system->geomTable,&system->gridTable,0,&withPotentials);
		
		if (!withPotentials)
		{
			for (int i = 0; i < system->geomTable.numNodes(); i++)
			{
				const float phi = system->configTable[system->gridTable.id(i)].phi();

				system->gridTable.node(i).setPhi(0,phi);
				system->gridTable.node(i).setPhi(1,phi);
			}

			info::begin() << "\t-> initializing potentials with preset values from configuration" << std::endl;
		}
		else
			info::begin() << "\t-> setting included potential values" << std::endl;
	}
	
	std::map<Configuration::NodeId,int> nodeStatistics;
	for (int i = 0; i < system->gridTable.numNodes(); i++)
		nodeStatistics[system->gridTable.id(i)]++;
	
	info::begin() << "\t-> node statistics:" << std::endl;
	for (std::map<Configuration::NodeId,int>::const_iterator i = nodeStatistics.begin(); i != nodeStatistics.end(); i++)
		info::begin() << "\t\t - " << system->configTable[i->first].name() << ": " << i->second << std::endl;
	info::begin() << "\t-> total number of nodes: " << system->geomTable.numNodes() << std::endl;

	info::begin() << "\t-> maximum (automatic) assigned unique number: ";
	if (system->gridTable.numNodes() > 0)
	{
		Configuration::NodeNumber maxNumber = system->gridTable.number(0);
		if (!maxNumber.isValid()) throw std::runtime_error("Inconsistent node numbering!");

		for (int i = 0; i < system->gridTable.numNodes(); i++)
			if (system->gridTable.number(i) > maxNumber) maxNumber = system->gridTable.number(i);

		info::out() << maxNumber.toValue() << std::endl;
	}
	else
		info::out() << "none" << std::endl;
	
	startTime = std::time(0);

	system->geomTable.tetrahedralize();

	info::begin() << "Computing delaunay tetrahedralization ";
	info::out() << "(elapsed time = " << std::difftime(std::time(0),startTime) / 60.0f;
	info::out() << " min)" << std::endl;

	info::begin() << "\t-> number of tetrahedra: " << system->geomTable.numTetrahedra() << std::endl;

	info::begin() << "\t-> geometric extents: ";
	info::out() << system->geomTable.min() << " - " << system->geomTable.max();
	info::out() << std::endl;

	info::begin() << "\t-> constrained geometric extents: ";
	info::out() << system->geomTable.constrainedMin() << " - " << system->geomTable.constrainedMax();
	info::out() << std::endl;
	
	// ***

	startTime = std::time(0);

	system->gridTable.fastSync(system->geomTable,system->configTable);

	info::begin() << "Syncronizing data structures ";
	info::out() << "(elapsed time = " << std::difftime(std::time(0),startTime) / 60.0f;
	info::out() << " min)" << std::endl;
}

// ***

void Process::relaxation(const float threshold, const unsigned int cycleSize, const unsigned int queueSize, System_3d* system)
{
	info::open("relaxation");

	info::begin() << "Error threshold: " << threshold << std::endl;
	info::begin() << "Number of iterations per cycle: " << cycleSize << std::endl;
	info::begin() << "Deviation queue-size: " << queueSize << std::endl;
	
	time_t startTime = std::time(0);

	try
	{
		signed long stepCnt = system->gridTable.relax(threshold,cycleSize,queueSize);
		info::begin() << "Iteration steps: " << stepCnt << std::endl;
	}
	catch (Grid_3d::Table::RelaxationFault& error)
	{
		info::begin() << "Relaxing potential => numerical limit reached: " << error.iteration() << " steps ";
		info::out() << "(deviation = " << error.deviation() << ", slope = " << error.slope() << ")" << std::endl;
	}

	info::begin() << "Elapsed time: " << std::difftime(std::time(0),startTime) / 60.0 << " min" << std::endl;

	info::close("relaxation");
}

// ***

Process::EvaporationOptions::EvaporationOptions()
	: resultsFile(),
	  resultsMode(0),
	  resultsChunkSize(0),
	  trajectoryFile(),
	  trajectoryMode(0),
	  trajectoryChunkSize(0),
	  gridFile(),
	  gridMode(0),
	  gridInterval(0),
	  surfaceFile(),
	  surfaceMode(0),
	  surfaceInterval(0),
	  geometryFile(),
	  geometryParams(0),
	  geometryMode(0),
	  dumpFile(),
	  dumpInterval(0),
	  delayTime(0),

	//////////////////////////////////////////////////////
	//                                                  //
	//                     change                       //
	//                                                  //
	//////////////////////////////////////////////////////

	  probMode(0),

	//////////////////////////////////////////////////////
	//                                                  //
	//                     change                       //
	//                                                  //
	//////////////////////////////////////////////////////

	  evapMode(0),


	  vacuumName(),
	  trajectoryIntegrator(0),
	  trajectoryStepper(0),
	  trajectoryTimeStep(0.0f),
	  trajectoryTimeStepLimit(0.0f),
	  trajectoryErrorThreshold(0.0f),
	  fixedInitialPosition(0),
	  noInitialVelocity(0),
	  initShrinkage(0.0f),
	  shrinkLimit(0.0f),
	  initEventCnt(0),
	  eventCntLimit(0),
	  voltageQueueSize(0),
	  relaxThreshold(0.0f),
	  relaxShellSize(0),
	  relaxCycleSize(0),
	  relaxQueueSize(0),
	  relaxGlobalCycles(0),
	  refreshInterval(0),
	  refreshThreshold(0.0f),
	  refreshCycleSize(0),
	  refreshQueueSize(0)
{}

void Process::EvaporationOptions::setDefaults(const char* filename, EvaporationOptions* obj)
{
	std::map<std::string,std::string> defaults;

	if (std::strlen(filename) == 0)
	{
		// fall back on internal defaults:
		
		std::list<File_Io::KeyValue> defaultParams;
		makeIniList(&defaultParams);

		for (std::list<File_Io::KeyValue>::const_iterator i = defaultParams.begin(); i != defaultParams.end(); i++)
		{
			if (i->first.empty()) continue;
			const std::pair<std::map<std::string,std::string>::iterator,bool> result = defaults.insert(*i);
			
			if (!result.second) throw std::runtime_error("Process::EvaporationOptions::setDefaults(): duplicate default keys!"); 
		}
	}
	else
		File_Io::readInitialization(filename,&defaults," = ");

	// ***

	std::map<std::string,std::string>::iterator entry;

	entry = defaults.find("RESULTS_FILENAME");
	if (defaults.end() == entry)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'RESULTS_FILENAME' error!");
	else
		obj->resultsFile = entry->second;

	entry = defaults.find("RESULTS_BINARY_OUTPUT");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->resultsMode) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'RESULTS_BINARY_OUTPUT' error!");

	entry = defaults.find("RESULTS_CHUNK_SIZE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->resultsChunkSize) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'RESULTS_CHUNK_SIZE' error!");

	// ***

	entry = defaults.find("TRAJECTORY_FILENAME");
	if (defaults.end() == entry)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_FILENAME' error!");
	else
		obj->trajectoryFile = entry->second;

	entry = defaults.find("TRAJECTORY_BINARY_OUTPUT");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->trajectoryMode) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_BINARY_OUTPUT' error!");

	entry = defaults.find("TRAJECTORY_CHUNK_SIZE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->trajectoryChunkSize) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_CHUNK_SIZE' error!");

	// ***
	
	entry = defaults.find("GRID_FILENAME");
	if (defaults.end() == entry)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'GRID_FILENAME' error!");
	else
		obj->gridFile = entry->second;

	entry = defaults.find("GRID_BINARY_OUTPUT");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->gridMode) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'GRID_BINARY_OUTPUT' error!");

	entry = defaults.find("GRID_INTERVAL");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->gridInterval) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'GRID_INTERVAL' error!");

	// ***

	entry = defaults.find("SURFACE_FILENAME");
	if (defaults.end() == entry)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'SURFACE_FILENAME' error!");
	else
		obj->surfaceFile = entry->second;

	entry = defaults.find("SURFACE_BINARY_OUTPUT");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->surfaceMode) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'SURFACE_BINARY_OUTPUT' error!");

	entry = defaults.find("SURFACE_INTERVAL");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->surfaceInterval) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'SURFACE_INTERVAL' error!");

	// ***

	entry = defaults.find("OUTPUT_DELAY_TIME");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->delayTime) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'OUTPUT_DELAY_TIME' error!");

	// ***

	entry = defaults.find("GEOMETRY_FILENAME");
	if (defaults.end() == entry)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'GEOMETRY_FILENAME' error!");
	else
		obj->geometryFile = entry->second;

	
	entry = defaults.find("GEOMETRY_BINARY_OUTPUT");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->geometryMode) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'GEOMETRY_BINARY_OUTPUT' error!");

	{
		unsigned int hexValue;
		
		entry = defaults.find("GEOMETRY_CONTENTS");
		if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%x",&hexValue) != 1)
			throw std::runtime_error("EvaporationOptions::setDefaults(): 'GEOMETRY_CONTENTS' error!");

		if (std::numeric_limits<unsigned char>::max() >= hexValue)
			obj->geometryParams = hexValue;
		else
			throw std::runtime_error("EvaporationOptions::setDefaults(): 'GEOMETRY_CONTENTS' error!");
	}

	// ***

	entry = defaults.find("DUMP_FILENAME");
	if (defaults.end() == entry)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'DUMP_FILENAME' error!");
	else
		obj->dumpFile = entry->second;

	entry = defaults.find("DUMP_INTERVAL");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->dumpInterval) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'DUMP_INTERVAL' error!");

	// ***

	entry = defaults.find("METHOD_EVAPORATION_PROBABILITY");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->probMode) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'METHOD_EVAPORATION_PROBABILITY' error!");

	//////////////////////////////////////////////////////
	//                                                  //
	//                     change                       //
	//                                                  //
	//////////////////////////////////////////////////////

	obj->probMode =0	;

	entry = defaults.find("METHOD_EVAPORATION_CHOICE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->evapMode) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'METHOD_EVAPORATION_CHOICE' error!");


	//////////////////////////////////////////////////////
	//                                                  //
	//                     change                       //
	//                                                  //
	//////////////////////////////////////////////////////

	obj->evapMode = 2;
	obj->evapMode = 0;
	//evapMode = 2;

	// ***
	
	entry = defaults.find("VACUUM_CELL_IDENTIFIER");
	if (defaults.end() == entry)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'VACUUM_CELL_IDENTIFIER' error!");
	else
		obj->vacuumName = entry->second;

	// ***

	entry = defaults.find("TRAJECTORY_INTEGRATOR_TYPE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->trajectoryIntegrator) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_INTEGRATOR_TYPE' error!");

	entry = defaults.find("TRAJECTORY_STEPPER_TYPE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->trajectoryStepper) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_STEPPER_TYPE' error!");

	entry = defaults.find("TRAJECTORY_INITIAL_TIME_STEP");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%f",&obj->trajectoryTimeStep) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_INITIAL_TIME_STEP' error!");

	entry = defaults.find("TRAJECTORY_MINIMUM_TIME_STEP");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%f",&obj->trajectoryTimeStepLimit) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_MINIMUM_TIME_STEP' error!");

	entry = defaults.find("TRAJECTORY_ERROR_THRESHOLD");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%f",&obj->trajectoryErrorThreshold) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_ERROR_THRESHOLD' error!");

	entry = defaults.find("TRAJECTORY_NON_RANDOM_START_POSITION");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->fixedInitialPosition) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_NON_RANDOM_START_POSITION' error!");

	entry = defaults.find("TRAJECTORY_NO_INITIAL_VELOCITY");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->noInitialVelocity) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'TRAJECTORY_NO_INITIAL_VELOCITY' error!");

	// ***
	
	obj->initShrinkage = 0.0f;
	obj->shrinkLimit = std::numeric_limits<float>::max();
	
	obj->initEventCnt = 0;
	obj->eventCntLimit = std::numeric_limits<unsigned int>::max();

	// ***

	entry = defaults.find("VOLTAGE_QUEUE_SIZE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->voltageQueueSize) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'VOLTAGE_QUEUE_SIZE' error!");
	
	// ***

	entry = defaults.find("LOCAL_RELAXATION_ERROR_THRESHOLD");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%f",&obj->relaxThreshold) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'LOCAL_RELAXATION_ERROR_THRESHOLD' error!");

	entry = defaults.find("LOCAL_RELAXATION_SHELL_SIZE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->relaxShellSize) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'LOCAL_RELAXATION_SHELL_SIZE' error!");

	entry = defaults.find("LOCAL_RELAXATION_CYCLE_SIZE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->relaxCycleSize) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'LOCAL_RELAXATION_CYCLE_SIZE' error!");

	entry = defaults.find("LOCAL_RELAXATION_QUEUE_SIZE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->relaxQueueSize) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'LOCAL_RELAXATION_QUEUE_SIZE' error!");

	entry = defaults.find("GLOBAL_RELAXATION_STEPS");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->relaxGlobalCycles) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'GLOBAL_RELAXATION_STEPS' error!");
	
	// ***

	entry = defaults.find("REFRESH_RELAXATION_INTERVAL");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%d",&obj->refreshInterval) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'REFRESH_RELAXATION_INTERVAL' error!");

	entry = defaults.find("REFRESH_RELAXATION_ERROR_THRESHOLD");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%f",&obj->refreshThreshold) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'REFRESH_RELAXATION_ERROR_THRESHOLD' error!");

	entry = defaults.find("REFRESH_RELAXATION_CYCLE_SIZE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->refreshCycleSize) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'REFRESH_RELAXATION_CYCLE_SIZE' error!");

	entry = defaults.find("REFRESH_RELAXATION_QUEUE_SIZE");
	if (defaults.end() == entry || std::sscanf(entry->second.c_str(),"%u",&obj->refreshQueueSize) != 1)
		throw std::runtime_error("EvaporationOptions::setDefaults(): 'REFRESH_RELAXATION_QUEUE_SIZE' error!");
}



void Process::evaporation(const EvaporationOptions& options, const std::string& outputHeader, System_3d* system)
{
	info::open("evaporation-sequence");

	// *** setup recording of evaporation event data

		File_Io::LogResults resultsHandle;

		if (!options.resultsFile.empty())
		{
			resultsHandle.init(options.resultsFile.c_str(),options.resultsMode,options.delayTime,options.resultsChunkSize);
			resultsHandle.setHeader(outputHeader.c_str());
			
			
			info::begin() << "Logging evaporation events in file: \"" << options.resultsFile << "\"";

			info::out() << " (";

			if (options.resultsMode == File_Io::ASCII)
				info::out() << "ascii mode"; 
			else
				info::out() << "binary mode";

			if (options.resultsChunkSize > 0)
				info::out() << ", chunk-size = " << options.resultsChunkSize; 
			else 
				info::out() << "no chunks";

			info::out() << ")" << std::endl;
		}

	// *** setup recording of trajectory data

		File_Io::LogTrajectory trajectoryHandle;

		if (!options.trajectoryFile.empty())
		{
			trajectoryHandle.init(options.trajectoryFile.c_str(),options.trajectoryMode,options.delayTime,options.trajectoryChunkSize);
			
			info::begin() << "Logging trajectory information in file: \"" << options.trajectoryFile << "\"";
			
			info::out() << " (";

			if (options.trajectoryMode == File_Io::ASCII)
				info::out() << "ascii mode";
			else 
				info::out() << "binary mode";

			if (options.trajectoryChunkSize > 0)
				info::out() << ", chunk-size = " << options.trajectoryChunkSize;
			else
				info::out() << "no chunks";

			info::out() << ")" << std::endl;
		}

	// *** setup recording of grid data

		File_Io::LogGrid gridHandle;

		if (!options.gridFile.empty())
		{
			gridHandle.init(options.gridFile.c_str(),&system->geomTable,&system->configTable,options.gridMode,options.delayTime);

			info::begin() << "Logging grid data in file: \"" << options.gridFile << "\"";
			
			info::out() << " (";

			if (options.gridMode == File_Io::ASCII)
				info::out() << "acii mode";
			else
				info::out() << "binary mode";
			
			info::out() << ")" << std::endl;
		}

	// *** setup recording of surface data

		File_Io::LogSurface surfaceHandle;

		if (!options.surfaceFile.empty())
		{
			surfaceHandle.init(options.surfaceFile.c_str(),system,options.surfaceMode,options.delayTime);

			info::begin() << "Logging surface information in file: \"" << options.surfaceFile << "\"";
			
			info::out() << " (";

			if (options.surfaceMode == File_Io::ASCII)
				info::out() << "acii mode";
			else
				info::out() << "binary mode";
			
			info::out() << ")" << std::endl;
		}

	// *** probability function

		info::begin() << "Method used to compute evaporation probabilities: ";

		switch (options.probMode)
		{
			case Surface_3d::PROB_LINEAR_FIELD:
				info::out() << "\"linear field\"";
				break;
			case Surface_3d::PROB_BOLTZMANN:
				info::out() << "\"boltzmann\"";
				break;
			case Surface_3d::PROB_LINEAR_FORCE:
				info::out() << "\"linear force\"";
				break;
			case Surface_3d::PROB_VORONOI_FLUX_FORCE:
				info::out() << "\"voronoi flux and force\"";
				break;
			default:
				throw std::runtime_error("evaporation(): Unknown method for computing evaporation probabilies!");
		}
		
		info::out() << std::endl;

	// *** evaporation condition

		info::begin() << "Method used to determine next evaporation event: ";

		switch (options.evapMode)
		{
			case Surface_3d::EVAP_MAXIMUM:
				info::out() << "\"maximum probability\"";
				break;
			case Surface_3d::EVAP_MONTE_CARLO:
				info::out() << "\"monte carlo\"";
				break;
			case Surface_3d::EVAP_ME:
				info::out() << "\"my method\"";
				break;
			case Surface_3d::EVAP_ME2:
				info::out() << "\"my method 2\"";
				break;
			default:
				throw std::runtime_error("evaporation(): Unknown method for determining next evaporation event!");
		}
		
		info::out() << std::endl;


	// *** general parameters used for trajectory computation
		
		info::begin() << "Parameters used for trajectory computation:" << std::endl;
		info::begin() << "\t-> integrator type = " << Trajectory_3d::integrator_str(options.trajectoryIntegrator) << std::endl;
		info::begin() << "\t-> stepper type = " << Trajectory_3d::stepper_str(options.trajectoryStepper) << std::endl;
		info::begin() << "\t-> initial time step = " << options.trajectoryTimeStep << " s" << std::endl;
		info::begin() << "\t-> time step limit = " << options.trajectoryTimeStepLimit << " s" << std::endl;
		info::begin() << "\t-> error threshold = " << options.trajectoryErrorThreshold << std::endl;
		
	// *** some other settings which affect the trajectories
		
		if (options.fixedInitialPosition)
			info::begin() << "Initial position of atoms is fixed (= lattice position)." << std::endl;
		else
			info::begin() << "Initial position of atoms is NOT fixed (not implemented?)." << std::endl;
		
		if (options.noInitialVelocity)
			info::begin() << "Initial velocity of atoms is zero." << std::endl;
		else
			info::begin() << "Initial velocity of atoms depends on temperature and direction is random." << std::endl;

		info::begin() << "Temperature value used for calculations: " << system->configTable.temperature() << " K." << std::endl;

	// *** look for surface sites and do initialization of the surface table

		Surface_3d::Table surfaceTable;
		
		surfaceTable.setVacuumId(system->configTable[options.vacuumName.c_str()].id());

		info::begin() << "Vacuum cells are defined by cell-type named ";
		info::out() << "\"" << system->configTable[surfaceTable.vacuumId()].name() << "\"";
		info::out() << std::endl;
		
		surfaceTable.init(*system);

		info::begin() << "Initial number of surface sites: " << surfaceTable.nodes().size() << std::endl;

		Surface_3d::evap_compute_specificFields(&surfaceTable,*system); // initializes the scaling reference value;
		Surface_3d::evap_compute_probabilities(options.probMode,&surfaceTable,*system);

		info::begin() << "Reference value for voltage/time rescaling: " << surfaceTable.scalingReference() << std::endl; 
		
	// *** look for surface site with maximum z-position

		float initApexZ = system->geomTable.nodeCoords(surfaceTable.apex(*system)->index()).z();
		
		info::begin() << "Initial apex position is at z = " << initApexZ;

		if (options.initShrinkage != 0.0f) info::out() << " (memorized shrinkage = " << options.initShrinkage << ")";
		info::out() << std::endl;

	// *** backup / output of the initial state
		
		if (!options.geometryFile.empty())
		{
			info::begin() << "Writing geometry information to file \"" << options.geometryFile << "\"" << std::endl;
			File_Io::writeGeometry(options.geometryFile.c_str(),system->geomTable,options.geometryParams,options.geometryMode);
		}

		if (!options.dumpFile.empty()) File_Io::writeSystem(options.dumpFile.c_str(),*system,options.initEventCnt);

		if (!options.gridFile.empty()) gridHandle.push(options.initEventCnt,system->gridTable);

		if (!options.surfaceFile.empty()) surfaceHandle.push(options.initEventCnt,surfaceTable);

	// *** start evaporation sequence

	info::begin() << "Beginning evaporation sequence." << std::endl;
	
	std::queue<float> voltageQueue;
	float voltageQueueSum(0.0f);

	unsigned int eventCnt(options.initEventCnt);

	time_t startTime = std::time(0);
	// std::ofstream outflie("1.txt");

	int iter0=0;
	while (surfaceTable.nodes().size() > 0)
	{
		// std::cout<<iter0<<"\n";
		// if (iter0++>9999) break;
		#ifdef VERBOSE
		clock_t cycleClock = std::clock();
		#endif

		#ifdef VERBOSE
		info::open("cycle");
		#endif

		const float scalingReference = surfaceTable.scalingReference();

		File_Io::LogResults::Dataset resultsData;

		// ***

		resultsData.eventIndex = ++eventCnt;
 
		{
			resultsData.simTime = std::difftime(std::time(0),startTime);
			resultsData.simTime /= 60.0f;

			#ifdef VERBOSE
			info::begin() << "Time is " << resultsData.simTime;
			info::out() << " min (" << (resultsData.eventIndex - options.initEventCnt)/resultsData.simTime << " atoms/min)";
			info::out() << std::endl;
			#else
			if (eventCnt % options.dumpInterval == 0)
			{
				char eventString[100];
				std::sprintf(eventString,"#%08u",eventCnt);
				
				info::begin() << eventString << " Time is " << resultsData.simTime;
				info::out() << " min (" << (resultsData.eventIndex - options.initEventCnt)/resultsData.simTime << " atoms/min)";
				info::out() << ", apex position is at " << system->geomTable.nodeCoords(surfaceTable.apex(*system)->index());
				info::out() << std::endl;
			}
			#endif
		}

		//////////////////////////////////////////////////////
		//                                                  //
		//                     change                       //
		//                                                  //
		//////////////////////////////////////////////////////

		/*
		std::vector<st_me> v_me;

		{
			st_me tmp;
			for (int i = 0; i < system->geomTable.size_me(); i++) {
				tmp.num = i;
				tmp.z=
			}

		}
		*/

		// I) find atom for evaporation

		// std::cout<<"outer\n";

		bool skipb=0;
		std::vector<Surface_3d::Nodeset::const_iterator> evapSet;
		{
			#ifdef VERBOSE
			clock_t startClock = std::clock();
			#endif

			// *** choose atom

			//std::cout << system->geomTable.size_me() << "\n";
			//std::cout << system->gridTable.size_me() << "\n";

			//double randd=std::rand()/double(RAND_MAX);

			int mode;
			//if (randd>0.8) mode=3; else mode=2;

			////////////////////////////////
			//////////   change  ///////////
			////////////////////////////////
			mode=0;

			
			const Surface_3d::Nodeset::const_iterator node = Surface_3d::evap_findCandidate(mode,surfaceTable, system->geomTable,skipb,evapSet);
			//if (skipb==1) std::cout<<"here!\n"; else std::cout<<"0 ";

			// *** collect specific data

			resultsData.nodeIndex = node->index();
			
			resultsData.nodeId = system->gridTable.id(resultsData.nodeIndex);
			resultsData.nodeNumber= system->gridTable.number(resultsData.nodeIndex);

			resultsData.probability = node->probability();
			
			resultsData.potentialBefore = system->gridTable.potential(resultsData.nodeIndex);
			resultsData.fieldBefore = system->gridTable.field_o1(resultsData.nodeIndex,system->geomTable);

			resultsData.apex = system->geomTable.nodeCoords(surfaceTable.apex(*system)->index());
			resultsData.normal = surfaceTable.normal(resultsData.nodeIndex,*system);

			// ***

			#ifdef VERBOSE
			info::begin() << "Evaporating atom at node #" << resultsData.nodeIndex;
			info::out() << " with field strength " << resultsData.fieldBefore.length() << " V/m" << std::endl;
			info::begin() << "Node type is \"" << system->configTable[resultsData.nodeId].name() << "\"" << std::endl;
			info::begin() << "Unique number is " << resultsData.nodeNumber.toValue() << std::endl;
 
			info::begin() << "Related probability: " << 100.0f * node->probability() << "%" << std::endl;
 
			info::begin() << "*** Clocks (select atom for evaporation): ";
			info::out() << (std::clock() - startClock) / static_cast<float>(CLOCKS_PER_SEC)  * 1.0e3f;
			info::out() << " ms" << std::endl;
			#endif
		}

		// II) let atom evaporate

		{
			#ifdef VERBOSE
			clock_t startClock = std::clock();
			#endif

 			evap_takeAway(system,&surfaceTable,resultsData.nodeIndex);

			#ifdef VERBOSE
			info::begin() << "Selected atom is removed from grid (#" << resultsData.eventIndex << ")." << std::endl;
			info::begin() << "New number of surface-cells: " << surfaceTable.nodes().size() << std::endl;

			info::begin() << "*** Clocks (evaporation): ";
			info::out() << (std::clock() - startClock) / static_cast<float>(CLOCKS_PER_SEC)  * 1.0e3f;
			info::out() << " ms" << std::endl;
			#endif
			
			#ifdef DEMO_MESH_MODE
			const Geometry_3d::Point position = system->geomTable.nodeCoords(resultsData.node);
			const Configuration::NodeId neumannId = system->configTable["Neumann"].id();
			
			for (int i = 0; i < system->geomTable.numNodes(); i++)
			{
				if (i == resultsData.node) continue;
				if (system->geomTable.nodeCoords(i).x() == position.x() && system->geomTable.nodeCoords(i).z() == position.z())
				{
					system->gridTable.node(i).setId(neumannId);
					system->gridTable.fastResync(i,system->geomTable,system->configTable);
				}
			}
			#endif
		}

		// III) relaxation of potentials

		bool skip=skipb;
		if (skip==1) {
			//double pickd=std::rand()/double(RAND_MAX);
			//double threshold=0.8;
			//if (pickd>threshold) skip=0;


			skip=0;
			//暂时不跳过

		}
		if (!skip)
		{
			#ifdef VERBOSE
			clock_t startClock = std::clock();
			#endif

			try
			{
				#ifdef VERBOSE
				const unsigned long stepCnt = system->gridTable.localRelax
					(
						resultsData.nodeIndex,
						options.relaxShellSize,
						options.relaxThreshold,
						options.relaxCycleSize,
						options.relaxQueueSize
					);
				info::begin() << "Relaxing potential locally => " <<  stepCnt << " steps." << std::endl;
				#else
				system->gridTable.localRelax
					(
						resultsData.nodeIndex,
						options.relaxShellSize,
						options.relaxThreshold,
						options.relaxCycleSize,
						options.relaxQueueSize
					);
				#endif
			}
			catch (Grid_3d::Table::RelaxationFault& error)
			{
				#ifdef VERBOSE
				info::begin() << "Relaxing potential locally => limit reached: " << error.iteration() << " steps ";
				info::out() << "(deviation = " << error.deviation() << ", slope = " << error.slope() << ")." << std::endl;
				#endif
			}

			try
			{
				#ifdef VERBOSE
				double deviation = system->gridTable.relax(options.relaxGlobalCycles);
				info::begin() << "Relaxing potential => deviation = " <<  deviation << "." << std::endl;
				#else
				system->gridTable.relax(options.relaxGlobalCycles);
				#endif
			}
			catch (Grid_3d::Table::RelaxationFault& error)
			{
				#ifdef VERBOSE
				info::begin() << "Relaxing potential => limit reached: " << error.iteration() << " steps ";
				info::out() << "(deviation = " << error.deviation() << ", slope = " << error.slope() << ")." << std::endl;
				#endif
			}
			
			if (options.refreshInterval > 0 && (options.initEventCnt + resultsData.eventIndex) % options.refreshInterval == 0)
			{
				try
				{
					#ifdef VERBOSE
					signed long stepCnt = system->gridTable.relax(options.refreshThreshold,options.refreshCycleSize,options.refreshQueueSize);
					info::begin() << "Relaxing potential (refresh) => iteration steps: " << stepCnt << std::endl;
					#else
					system->gridTable.relax(options.refreshThreshold,options.refreshCycleSize,options.refreshQueueSize);
					#endif
				}
				catch (Grid_3d::Table::RelaxationFault& error)
				{
					#ifdef VERBOSE
					info::begin() << "Relaxing potential (refresh) => numerical limit reached: " << error.iteration() << " steps ";
					info::out() << "(deviation = " << error.deviation() << ", slope = " << error.slope() << ")" << std::endl;
					#endif
				}
			}

			#ifdef VERBOSE
			info::begin() << "*** Clocks (relaxation): ";
			info::out() << (std::clock() - startClock) / static_cast<float>(CLOCKS_PER_SEC)  * 1.0e3f;
			info::out() << " ms" << std::endl;
			#endif
		}

		// IV) update evaporation probabilities and surface data 

		{
			#ifdef VERBOSE
			clock_t startClock = std::clock();
			#endif

			Surface_3d::evap_compute_specificFields(&surfaceTable,*system); // updates the scaling reference value
			Surface_3d::evap_compute_probabilities(options.probMode,&surfaceTable,*system);
			
			#ifdef VERBOSE
			info::begin() << "*** Clocks (compute/update evaporation-probabilities): ";
			info::out() << (std::clock() - startClock) / static_cast<float>(CLOCKS_PER_SEC)  * 1.0e3;
			info::out() << " ms" << std::endl;
			#endif
		}

		// V) computation of trajectory

		{
			#ifdef VERBOSE
			clock_t startClock = std::clock();
			#endif

			// ***

			float charge = system->configTable[resultsData.nodeId].evapCharge();	// [charge] = 1
			charge *= Trajectory_3d::eCharge;					// [charge] = C
		
			float mass = system->configTable[resultsData.nodeId].mass();		// [mass] = amu
			mass *= Trajectory_3d::amu2kg;						// [mass] = kg

			Geometry_3d::Point position;
			Geometry_3d::Point position_o;

			////////////////    change////////////////
			//////

			
			// position_o=system->geomTable.nodeCoords(resultsData.nodeIndex);
			// if (true)
			// std::cout<<"here\n";
			if (false)
			// if (options.fixedInitialPosition)
				position = system->geomTable.nodeCoords(resultsData.nodeIndex);
			else
				// std::cout<<"yes\n";
				position = evap_initialPosition(resultsData.nodeIndex,system->geomTable,system->gridTable,resultsData.normal[0],resultsData.normal[1],resultsData.normal[2]);

			MathVector3d<float> velocity;
			if (options.noInitialVelocity)
				velocity = MathVector3d<float>(0.0f);
			else
				velocity = evap_initialVelocity(system->gridTable.node(resultsData.nodeIndex).id(),system->configTable);

			// ***
			// outflie<<position_o[0]<<" "<<position_o[1]<<" "<<position_o[1]<<std::endl;


		
			resultsData.trajectory.setGeometry(&system->geomTable);
			resultsData.trajectory.setGrid(&system->gridTable);
		
			resultsData.trajectory.setIntegratorType(options.trajectoryIntegrator);
			resultsData.trajectory.setStepperType(options.trajectoryStepper);
			resultsData.trajectory.setTimeStepLimit(options.trajectoryTimeStepLimit);

			const int tetGuessIndex = system->geomTable.node(resultsData.nodeIndex).associatedTetrahedron;
			
			resultsData.trajectory.init(position,velocity,charge,mass,tetGuessIndex);
			resultsData.trajectory.integrate(options.trajectoryTimeStep,options.trajectoryErrorThreshold);
			resultsData.trajectory.extrapolate(Geometry_3d::Point::Z,system->geomTable.max().z());

			// ***

			#ifdef VERBOSE
			info::open("trajectory");
		
			info::begin() << "mass = " << resultsData.trajectory.mass() << " kg" << std::endl;
			info::begin() << "charge = " << resultsData.trajectory.charge() << " C" << std::endl;
		
			info::begin() << "initial position = " << resultsData.trajectory.data().front().position() << " m" << std::endl;
			info::begin() << "final position = " << resultsData.trajectory.data().back().position() << " m" << std::endl;
		
			info::begin() << "error estimation =  " << resultsData.trajectory.error_estimate().position() << std::endl;
		
			info::close("trajectory");
		
			info::begin() << "*** Clocks (trajectory): ";
			info::out() << (std::clock() - startClock) / static_cast<float>(CLOCKS_PER_SEC)  * 1.0e3f;
			info::out() << " ms" << std::endl;
			#endif
		}

		// VI) compute voltage and time-scale values

		{
			resultsData.potentialAfter = system->gridTable.potential(resultsData.nodeIndex);
			resultsData.fieldAfter = system->gridTable.field_o1(resultsData.nodeIndex,system->geomTable);

			// ***

			std::vector<Trajectory_3d::phaseVector>::const_reverse_iterator backIndex = resultsData.trajectory.data().rbegin();
			while (resultsData.trajectory.data().rend() != backIndex && backIndex->tetIndex() < 0) backIndex++;

			if (resultsData.trajectory.data().rend() != backIndex && resultsData.fieldBefore.length() != 0.0f) 
			{
				// *** compute measurement voltage

				float tmpVoltage = resultsData.potentialBefore;
				tmpVoltage -= Grid_3d::potential(system->geomTable,system->gridTable,backIndex->position(),backIndex->tetIndex());
				tmpVoltage *= system->configTable[resultsData.nodeId].evapField();
				tmpVoltage /= resultsData.fieldBefore.length();

				tmpVoltage /= scalingReference; // !!! ensure independent scaling !!!

				voltageQueue.push(tmpVoltage);
				voltageQueueSum += tmpVoltage;
					
				while (voltageQueue.size() > options.voltageQueueSize)
				{
					voltageQueueSum -= voltageQueue.front();
					voltageQueue.pop();
				}
					
				resultsData.voltage = voltageQueueSum;
				resultsData.voltage /= voltageQueue.size();

				// *** compute time-of-flight scaling factor
				
				resultsData.timeScale = resultsData.potentialAfter;
				resultsData.timeScale -= Grid_3d::potential(system->geomTable,system->gridTable,backIndex->position(),backIndex->tetIndex());
				resultsData.timeScale /= resultsData.voltage;
				resultsData.timeScale = std::sqrt(resultsData.timeScale);
			}
			else
			{
				if (resultsData.fieldBefore.length() == 0.0f)
				{
					resultsData.voltage = 1.0f;
					resultsData.timeScale = 1.0f;
				}
				else
				{
					resultsData.voltage = 0.0f;
					resultsData.timeScale = 1.0f;
				}
			}
		}

		// VII) save cycle data in file

		{
			#ifdef VERBOSE
			clock_t startClock = std::clock();
			#endif

			if (!options.resultsFile.empty()) 
				resultsHandle.push(resultsData);
			if (!options.trajectoryFile.empty()) 
				trajectoryHandle.push(resultsData.eventIndex,resultsData.nodeNumber.toValue(),resultsData.timeScale,resultsData.trajectory);
			
			#ifdef VERBOSE
			info::begin() << "*** Clocks (file output): ";
			info::out() << (std::clock() - startClock) / static_cast<float>(CLOCKS_PER_SEC)  * 1.0e3f;
			info::out() << " ms" << std::endl;
			#endif
		}

		// VIII) backup / occasional output of the current state

		{
			#ifdef VERBOSE
			clock_t startClock = std::clock();
			#endif
			
			if ((options.initEventCnt + resultsData.eventIndex) % options.dumpInterval == 0 && !options.dumpFile.empty())
				File_Io::writeSystem(options.dumpFile.c_str(),*system,resultsData.eventIndex);

			if ((options.initEventCnt + resultsData.eventIndex) % options.gridInterval == 0 && !options.gridFile.empty())
				gridHandle.push(resultsData.eventIndex,system->gridTable);

			if ((options.initEventCnt + resultsData.eventIndex) % options.surfaceInterval == 0 && !options.surfaceFile.empty())
				surfaceHandle.push(resultsData.eventIndex,surfaceTable);
			
			#ifdef VERBOSE
			info::begin() << "*** Clocks (backup/surface): ";
			info::out() << (std::clock() - startClock) / static_cast<float>(CLOCKS_PER_SEC)  * 1.0e3f;
			info::out() << " ms" << std::endl;
			#endif

			#ifdef DEMO_MESH_MODE

			char filename[150];

			std::sprintf(filename,"%s.%08u.csv","gridStatus",eventCnt);
			//Debug::writeData(filename,*system,',','\n');
			
			std::sprintf(filename,"%s.%08u.vtk","surfaceStatus",eventCnt);
			//Debug::writeData_PARAVIEW_surfaceCells(filename,surfaceTable,*system);

			std::sprintf(filename,"%s.%08u.vtk","potentialStatus",eventCnt);
			//Debug::writeData_PARAVIEW_potentials(filename,*system);

			std::sprintf(filename,"%s.%08u","surfaceTrajectories",eventCnt);
			//File_Io::write_surfaceTrajectories(filename,surfaceTable,*system,1e-6f,File_Io::ASCII);
			
			#endif
		}

		#ifdef VERBOSE
		info::close("cycle");

		info::begin() << "+++ Clocks (cycle): ";
		info::out() << (std::clock() - cycleClock) / static_cast<float>(CLOCKS_PER_SEC)  * 1.0e3f;
		info::out() << " ms" << std::endl;
		#endif

		if (eventCnt - options.initEventCnt >= options.eventCntLimit) break;
		if (initApexZ - resultsData.apex.z() >= options.shrinkLimit - options.initShrinkage) break;
	}

	// *** backup / output of the final state

	if (!options.dumpFile.empty())
	{
		char filename[150];
		std::sprintf(filename,"%s.%08u",options.dumpFile.c_str(),eventCnt);

		File_Io::writeSystem(filename,*system,eventCnt);
	}

	if (surfaceTable.nodes().size() > 0 && !options.surfaceFile.empty()) surfaceHandle.push(eventCnt,surfaceTable);

	info::close("evaporation-sequence");
}
