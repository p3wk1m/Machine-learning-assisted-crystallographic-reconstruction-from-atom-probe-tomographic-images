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

#include "surface_3d.h"
#define PI 3.1415926535898
#define DEC (PI/180)


#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <set>
#include <cmath>
#include <vector>

#include <algorithm>

#include "info.h" /* added for the status informatin in evap_takeAway() */
#include <list>

#include "utils.h"


/* ********** ----------- ********** */

Surface_3d::Table::Table()
	: _nodes(),
	  _vacuumId(-1),
	  _scalingReference(1.0f),
	  _skip(0)
{}

Surface_3d::Table::Table(const System_3d& system, const Configuration::NodeId vacuumId)
	: _nodes(),
	  _vacuumId(vacuumId),
	  _scalingReference(1.0f),
	  _skip(0)
{
	init(system);
}

void Surface_3d::Table::init(const System_3d& system)
{
	// *** local switch, not included in the executable's general parameter list
	// *** => determines distinction between removable and non removable surface nodes

	const bool acceptRemovable(true);

	// ***

	_nodes.clear();

	for (int i = 0; i < system.gridTable.numNodes(); i++)
	{
		const Configuration::NodeId id = system.gridTable.id(i);

		if (id == _vacuumId) continue;

		if (!system.configTable[id].removable() && acceptRemovable) continue;

		for (int j = 0; j < system.gridTable.node(i).numNeighbours(); j++)
		{
			const int neighIndex = system.gridTable.node(i).neighbour(j);

			if (system.gridTable.id(neighIndex) == _vacuumId && system.geomTable.voronoiArea(i,neighIndex,1.0f) > 0.0f)
			{
				const Node obj(i,0.0);
				_nodes.insert(obj);
				break;
			}
		}
	}
}

void Surface_3d::Table::update(const int nodeIndex, const System_3d& system)
{
	// *** local switch, not included in the executable's general parameter list
	// *** => determines distinction between removable and non removable surface nodes

	const bool acceptRemovable(true);

	// ***

	if (_nodes.erase(nodeIndex) != 1) throw std::runtime_error("Surface_3d::Table::update()");

	// ***

	for (int i = 0; i < system.gridTable.node(nodeIndex).numNeighbours(); i++)
	{
		const int neighIndex = system.gridTable.node(nodeIndex).neighbour(i);

		if (system.gridTable.id(neighIndex) == _vacuumId) continue;

		if (!system.configTable[system.gridTable.id(neighIndex)].removable() && acceptRemovable) continue;

		// *** (redundant) test if the neighbour node has at least one vacuum-node in 
		// *** its neighbourhood

		for (int j = 0; j < system.gridTable.node(neighIndex).numNeighbours(); j++)
		{
			const int nnIndex = system.gridTable.node(neighIndex).neighbour(j);

			if (system.gridTable.id(nnIndex) == _vacuumId && system.geomTable.voronoiArea(neighIndex,nnIndex,1.0f) > 0.0f)
			{
				const Node obj(neighIndex,0.0);
				_nodes.insert(obj);
				break;
			}
		}
	}
}

/* ********** ----------- ********** */

Geometry_3d::Point Surface_3d::Table::normal(const int nodeIndex, const System_3d& system) const
{
	if (_nodes.find(nodeIndex) == _nodes.end()) return Geometry_3d::Point(0.0f);
	
	std::set<int> localNodes;
	system.geomTable.adjacentNodes(nodeIndex,&localNodes,3); // limit search to 3rd neighbours

	std::vector<int> vacNodes;
	vacNodes.reserve(localNodes.size());

	{
		std::set<int>::const_iterator i = localNodes.begin();

		while (localNodes.end() != i)
		{
			if (system.gridTable.node(*i).id() == _vacuumId) vacNodes.push_back(*i);
			
			if (_nodes.find(*i) == _nodes.end())
			{
				const std::set<int>::const_iterator j(i);

				advance(i,1);
				localNodes.erase(j);
			}
			else
				advance(i,1);
		}
	}

	// ***

	std::vector<Geometry_3d::Point> coords;
	for (std::set<int>::const_iterator i = localNodes.begin(); i != localNodes.end(); i++)
		coords.push_back(system.geomTable.nodeCoords(nodeIndex) - system.geomTable.nodeCoords(*i));

	Geometry_3d::Point value = Geometry_3d::find_best_plane(coords);

	// *** adjust orientation
	
	Geometry_3d::Point tmp;
	for (std::vector<int>::const_iterator i = vacNodes.begin(); i != vacNodes.end(); i++)
		tmp += system.geomTable.nodeCoords(*i) - system.geomTable.nodeCoords(nodeIndex);
	
	if (value * tmp < 0.0) value *= -1.0f;
	
	return value;
}

Surface_3d::Nodeset::const_iterator Surface_3d::Table::apex(const System_3d& system) const
{
	Surface_3d::Nodeset::const_iterator topNode = _nodes.begin();
	float topCoordinate = system.geomTable.nodeCoords(topNode->index()).z();
	
	// ***
	
	for (Nodeset::const_iterator i = _nodes.begin(); i != _nodes.end(); i++)
	{
		const float coordinate = system.geomTable.nodeCoords(i->index()).z();
		
		if (coordinate > topCoordinate)
		{
			topCoordinate = coordinate;
			topNode = i;
		}
	}
	
	return topNode;
}

/* ********** ----------- ********** */

void Surface_3d::evap_compute_specificFields(Surface_3d::Table* surfaceTable, const System_3d& system)
{
	/* probably this is just a bad hack ... */

	if (surfaceTable == 0 || surfaceTable->nodes().empty()) return;
	
	Surface_3d::Nodeset::iterator i = surfaceTable->nodes().begin();

	float value = system.gridTable.field_o1(i->index(),system.geomTable).length();
	value /= system.configTable[system.gridTable.id(i->index())].evapField();
	advance(i,1);
	
	float maxValue(value);
	
	while (surfaceTable->nodes().end() != i)
	{
		value = system.gridTable.field_o1(i->index(),system.geomTable).length();
		value /= system.configTable[system.gridTable.id(i->index())].evapField();

		if (maxValue < value) maxValue = value;

		advance(i,1);
	}

	surfaceTable->setScalingReference(maxValue);
}

namespace
{
	inline void probBoltzmann(Surface_3d::Table* surfaceTable, const System_3d& system)
	{
		/*
		*** The problem of picking the correct distribution for the field dependant evaporation
		*** probability using the Boltzmann Equation is solved by assuming that the surface atom
		*** exposed to the highest field with respect to its specific evaporation field strength
		*** has a evaporation probability of 1.0. This means the activation barrier vanishes.
		*/

		if (system.configTable.temperature() <= 0.0)
			throw std::runtime_error("probBoltzmann(): Cannot compute probability with chosen temperature value!");
		

		const float kBoltzmann = 8.6173431e-5; // [kBoltzmann] =  eV/K

		float sum(0.0);

		for (Surface_3d::Nodeset::iterator i = surfaceTable->nodes().begin(); i != surfaceTable->nodes().end(); i++)
		{
			bool b = 1;
			float probability(1.0);
			//float probability(-1.0);
			probability *= system.gridTable.field_o1(i->index(),system.geomTable).length();
			if (!b) std::cout <<"field   "<< system.gridTable.field_o1(i->index(), system.geomTable).length() << "\n";

			probability /= system.configTable[system.gridTable.id(i->index())].evapField();
			if (!b) std::cout <<"evapField   "<< system.configTable[system.gridTable.id(i->index())].evapField() << "\n";
			if (!b) std::cout << probability << "\n";

			probability = -	sqrt(probability);
			probability += 1.0;
			if (!b) std::cout << "1-sqrt   " << probability << "\n";

			probability *= system.configTable[system.gridTable.id(i->index())].evapEnergy();
			if (!b) std::cout <<"evapEnergy  "<< system.configTable[system.gridTable.id(i->index())].evapEnergy() << "\n";

			probability /= system.configTable.temperature();
			if (!b) std::cout << "temperature  " << system.configTable.temperature() << "\n";

			probability /= kBoltzmann;
			probability /= -2.0;
			if (!b) std::cout << "added_up  " << probability << "\n";

			probability = std::exp(probability);
			const_cast<Surface_3d::Node&>(*i).setProbability(probability);

			if (!b) std::cout << probability << "\n";
			sum += probability;
		}	
		
		// *** normalize

		for (Surface_3d::Nodeset::iterator i = surfaceTable->nodes().begin(); i != surfaceTable->nodes().end(); i++)
		{
			float value = i->probability();
			value /= sum;
			
			const_cast<Surface_3d::Node&>(*i).setProbability(value);
		}
	}

	//inline void probLinear_field(Surface_3d::Table* surfaceTable, const System_3d& system)
	void probLinear_field(Surface_3d::Table* surfaceTable, const System_3d& system)
	{
		/*
		*** This method regards different evaporation probabilities for species with
		*** heterogeneous evaporation field strengths. This is achieved by rescaling
		*** of the computed field strength with the inverse of the specific evaporation
		*** field strength. The evaporation probability is computed as the ratio of this.
		*/

		float sum(0.0);
		bool b = 1;
		for (Surface_3d::Nodeset::iterator i = surfaceTable->nodes().begin(); i != surfaceTable->nodes().end(); i++)
		{
			
			float value = system.gridTable.field_o1(i->index(),system.geomTable).length();
			if (!b) std::cout << "field   " << value << "\n";
			value /= system.configTable[system.gridTable.id(i->index())].evapField();
			if (!b) std::cout << "evapField   " << system.configTable[system.gridTable.id(i->index())].evapField() << "\n";
			const_cast<Surface_3d::Node&>(*i).setProbability(value);

			sum += i->probability();
		}

		// *** normalize

		for (Surface_3d::Nodeset::iterator i = surfaceTable->nodes().begin(); i != surfaceTable->nodes().end(); i++)
		{
			float value = i->probability();
			value /= sum;

			const_cast<Surface_3d::Node&>(*i).setProbability(value);
		}
	}

	inline void probLinear_force(Surface_3d::Table* surfaceTable, const System_3d& system)
	{
		// computes the probabiliy for field evapoation of each atom by taking into account
		// the acting force on the atom, calculation is started from scratch ignoring any 
		// formerly conducted calculations (e.g. from evap_compute_specificFields() above)

		float sum(0.0);

		for (Surface_3d::Nodeset::iterator i = surfaceTable->nodes().begin(); i != surfaceTable->nodes().end(); i++)
		{
			float fieldstrength, charge;
			fieldstrength = system.gridTable.field_o2(i->index(),system.geomTable,&charge).length();

			float probValue = charge;
			probValue *= fieldstrength;
			probValue /= std::pow(system.configTable[system.gridTable.node(i->index()).id()].evapField(),2.0f);

			const_cast<Surface_3d::Node&>(*i).setProbability(probValue);

			sum += probValue;
		}

		// *** normalize

		for (Surface_3d::Nodeset::iterator i = surfaceTable->nodes().begin(); i != surfaceTable->nodes().end(); i++)
		{
			float value = i->probability();
			value /= sum;

			const_cast<Surface_3d::Node&>(*i).setProbability(value);
		}
	}

	inline void probVoronoiFluxForce(Surface_3d::Table* surfaceTable, const System_3d& system)
	{
		const float epsilon0 = 8.85418781762e-12; // [epsilon0] = As/(Vm)

		float sum(0.0);
		
		for (Surface_3d::Nodeset::iterator i = surfaceTable->nodes().begin(); i != surfaceTable->nodes().end(); i++)
		{
			const Grid_3d::Node& node = system.gridTable.node(i->index());
			
			// *** BEGIN BUGFIX +++ PROPER SCALING
			float surfaceArea(0.0f);
			// *** END BUGFIX +++ PROPER SCALING

			MathVector3d<float> force(0.0f);
			for (int j = 0; j < node.numNeighbours(); j++)
			{
				const int neighIndex = node.neighbour(j);
				
				float value = system.gridTable.potential(neighIndex);
				value -= system.gridTable.potential(i->index());
				value *= value;
				value *= -1.0 * epsilon0;
				value *= node.coupling(j);

				const float epsilon_1 = 1.0 / system.configTable[system.gridTable.node(i->index()).id()].epsilon();
				const float epsilon_2 = 1.0 / system.configTable[system.gridTable.node(neighIndex).id()].epsilon();
				
				value *= epsilon_2 - epsilon_1;

				// *** BEGIN BUGFIX +++ PROPER SCALING
				float corrArea = node.coupling(j);
				corrArea *= system.geomTable.distance(i->index(),neighIndex);
				corrArea /= 2.0;
				corrArea *= epsilon_1 + epsilon_2;
				surfaceArea += corrArea;
				// *** END BUGFIX +++ PROPER SCALING
				
				MathVector3d<float> localForce = system.geomTable.nodeCoords(neighIndex);
				localForce -= system.geomTable.nodeCoords(i->index());
				localForce /= localForce*localForce;
				localForce *= value;
				
				force += localForce;
			}
			
			float probValue = force.length();

			// *** BEGIN BUGFIX +++ PROPER SCALING
			probValue /= surfaceArea;
			// *** END BUGFIX +++ PROPER SCALING

			probValue /= std::pow(system.configTable[system.gridTable.node(i->index()).id()].evapField(),2.0f);
			const_cast<Surface_3d::Node&>(*i).setProbability(probValue);

			sum += probValue;
		}

		// *** normalize

		for (Surface_3d::Nodeset::iterator i = surfaceTable->nodes().begin(); i != surfaceTable->nodes().end(); i++)
		{
			float value = i->probability();
			value /= sum;

			const_cast<Surface_3d::Node&>(*i).setProbability(value);
		}
	}
}

void Surface_3d::evap_compute_probabilities(const int mode, Table* surfaceTable, const System_3d& system)
{
	switch (mode)
	{
		case PROB_BOLTZMANN:
			return probBoltzmann(surfaceTable,system);
		case PROB_LINEAR_FIELD:
			return probLinear_field(surfaceTable,system);
		case PROB_LINEAR_FORCE:
			return probLinear_force(surfaceTable,system);
		case PROB_VORONOI_FLUX_FORCE:
			return probVoronoiFluxForce(surfaceTable,system);
		default:
			throw std::runtime_error("Surface_3d::evap_compute_probabilities()");
	}
}

/* ********** ----------- ********** */

namespace
{
	inline Surface_3d::Nodeset::const_iterator evapMaximum(const Surface_3d::Table& surfaceTable)
	{
		Surface_3d::Nodeset::const_iterator pickNode = surfaceTable.nodes().begin();

		for (Surface_3d::Nodeset::const_iterator i = surfaceTable.nodes().begin(); i != surfaceTable.nodes().end(); i++)
			if (pickNode->probability() < i->probability()) pickNode = i;

		return pickNode;
	}

	inline Surface_3d::Nodeset::const_iterator evapMonteCarlo(const Surface_3d::Table& surfaceTable)
	{
		double pickThreshold = std::rand();
		pickThreshold /= RAND_MAX;

		Surface_3d::Nodeset::const_iterator pickCell = surfaceTable.nodes().begin();

		double threshold = pickCell->probability();

		while (pickThreshold > threshold)
		{
			pickCell++;
			threshold += pickCell->probability();
		}

		return pickCell;
	}
	//inline 
	Surface_3d::Nodeset::const_iterator evap_me(const Surface_3d::Table& surfaceTable, const Geometry_3d::Table& geomTable)
	{
		float high = -999;

		//std::set<int> v_high;

		for (Surface_3d::Nodeset::const_iterator i = surfaceTable.nodes().begin(); i != surfaceTable.nodes().end(); i++)
			if (geomTable.nodeCoords(i->index())[2] > high) {
				//v_high.clear();
				//v_high.insert(i->index());
				high = geomTable.nodeCoords(i->index())[2];
				//pickNode = i;
			}
			/*
			else if (geomTable.nodeCoords(i->index())[2] == high) {
				v_high.insert(i->index());
			}
			*/
		//std::cout << v_high.size()<<"\n";

		Surface_3d::Nodeset::const_iterator pickNode;
		Surface_3d::Nodeset::const_iterator i;

		for (i = surfaceTable.nodes().begin(); i != surfaceTable.nodes().end(); i++) {
			if (geomTable.nodeCoords(i->index())[2]>high-1e-9) {
				pickNode = i;
				break;
			}
		}

		// double pickThreshold = 0.2;
		double pickThreshold = 0.1;

		for (; i != surfaceTable.nodes().end(); i++) {
			double pick = std::rand();
			pick /=double(RAND_MAX);
			if (pick<pickThreshold) continue;
			if (geomTable.nodeCoords(i->index())[2]<high-1e-9)  continue;
			if (pickNode->probability() < i->probability()) pickNode = i;
		}

		return pickNode;
	}

	struct ss{
		double d;
		Surface_3d::Nodeset::const_iterator i;
	};


	int cmp(Surface_3d::Nodeset::const_iterator a1,Surface_3d::Nodeset::const_iterator a2){
		return a1->probability()>a2->probability();
	}

	int cmpva(ss s1,ss s2){
		return s1.d>s2.d;
	}

	int cmp0(double d1,double d2){
		return d1>d2;
	}


/*
	Surface_3d::Nodeset::const_iterator evap_me_2(const Surface_3d::Table& surfaceTable, const Geometry_3d::Table& geomTable,bool& skip,std::vector<Surface_3d::Nodeset::const_iterator>& evapSet)
	{
		float high = -999;
		
		
		std::vector<Surface_3d::Nodeset::const_iterator> v_high;
		std::vector<double> topN_Z;
		std::vector<ss> lowM_XY;
		int topN=3;
		int lowM=5;
		{
			bool bb;
			Surface_3d::Nodeset::const_iterator i = surfaceTable.nodes().begin();
			while (i != surfaceTable.nodes().end()&&topN_Z.size()<topN){
				bb=0;
				for (int ii=0;ii<topN_Z.size();ii++){
					if (topN_Z[ii]==geomTable.nodeCoords(i->index())[2]){
						bb=1;
						break;
					}
				}
				if (!bb) {
					topN_Z.push_back(geomTable.nodeCoords(i->index())[2]);
				}
				i++;
			}
			sort(topN_Z.begin(),topN_Z.end(),cmp0);

			for (;i != surfaceTable.nodes().end();i++){
				if (geomTable.nodeCoords(i->index())[2]>topN_Z.back()){
					topN_Z.pop_back();
					topN_Z.push_back(geomTable.nodeCoords(i->index())[2]);
					sort(topN_Z.begin(),topN_Z.end(),cmp0);
				}
			}
		}


		ss sst;
		for (Surface_3d::Nodeset::const_iterator i = surfaceTable.nodes().begin(); i != surfaceTable.nodes().end(); i++){
			if (find(topN_Z.begin(),topN_Z.end(),geomTable.nodeCoords(i->index())[2]) != topN_Z.end()) {
				v_high.push_back(i);
			} else continue;
			sst.d=geomTable.nodeCoords(i->index())[0]*geomTable.nodeCoords(i->index())[0]+geomTable.nodeCoords(i->index())[1]*geomTable.nodeCoords(i->index())[1];
			sst.i=i;
			lowM_XY.push_back(sst);
			sort(lowM_XY.begin(),lowM_XY.end(),cmpva);
			if (lowM_XY.size()==lowM) lowM_XY.pop_back();
		}

		sort(v_high.begin(),v_high.end(),cmp);

		double sumd=0;

		
		int limit=10;
		if (v_high.size()>limit){
			std::vector<Surface_3d::Nodeset::const_iterator> tmpv;
			for (int i=0;i<limit;i++){
				tmpv.push_back(v_high[i]);
			}
			v_high=tmpv;
		}
		
		bool bb0;
		for (int i=0;i<v_high.size();i++){
			bb0=0;
			for (int j=0;j<lowM_XY.size();j++){
				if (lowM_XY[j].i==v_high[i]) {
					sumd+=v_high[i]->probability()*0.01;
					bb0=1;
					break;
				}
			}
			if (bb0) continue;
			sumd+=v_high[i]->probability();
		}

		int pickNode=0;

		//////////

		double pickThreshold = std::rand();
		pickThreshold /=double(RAND_MAX);

		double threshold = v_high[pickNode]->probability();

		for (int i=1; i <v_high.size(); i++) {
			bb0=0;
			if (pickThreshold < threshold/sumd) break;
			for (int j=0;j<lowM_XY.size();j++){
				if (lowM_XY[j].i==v_high[i]) {
					threshold+=v_high[i]->probability()*0.01;
					bb0=1;
					break;
				}
			}
			if (bb0) continue;
			threshold += v_high[i]->probability();
			pickNode = i;
		}
		
		//////////
		return v_high[pickNode];
		
		
	}
*/

	Surface_3d::Nodeset::const_iterator evap_me_2(const Surface_3d::Table& surfaceTable, const Geometry_3d::Table& geomTable,bool& skip,std::vector<Surface_3d::Nodeset::const_iterator>& evapSet)
	{
		float high = -999;

		std::set<int> v_high;

		for (Surface_3d::Nodeset::const_iterator i = surfaceTable.nodes().begin(); i != surfaceTable.nodes().end(); i++)
			if (geomTable.nodeCoords(i->index())[2] > high) {
				v_high.clear();
				v_high.insert(i->index());
				high = geomTable.nodeCoords(i->index())[2];
				//pickNode = i;
			}
			else if (geomTable.nodeCoords(i->index())[2] == high) {
				v_high.insert(i->index());
			}

		//std::cout << v_high.size()<<"\n";

		Surface_3d::Nodeset::const_iterator pickNode;
		Surface_3d::Nodeset::const_iterator i;

		for (i = surfaceTable.nodes().begin(); i != surfaceTable.nodes().end(); i++) {
			if (v_high.find(i->index()) != v_high.end()) {
				pickNode = i;
				break;
			}
		}

		double pickThreshold = 0.3;

		float x=geomTable.nodeCoords(pickNode->index())[0],y=geomTable.nodeCoords(pickNode->index())[1];
		float tmp=x*x+y*y;

		for (; i != surfaceTable.nodes().end(); i++) {
			if (v_high.find(i->index()) == v_high.end()) continue;
			x=geomTable.nodeCoords(i->index())[0];
			y=geomTable.nodeCoords(i->index())[1];
			if (x*x+y*y>tmp){
				tmp=x*x+y*y;
				pickNode=i;
			}
		}

		return pickNode;
		
		
	}

}

Surface_3d::Nodeset::const_iterator Surface_3d::evap_findCandidate(const int mode, const Table& surfaceTable, const Geometry_3d::Table& geomTable,bool& skip,std::vector<Surface_3d::Nodeset::const_iterator>&evapSet)
{
	switch (mode)
	{
		case EVAP_MAXIMUM:
			return evapMaximum(surfaceTable);
		case EVAP_MONTE_CARLO:
			return evapMonteCarlo(surfaceTable);
		case EVAP_ME:
			return evap_me(surfaceTable,geomTable);
		case EVAP_ME2:
			return evap_me_2(surfaceTable, geomTable, skip, evapSet);
		default:
			throw std::runtime_error("Surface_3d::evap_findCandidate()");
	}
}

/* ********** ----------- ********** */

MathVector3d<float> evap_initialPosition(const int nodeIndex, const Geometry_3d::Table& geomTable,const Grid_3d::Table& gridTable,float nx,float ny,float nz)
{
	MathVector3d<float> tmp =geomTable.node(nodeIndex).coords;
	float r1=rand()/float(RAND_MAX);
	float r2=rand()/float(RAND_MAX);
	float r3=rand()/float(RAND_MAX);
	float s1=rand()/float(RAND_MAX);
	float s2=rand()/float(RAND_MAX);
	float s3=rand()/float(RAND_MAX);


	int atomn=0;
	for (int j = 0; j < gridTable.node(nodeIndex).numNeighbours(); j++)
	{
		const int neighIndex = gridTable.node(nodeIndex).neighbour(j);

		if (gridTable.id(neighIndex) != 0 && geomTable.voronoiArea(nodeIndex,neighIndex,1.0f) > 0.0f) //_vacuumId 0
		{
			atomn++;
		}
	}

	//lm2.5
	float a0=1.0,b0=1.0,c0=1.0;
	float pdx=(a0*1e-9)*(1)*(s1>0.5?1:-1)*(0.346-0.045*atomn);
	float pdy=(a0*1e-9)*(1)*(s2>0.5?1:-1)*(0.346-0.045*atomn);
	//float pdz=(5e-10)*(1+r3)*(s3>0.5?1:-1)*exp(float(3-atomn));
	float pdz=0;

	//lm3.5
	// float a0=1.0,b0=1.0,c0=1.0;
	// float pdx=(a0*1e-9)*(1)*(s1>0.5?1:-1)*(0.46-0.025*atomn);
	// float pdy=(a0*1e-9)*(1)*(s2>0.5?1:-1)*(0.46-0.025*atomn);
	// //float pdz=(5e-10)*(1+r3)*(s3>0.5?1:-1)*exp(float(3-atomn));
	// float pdz=0;


	if (atomn<3) {
		pdx=0;
		pdy=0;
	}
	//std::cout<<nx<<" "<<ny<<" "<<nz<<" "<<atomn<<"\n";
	float u1=nx/2;
	float u2=ny/2;
	float u3=(nz+1)/2;
	float theta=acos(nz);
	float r11=cos(theta)+u1*u1*(1-cos(theta));
	float r12=u1*u2*(1-cos(theta))-u3*sin(theta);
	float r13=u1*u3*(1-cos(theta))+u2*sin(theta);
	float r21=u2*u1*(1-cos(theta))+u3*sin(theta);
	float r22=cos(theta)+u2*u2*(1-cos(theta));
	float r23=u2*u3*(1-cos(theta))-u1*sin(theta);
	float r31=u3*u1*(1-cos(theta))-u2*sin(theta);
	float r32=u3*u2*(1-cos(theta))+u1*sin(theta);
	float r33=cos(theta)+u3*u3*(1-cos(theta));

	float dx=pdx*r11+pdy*r12+pdz*r13;
	float dy=pdx*r21+pdy*r22+pdz*r23;
	float dz=pdx*r31+pdy*r32+pdz*r33;


	tmp[0]+=dx;
	tmp[1]+=dy;
	tmp[2]+=dz;

	return tmp;
}

MathVector3d<float> evap_initialVelocity(const Configuration::NodeId& id, const Configuration::Table& configTable)
{
	const double kBoltzmann = 1.3806504e-23;		// [kBoltzmann] = J/K
	const double amu2kg = 1.660538782e-27;			// [amu2kg] = kg/amu

	double max(2.0);
	max *= kBoltzmann;					// [max] = J/K
	max *= configTable.temperature();			// [max] = J
	max /= configTable[id].mass();				// [max] = kg * m² / s² / amu
	max /= amu2kg;						// [max] = m² / s²

	MathVector3d<float> velocity;				// [velocity] = 1

	velocity.x() = std::rand();
	velocity.x() /= RAND_MAX;
	velocity.x() *= max;					// [velocity.x()] = m²/s²

	velocity.y() = std::rand();
	velocity.y() /= RAND_MAX;
	velocity.y() *= max - velocity.x();			// [velocity.y()] = m²/s²

	velocity.z() = max;
	velocity.z() -= velocity.x();
	velocity.z() -= velocity.y();				// [velocity.z()] = m²/s²

	for (short i = MathVector3d<float>::X; i <= MathVector3d<float>::Z; i++)
	{
		velocity[i] = std::sqrt(velocity[i]);
		if (std::rand() % 2) velocity[i] *= -1.0f;
	}
	return velocity;
}

/* ********** ----------- ********** */

void evap_takeAway(System_3d* system, Surface_3d::Table* surfaceTable, const int nodeIndex)
{
	static int lonelyCnt(0);

	// remove atom from surface

	system->gridTable.node(nodeIndex).setId(surfaceTable->vacuumId());
	system->gridTable.fastResync(nodeIndex,system->geomTable,system->configTable);

	surfaceTable->update(nodeIndex,*system);

	// check for detached surface sites in the vicinity of the former atom location

	{
		std::list<int> lonelySites;

		for (int i = 0; i < system->gridTable.node(nodeIndex).numNeighbours(); i++)
		{
			const int neighIndex = system->gridTable.node(nodeIndex).neighbour(i);
			if (system->gridTable.node(neighIndex).id() == surfaceTable->vacuumId()) continue;

			bool flag(true);
			for (int j = 0; j < system->gridTable.node(neighIndex).numNeighbours(); j++)
			{
				const int index = system->gridTable.node(neighIndex).neighbour(j);
				if (system->gridTable.node(index).id() != surfaceTable->vacuumId())
				{
					flag = false;
					break;
				}
			}
			
			if (flag) lonelySites.push_back(neighIndex);
		}

		if (lonelySites.size() != 0)
		{
			lonelyCnt += lonelySites.size();

			info::out() << "*** Detected detached surface atoms:" << std::endl;
			info::out() << "\t-> node = #" << nodeIndex << std::endl;
			info::out() << "\t-> coordinates = " << system->geomTable.nodeCoords(nodeIndex) << std::endl;
			info::out() << "\t-> type = '" << system->configTable[system->gridTable.id(nodeIndex)].name() << "'" << std::endl;
			info::out() << "\t-> actual number of detached atoms = " << lonelySites.size() << std::endl;
			info::out() << "\t-> total number of detached atoms so far: " << lonelyCnt << std::endl;

			// remove detached surface atoms
			
			for (std::list<int>::const_iterator i = lonelySites.begin(); i != lonelySites.end(); i++)
			{
				system->gridTable.node(*i).setId(surfaceTable->vacuumId());
				system->gridTable.fastResync(*i,system->geomTable,system->configTable);

				surfaceTable->update(*i,*system);
			}
		}
	}
}
