#ifndef TAPSIM_SURFACE_3D_H
#define TAPSIM_SURFACE_3D_H

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

#include <set>

#include "system_3d.h"

namespace Surface_3d
{
	enum ProbModes { PROB_LINEAR_FIELD, PROB_BOLTZMANN, PROB_LINEAR_FORCE, PROB_VORONOI_FLUX_FORCE } ;
	enum EvapModes { EVAP_MAXIMUM, EVAP_MONTE_CARLO, EVAP_ME, EVAP_ME2
	};
	
	class Node
	{
		public:
			struct Compare
			{
				bool operator()(const Node& lhs, const Node& rhs) const 
					{ return lhs.index() < rhs.index(); }
			};
			
			// ***
			
			Node(const int indexValue =-1, const float probabilityValue =0.0f)
				: _index(indexValue),
				  _probability(probabilityValue)
			{}

			void setIndex(const int value) { _index = value; }
			int index() const { return _index; }

			void setProbability(const float value) { _probability = value; }
			float probability() const { return _probability; }

		private:
			int _index;
			float _probability;
	};

	typedef std::set<Node,Node::Compare> Nodeset;

	class Table 
	{
		public:
			Table();
			Table(const System_3d&, const Configuration::NodeId);

			void init(const System_3d&);
			void update(const int, const System_3d&);

			Geometry_3d::Point normal(const int, const System_3d&) const;

			const Nodeset& nodes() const { return _nodes; }

			Nodeset::const_iterator apex(const System_3d&) const;

			void setVacuumId(const Configuration::NodeId value) { _vacuumId = value; }
			Configuration::NodeId vacuumId() const { return _vacuumId; }
			
			void setScalingReference(const float value) { _scalingReference = value; }
			float scalingReference() const { return _scalingReference; }
			bool getSkip() const {return _skip;}
			void setSkip(const bool b)  {_skip=b;}

		private:
			Nodeset _nodes;
			Configuration::NodeId _vacuumId;
			
			float _scalingReference;
			bool _skip;
	};

	void evap_compute_specificFields(Table*, const System_3d&);
	void evap_compute_probabilities(const int, Table*, const System_3d&);

	Nodeset::const_iterator evap_findCandidate(const int, Surface_3d::Table const&, const Geometry_3d::Table&, bool&, std::vector<Surface_3d::Nodeset::const_iterator>&);
}

// ***

MathVector3d<float> evap_initialPosition(const int, const Geometry_3d::Table&,const Grid_3d::Table&,float ,float ,float );
MathVector3d<float> evap_initialVelocity(const Configuration::NodeId&, const Configuration::Table&);

// ***

void evap_takeAway(System_3d* system, Surface_3d::Table* surfaceTable, const int nodeIndex);

#endif
