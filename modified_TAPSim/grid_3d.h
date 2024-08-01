#ifndef TAPSIM_GRID_3D_H
#define TAPSIM_GRID_3D_H

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

#include <string>
#include <vector>
#include <limits>
#include <string>
#include <map>

#include "configuration.h"
#include "geometry_3d.h"

#include "../vector/vector.h"

#define DOUBLE_NAN std::numeric_limits<double>::quiet_NaN()

namespace Grid_3d
{
	extern "C" void* cwrapper_threadedRelax(void*);
	extern "C" void* cwrapper_threadedLocalRelax(void*);

	class Node
	{
		public:
			Node();
			~Node();

			// ***

			void setId(const Configuration::NodeId value) { _id = value; }
			Configuration::NodeId id() const { return _id; }

			void setNumber(const Configuration::NodeNumber value) { _number = value; }
			Configuration::NodeNumber number() const { return _number; }

			void setPhi(const int index, const float value) { _phi[index] = value; }
			const float& phi(const int index) const { return _phi[index]; }

			void setCharge(const float value) { _charge = value; }
			float charge() const { return _charge; }

			void setBoundary(const bool);
			bool boundary() const { return (_properties & BOUNDARY); }

			void setDirichletBoundary(const bool);
			bool dirichletBoundary() const { return (_properties & DIRICHLET); }

			void setNeumannBoundary(const bool);
			bool neumannBoundary() const { return  (_properties & NEUMANN); }

			// ***

			void setNumNeighbours(const int);
			int numNeighbours() const { return _numNeighbours; }

			void setNeighbour(const int index, const int value) { _neighbours[index] = value; }
			int neighbour(const int index) const { return _neighbours[index]; }

			void setCoupling(const int index, const float value) { _couplingValues[index] = value; }
			float coupling(const int index) const { return _couplingValues[index]; }

		private:
			enum { BOUNDARY =0x01, DIRICHLET =0x02, NEUMANN =0x04 };

			Configuration::NodeId _id;
			Configuration::NodeNumber _number;

			float _phi[2];
			float _charge;

			unsigned char _properties;

			int _numNeighbours;

			int* _neighbours;
			float* _couplingValues;
	};

	class Table
	{
		public:
			class RelaxationFault
			{
				public:
					RelaxationFault(const char* a, const signed long b=-1, const double c =DOUBLE_NAN, const double d =DOUBLE_NAN)
						: _what(a),
						  _iteration(b),
						  _deviation(c),
						  _slope(d)
					{}

					const char* what() const { return _what.c_str(); }

					signed long iteration() const { return _iteration; }

					double deviation() const { return _deviation; }
					double slope() const { return _slope; }

				private:
					RelaxationFault();

					std::string _what;

					signed long _iteration;

					double _deviation;
					double _slope;
			};

			// ***

			Table(const unsigned int =0);
			~Table();

			void operator<<(std::ifstream&);
			void operator>>(std::ofstream&) const;

			// ***
			
			void setThreadNum(const unsigned int =0);
			unsigned int threadNum() const { return _threadNum; }
			
			// ***

			void allocate(const int size) { _nodes.allocate(size); }

			// ***

			int numNodes() const { return _nodes.size(); }

			const Node& node(const int index) const { return _nodes[index]; }
			Node& node(const int index) { return _nodes[index]; }

			// ***

			void sync(const Geometry_3d::Table&, const Configuration::Table&);
			void resync(const int, const Geometry_3d::Table&, const Configuration::Table&);

			void fastSync(const Geometry_3d::Table&, const Configuration::Table&);
			void fastResync(const int, const Geometry_3d::Table&, const Configuration::Table&);

			// ***

			double relax(const unsigned int);
			signed long relax(const double, const unsigned int, const unsigned int =0);

			double localRelax(const int, const unsigned int, const unsigned int);
			signed long localRelax(const int, const unsigned int, const double, const unsigned int, const unsigned int =0);

			// ***

			void reset(const Configuration::Table&);
			void reset(const int, const Configuration::Table&);

			void randomReset(const Configuration::Table&);
			void randomReset(const int, const Configuration::Table&);

			// ***

			Configuration::NodeId id(const int index) const { return _nodes[index].id(); }
			Configuration::NodeNumber number(const int index) const { return _nodes[index].number(); }

			float potential(const int index) const { return _nodes[index].phi((_phiSwitch+1)%2); } // [potential()] = V

			float flux(const int, const Geometry_3d::Table&) const; // [flux()] = Vm

			MathVector3d<float> field_o1(const int index, const Geometry_3d::Table&) const; // [field_o1()] = V/m
			MathVector3d<float> field_o2(const int index, const Geometry_3d::Table&, float* charge =0) const; // [field_o2()] = V/m

			MathVector3d<float> force(const int, const Geometry_3d::Table&, const Configuration::Table&) const; // [force()] = N

		private:

			static const std::string _binaryVersion;

			unsigned int _threadNum;

			// ***

			friend void* cwrapper_threadedRelax(void*);

			struct ThreadedRelax_Params
			{
				ThreadedRelax_Params(Table* ptr);
				~ThreadedRelax_Params();

				void lock() { pthread_mutex_lock(&mutex); }
				void unlock() { pthread_mutex_unlock(&mutex); }

				Table* obj;

				pthread_mutex_t mutex;

				bool keepRunning;
				pthread_cond_t runCondition;

				int cycleNum;

				int index;
				int delta;

				int phiSwitch;
	
				unsigned short workCnt;
				pthread_cond_t workCondition;
			};

			pthread_t* _threadedRelax_ids;

			ThreadedRelax_Params _threadedRelax_params;

			// ***

			friend void* cwrapper_threadedLocalRelax(void*);

			struct ThreadedLocalRelax_Params
			{
				ThreadedLocalRelax_Params(Table* ptr);
				~ThreadedLocalRelax_Params();

				void lock() { pthread_mutex_lock(&mutex); }
				void unlock() { pthread_mutex_unlock(&mutex); }

				Table* obj;

				pthread_mutex_t mutex;

				bool keepRunning;
				pthread_cond_t runCondition;

				int cycleNum;

				const std::set<int>* localNodes;

				std::set<int>::const_iterator index;

				int delta;

				int phiSwitch;
	
				unsigned short workCnt;
				pthread_cond_t workCondition;
			};

			pthread_t* _threadedLocalRelax_ids;

			ThreadedLocalRelax_Params _threadedLocalRelax_params;

			// ***

			void doRelax(const unsigned int);
			void doLocalRelax(const unsigned int, const std::set<int>&);

			void find_localNodes(const int, const int, std::set<int>*) const;

			//  ***

			double computePotential(const int) const;
			double computeCharge(const int, const Geometry_3d::Table&, const Configuration::Table&) const;
			double computeCoupling(const int, const int, const Geometry_3d::Table&, const Configuration::Table&) const;
	
			// ***

			constexpr static double _epsilon0 = 8.85418781762e-12; // [epsilon0] = As/(Vm)

			constexpr static float _infinityFactor = 1e2f; // should be as big as possible...

			float _relaxationFactor;

			int _phiSwitch;
			int _phiLocalSwitch;

			DynamicVector<Node> _nodes;
	};

	float potential(const Geometry_3d::Table&, const Table&, const Geometry_3d::Point&);
	float potential(const Geometry_3d::Table&, const Table&, const Geometry_3d::Point&, const int);
	
	MathVector3d<float> field(const Geometry_3d::Table&, const Table&, const Geometry_3d::Point&);
	MathVector3d<float> field(const Geometry_3d::Table&, const Table&, const Geometry_3d::Point&, const int);

	// ***

	class FastField
	{
		public:
			FastField(const Table* =0, const Geometry_3d::Table* =0, const unsigned int =0);

			void setGrid(const Table*);
			const Table* grid() const { return _gridTable; }

			void setGeometry(const Geometry_3d::Table*);
			const Geometry_3d::Table* geometry() const { return _geomTable; }

			// ***

			MathVector3d<float> compute(const Geometry_3d::Point&);
			MathVector3d<float> compute(const Geometry_3d::Point&, const int);

			void reset() { _buffer.clear(); }
		private:
			typedef MathVector3d<float> field_t;

			const Table* _gridTable;
			const Geometry_3d::Table* _geomTable;

			unsigned int _maxSize;
			std::map<int,field_t> _buffer;
	};
}

#undef DOUBLE_NAN

#endif
