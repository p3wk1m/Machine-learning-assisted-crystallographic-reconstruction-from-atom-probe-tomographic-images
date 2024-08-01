#ifndef TAPSIM_TRAJECTORY_3D_H
#define TAPSIM_TRAJECTORY_3D_H

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

#include <vector>
#include <limits>

#include "../vector/mathVector_3d.h"

#include "geometry_3d.h"
#include "grid_3d.h"

class Trajectory_3d
{
	public:
		typedef MathVector3d<float> Vector3d;

		// ***

		class phaseVector
		{
			public:
				phaseVector()
					: _time(),
					  _position(),
					  _velocity(),
					  _tetIndex(-1)
				{}

				phaseVector(const float t, const Vector3d& p, const Vector3d& v, const int i =-1)
					: _time(t),
					  _position(p),
					  _velocity(v),
					  _tetIndex(i)
				{}
		
				const float& time() const { return _time; }
				float& time() { return _time; }

				const Vector3d& position() const { return _position; }
				Vector3d& position() { return _position; }

				float position(short axis) const { return _position[axis]; }
				float& position(const short axis) { return _position[axis]; }

				const Vector3d& velocity() const { return _velocity; }
				Vector3d& velocity() { return _velocity; }

				float velocity(short axis) const { return _velocity[axis]; }
				float& velocity(const short axis) { return _velocity[axis]; }

				int tetIndex() const { return _tetIndex; }
				int& tetIndex() { return _tetIndex; }

			private:
				float _time;			// [_time] = s
				Vector3d _position;		// [_position] = m
				Vector3d _velocity;		// [_velocity] = m/s

				int _tetIndex;			// [_tetIndex] = 1
		};

		// ***

		enum { O1_INTEGRATOR, O5_INTEGRATOR };

		enum { NONE, ERROR_RESTRICTED, GEOMETRY_RESTRICTED, ADAPTIVE };

		enum { INVALID =-1, NO_INTEGRATION, NO_FIELD, STEPPER_LIMIT_PASSED, ITERATION_LIMIT_PASSED, GEOMETRY_LIMIT_PASSED, SYSTEM_LIMIT_PASSED };

		constexpr static float eCharge = 1.60217733e-19;	// [eCharge] = C
		constexpr static float protonMass = 1.6726231e-27;	// [protonMass] = kg
		constexpr static float amu2kg = 1.660538782e-27;	// [amu2kg] = kg/amu

		static const char* status_str(const int);
		static const char* integrator_str(const int);
		static const char* stepper_str(const int);

		// ***

		Trajectory_3d(const Geometry_3d::Table* =0, const Grid_3d::Table* =0, const int =O5_INTEGRATOR, const int =ERROR_RESTRICTED);

		void setGeometry(const Geometry_3d::Table* geomTable) {_fastField.setGeometry(geomTable); }
		const Geometry_3d::Table* geometry() const { return _fastField.geometry(); }

		void setGrid(const Grid_3d::Table* gridTable) { _fastField.setGrid(gridTable); }
		const Grid_3d::Table* grid() const { return _fastField.grid(); }

		void setIntegratorType(const int type) { _integratorType = type; }
		int integratorType() const { return _integratorType; }

		void setStepperType(const int type) { _stepperType = type; }
		int stepperType() const { return _stepperType; }

		void setStepperLimit(const unsigned long limit) { _stepperLimit = limit; }
		unsigned long stepperLimit() const { return _stepperLimit; }

		void setIterationLimit(const unsigned long limit) { _iterationLimit = limit; }
		unsigned long iterationLimit() const { return _iterationLimit; }

		void setTimeStepLimit(const float limit) { _timeStepLimit = limit; }
		float timeStepLimit() const { return _timeStepLimit; }

		// ***

		void init(const Vector3d& position,const float charge, const float mass, const int guessIndex =-1)
			{ init(phaseVector(0.0f,position,Vector3d(0.0),guessIndex),charge,mass); }

		void init(const Vector3d& position, const Vector3d& velocity, const float charge, const float mass, const int guessIndex =-1)
			{ init(phaseVector(0.0f,position,velocity,guessIndex),charge,mass); }

		void init(phaseVector, const float, const float);

		int integrate(const float, const float =0.0f);
		int integrate(const float, const float, const float);

		bool extrapolate(const int, const float);

		void reset();

		// ***

		int status() const { return _status; }

		float charge() const { return _charge; }
		float mass() const { return _mass; }

		const std::vector<phaseVector>& data() const { return _data; }

		phaseVector error_estimate() const { return _error_estimate; }

	private:
		bool ready() const;

		phaseVector errorRestricted_stepper(float* delta, const float errorLimit, phaseVector* error =0);
		phaseVector geometryRestricted_stepper(float* delta, const std::set<int>& allowedTetrahedra, phaseVector* error =0);
		phaseVector adaptive_stepper(float* delta, const float errorLimit, const std::set<int>& allowedTetrahedra, phaseVector* error =0);

		phaseVector doIntegration(const float delta, phaseVector* error =0) const;
		phaseVector o1_integrator(const float delta, phaseVector* error =0) const;
		phaseVector o5_integrator(const float delta, phaseVector* error =0) const;

		phaseVector cumulative_error(const phaseVector& a, const phaseVector& b) const;

		// ***

		constexpr static float _safetyFactor = 0.9f;

		// ***

		mutable Grid_3d::FastField _fastField;

		int _integratorType;

		int _stepperType;
		unsigned long _stepperLimit;

		unsigned long _iterationLimit;

		float _timeStepLimit;

		// ***

		int _status;

		float _charge;	// [_charge] = C
		float _mass;	// [_mass] = kg

		std::vector<phaseVector> _data;

		phaseVector _error_estimate;
};

#endif
