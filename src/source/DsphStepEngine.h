//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2025 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/).

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics.

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License
 as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>.
*/

/// \file DsphStepEngine.h \brief GPU-based SPH stepping engine for DLL API.

#ifndef _DsphStepEngine_
#define _DsphStepEngine_

#ifdef _WITHGPU

#include "DualSphDef.h"
#include "JCellDivGpuSingle.h"
#include "JArraysGpu.h"
#include <cuda_runtime.h>

//==============================================================================
/// Configuration for the step engine
//==============================================================================
struct StDsphEngineConfig {
  // Domain
  tdouble3 domainMin;
  tdouble3 domainMax;

  // Simulation parameters
  double dp;                    // Particle spacing
  tfloat3 gravity;              // Gravity vector
  TpKernel kernel;              // Kernel type
  TpVisco visco;                // Viscosity type
  float viscoValue;             // Viscosity coefficient
  float viscoBoundFactor;       // Boundary viscosity factor
  TpStep stepMethod;            // Time stepping method
  double cfl;                   // CFL number
  TpBoundary boundary;          // Boundary method
  TpDensity density;            // Density diffusion type
  float ddtValue;               // DDT coefficient
  float rho0;                   // Reference density
  float cs0;                    // Speed of sound
  float gamma;                  // Polytropic constant
  bool simulate2D;              // 2D mode
  double simulate2DPosY;        // Y position for 2D

  // Computed constants
  float kernelH;
  float kernelSize;
  float massFluid;
  float massBound;

  StDsphEngineConfig() { Reset(); }

  void Reset() {
    domainMin = TDouble3(0);
    domainMax = TDouble3(1);
    dp = 0.01;
    gravity = TFloat3(0, 0, -9.81f);
    kernel = KERNEL_Wendland;
    visco = VISCO_Artificial;
    viscoValue = 0.01f;
    viscoBoundFactor = 1.0f;
    stepMethod = STEP_Symplectic;
    cfl = 0.2;
    boundary = BC_DBC;
    density = DDT_DDT2;
    ddtValue = 0.1f;
    rho0 = 1000.0f;
    cs0 = 0.0f;  // Auto-calculate
    gamma = 7.0f;
    simulate2D = false;
    simulate2DPosY = 0.0;
    kernelH = 0;
    kernelSize = 0;
    massFluid = 0;
    massBound = 0;
  }
};

//==============================================================================
/// External buffer configuration
//==============================================================================
struct StDsphExternalBuffer {
  void* devicePtr;              // CUDA device pointer to external buffer
  unsigned int maxParticles;    // Maximum particles buffer can hold
  bool writeFluidOnly;          // Only write fluid particles
  bool enabled;                 // Buffer is configured

  StDsphExternalBuffer() : devicePtr(nullptr), maxParticles(0),
                           writeFluidOnly(true), enabled(false) {}
};

//==============================================================================
/// GPU-based SPH stepping engine
/// Manages particle data and performs simulation steps using CUDA kernels.
//==============================================================================
class DsphStepEngine {
private:
  bool Initialized;
  cudaStream_t Stream;
  bool OwnsStream;

  // Configuration
  StDsphEngineConfig Config;
  StDsphExternalBuffer ExtBuffer;

  // Particle counts
  unsigned int Np;              // Total particles
  unsigned int Npb;             // Boundary particles
  unsigned int Npf;             // Fluid particles

  // Domain configuration
  tdouble3 MapPosMin;
  tdouble3 MapPosMax;
  tuint3 MapCells;
  unsigned int CellCode;
  float Scell;                  // Cell size

  // GPU Arrays (managed by JArraysGpu)
  JArraysGpu* ArraysGpu;

  // Core particle data (GPU pointers)
  unsigned int* Idpg;           // Particle IDs
  typecode* Codeg;              // Particle type codes
  double2* Posxyg;              // Positions XY
  double* Poszg;                // Position Z
  float4* Velrhog;              // Velocities + density (vx,vy,vz,rho)
  float4* PosCellg;             // Position within cell (for neighbor search)
  unsigned int* Dcellg;         // Cell indices

  // Temporary arrays for force computation
  float3* Aceg;                 // Accelerations
  float* ViscDtg;               // Viscous timestep contributions
  float* Arg;                   // Density time derivative (DDT)
  float4* ShiftPosfsg;          // Shifting data

  // For Symplectic scheme
  double2* PosxyPreg;           // Predictor positions XY
  double* PoszPreg;             // Predictor position Z
  float4* VelrhoPreg;           // Predictor velocities + density

  // Cell division
  JCellDivGpuSingle* CellDiv;

  // Simulation state
  double TimeStep;              // Current simulation time
  unsigned int StepCount;       // Number of steps performed
  double LastDt;                // Last timestep used
  int VerletStep;               // Verlet step counter

  // Private methods
  void AllocateGpuMemory(unsigned int np);
  void FreeGpuMemory();
  void ComputeConstants();
  void InitializeCellDivision();
  void RunCellDivide(bool updatePeriodic);
  void UpdatePosCell();
  void InitAcceleration();

  // Force computation
  void PreInteraction();
  void PostInteraction();
  void ComputeForces();

  // Position/velocity update
  double ComputeDt();
  void ComputeVerletStep(double dt);
  void ComputeSymplecticPredictor(double dt);
  void ComputeSymplecticCorrector(double dt);

  // External buffer
  void CopyToExternalBufferInternal();

public:
  DsphStepEngine();
  ~DsphStepEngine();

  /// Initialize the engine with configuration and particle data.
  /// @param config Simulation configuration
  /// @param fluidPos Fluid particle positions [x0,y0,z0,x1,y1,z1,...] (npf*3)
  /// @param fluidVel Fluid particle velocities (can be nullptr for zero)
  /// @param npf Number of fluid particles
  /// @param boundPos Boundary particle positions (npb*3)
  /// @param boundNormals Boundary normals for mDBC (can be nullptr)
  /// @param npb Number of boundary particles
  /// @param stream CUDA stream to use (nullptr for default)
  /// @return true on success
  bool Initialize(const StDsphEngineConfig& config,
                  const double* fluidPos, const double* fluidVel, unsigned int npf,
                  const double* boundPos, const double* boundNormals, unsigned int npb,
                  cudaStream_t stream = nullptr);

  /// Check if engine is initialized.
  bool IsInitialized() const { return Initialized; }

  /// Set external buffer for particle output.
  void SetExternalBuffer(void* devicePtr, unsigned int maxParticles, bool fluidOnly);

  /// Clear external buffer configuration.
  void ClearExternalBuffer();

  /// Get the CUDA stream used by this engine.
  cudaStream_t GetStream() const { return Stream; }

  /// Compute recommended timestep based on CFL condition.
  double ComputeTimeStep();

  /// Perform a single simulation step.
  /// @param dt Timestep to use
  /// @return true on success
  bool Step(double dt);

  /// Perform a single step asynchronously.
  /// @param dt Timestep to use
  /// @return true on success
  bool StepAsync(double dt);

  /// Synchronize (wait for async operations to complete).
  void Synchronize();

  /// Copy particle data to external buffer (if configured).
  void CopyToExternalBuffer();

  /// Get current simulation time.
  double GetSimulationTime() const { return TimeStep; }

  /// Get number of steps performed.
  unsigned int GetStepCount() const { return StepCount; }

  /// Get particle count.
  unsigned int GetParticleCount() const { return Npf; }

  /// Get total particle count (including boundaries).
  unsigned int GetTotalParticleCount() const { return Np; }

  /// Copy positions from GPU to CPU buffer.
  void GetPositions(float* outPositions, unsigned int count);

  /// Copy velocities from GPU to CPU buffer.
  void GetVelocities(float* outVelocities, unsigned int count);

  /// Copy densities from GPU to CPU buffer.
  void GetDensities(float* outDensities, unsigned int count);

  /// Reset simulation to initial state.
  void Reset();

  /// Release all resources.
  void Shutdown();
};

#endif // _WITHGPU
#endif // _DsphStepEngine_
