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
///
/// This engine provides full SPH simulation capabilities including:
/// - Cell division for neighbor search
/// - SPH force computation (pressure, viscosity, density diffusion)
/// - Verlet and Symplectic time integration
/// - Dynamic boundary conditions (DBC/mDBC)

#ifndef _DsphStepEngine_
#define _DsphStepEngine_

#ifdef _WITHGPU

#include "DualSphDef.h"
#include "JCellDivGpuSingle.h"
#include "JCellDivDataGpu.h"
#include "JSphGpu_cte.h"
#include "JArraysGpu.h"
#include "FunSphKernelsCfg.h"
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
  TpKernel kernel;              // Kernel type (KERNEL_Cubic, KERNEL_Wendland)
  TpVisco visco;                // Viscosity type
  float viscoValue;             // Viscosity coefficient
  float viscoBoundFactor;       // Boundary viscosity factor
  TpStep stepMethod;            // Time stepping method (STEP_Verlet, STEP_Symplectic)
  double cfl;                   // CFL number
  TpBoundary boundary;          // Boundary method (BC_DBC, BC_MDBC)
  TpDensity density;            // Density diffusion type
  float ddtValue;               // DDT coefficient
  float rho0;                   // Reference density [kg/m3]
  float cs0;                    // Speed of sound [m/s] (0 = auto-calculate)
  float gamma;                  // Polytropic constant (7 for water)
  bool simulate2D;              // 2D mode
  double simulate2DPosY;        // Y position for 2D plane

  // Computed constants (filled by ComputeConstants)
  float kernelH;                // Smoothing length
  float kernelSize;             // Kernel support radius (2h)
  float kernelSize2;            // kernelSize^2
  float massFluid;              // Fluid particle mass
  float massBound;              // Boundary particle mass
  float eta2;                   // (0.1*h)^2 for viscosity
  float cteB;                   // Pressure constant B
  float movLimit;               // Maximum movement limit
  float ddtkh;                  // DDT constant
  float ddtgz;                  // DDT gravity constant
  fsph::StKCubicCte kcubic;     // Cubic kernel constants
  fsph::StKWendlandCte kwend;   // Wendland kernel constants

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
    cs0 = 0.0f;
    gamma = 7.0f;
    simulate2D = false;
    simulate2DPosY = 0.0;
    kernelH = 0;
    kernelSize = 0;
    kernelSize2 = 0;
    massFluid = 0;
    massBound = 0;
    eta2 = 0;
    cteB = 0;
    movLimit = 0;
    ddtkh = 0;
    ddtgz = 0;
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
/// Manages particle data and performs full SPH simulation steps using CUDA.
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
  unsigned int NpbOk;           // Valid boundary particles
  unsigned int Npf;             // Fluid particles

  // Domain configuration
  tdouble3 MapPosMin;           // Map minimum position
  tdouble3 MapPosMax;           // Map maximum position
  tdouble3 MapRealPosMin;       // Real map minimum
  tdouble3 MapRealSize;         // Real map size
  tuint3 MapCells;              // Number of cells in each dimension
  unsigned int DomCellCode;     // Domain cell encoding
  tdouble3 DomPosMin;           // Domain minimum position
  float Scell;                  // Cell size
  float PosCellSize;            // Position cell size

  // Cell division
  JCellDivGpuSingle* CellDivSingle;
  StDivDataGpu DivData;

  // GPU memory for particle data
  // Core arrays
  unsigned int* Idpg;           // Particle IDs
  typecode* Codeg;              // Particle type codes
  double2* Posxyg;              // Positions XY
  double* Poszg;                // Position Z
  float4* Velrhog;              // Velocities + density (vx,vy,vz,rho)
  float4* PosCellg;             // Position within cell
  unsigned int* Dcellg;         // Cell indices

  // Force computation arrays
  float3* Aceg;                 // Accelerations
  float* ViscDtg;               // Viscous timestep
  float* Arg;                   // Density rate (drho/dt)
  float* Deltag;                // Delta-SPH term
  float4* ShiftPosfsg;          // Shifting positions

  // Movement arrays
  double2* Movxyg;              // Movement XY (per step)
  double* Movzg;                // Movement Z

  // For Symplectic scheme
  double2* PosxyPreg;           // Predictor positions XY
  double* PoszPreg;             // Predictor position Z
  float4* VelrhoPreg;           // Predictor velocities + density

  // For Verlet scheme
  float4* VelrhoM1g;            // Previous step velocities

  // Auxiliary memory for reductions
  float* AuxMemg;
  unsigned int AuxMemSize;

  // Simulation state
  double TimeStep;              // Current simulation time
  unsigned int StepCount;       // Number of steps performed
  double LastDt;                // Last timestep used
  int VerletStep;               // Verlet step counter
  float ViscDtMax;              // Maximum viscous dt
  float AceMax;                 // Maximum acceleration

  // Private methods
  void AllocateGpuMemory(unsigned int np);
  void FreeGpuMemory();
  void ComputeConstants();
  void SetupCellDivision();
  void UploadConstants();
  void RunCellDivide(bool updatePeriodic);
  void SortParticleArrays();

  // Pre/Post interaction
  void PreInteraction_Forces();
  void PosInteraction_Forces();

  // Force computation
  void Interaction_Forces();
  float ComputeViscDtMax();
  float ComputeAceMax();

  // Position/velocity update
  double ComputeDtVariable();
  void ComputeStepVerlet(double dt);
  void ComputeStepSymplectic(double dt);
  void ComputeStepPos(double dt);

  // External buffer
  void CopyToExternalBufferInternal();

public:
  DsphStepEngine();
  ~DsphStepEngine();

  /// Initialize the engine with configuration and particle data.
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

  /// Perform a single simulation step (synchronous).
  bool Step(double dt);

  /// Perform a single step asynchronously.
  bool StepAsync(double dt);

  /// Synchronize (wait for async operations to complete).
  void Synchronize();

  /// Copy particle data to external buffer (if configured).
  void CopyToExternalBuffer();

  /// Get current simulation time.
  double GetSimulationTime() const { return TimeStep; }

  /// Get number of steps performed.
  unsigned int GetStepCount() const { return StepCount; }

  /// Get fluid particle count.
  unsigned int GetParticleCount() const { return Npf; }

  /// Get total particle count (including boundaries).
  unsigned int GetTotalParticleCount() const { return Np; }

  /// Copy positions from GPU to CPU buffer (interleaved xyz).
  void GetPositions(float* outPositions, unsigned int count);

  /// Copy velocities from GPU to CPU buffer (interleaved xyz).
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
