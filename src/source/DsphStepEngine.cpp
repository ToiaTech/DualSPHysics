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

/// \file DsphStepEngine.cpp \brief Implementation of GPU-based SPH stepping engine.

#ifdef _WITHGPU

#include "DsphStepEngine.h"
#include "DsphStepEngine_ker.h"
#include "FunSphKernelsCfg.h"
#include "Functions.h"
#include "FunctionsCuda.h"
#include "JSphGpuSimple_ker.h"
#include "JCellDivGpu_ker.h"
#include "JReduSum_ker.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

//==============================================================================
// Constructor / Destructor
//==============================================================================
DsphStepEngine::DsphStepEngine()
  : Initialized(false)
  , Stream(nullptr)
  , OwnsStream(false)
  , Np(0), Npb(0), Npf(0)
  , CellCode(0)
  , Scell(0)
  , ArraysGpu(nullptr)
  , Idpg(nullptr), Codeg(nullptr)
  , Posxyg(nullptr), Poszg(nullptr)
  , Velrhog(nullptr), PosCellg(nullptr), Dcellg(nullptr)
  , Aceg(nullptr), ViscDtg(nullptr), Arg(nullptr), ShiftPosfsg(nullptr)
  , PosxyPreg(nullptr), PoszPreg(nullptr), VelrhoPreg(nullptr)
  , CellDiv(nullptr)
  , TimeStep(0), StepCount(0), LastDt(0), VerletStep(0)
{
  MapPosMin = MapPosMax = TDouble3(0);
  MapCells = TUint3(0);
}

DsphStepEngine::~DsphStepEngine() {
  Shutdown();
}

//==============================================================================
// Memory Management
//==============================================================================
void DsphStepEngine::AllocateGpuMemory(unsigned int np) {
  if(ArraysGpu) delete ArraysGpu;
  ArraysGpu = new JArraysGpu();

  // Core arrays
  size_t memSize = 0;
  cudaMalloc(&Idpg, np * sizeof(unsigned int));
  cudaMalloc(&Codeg, np * sizeof(typecode));
  cudaMalloc(&Posxyg, np * sizeof(double2));
  cudaMalloc(&Poszg, np * sizeof(double));
  cudaMalloc(&Velrhog, np * sizeof(float4));
  cudaMalloc(&PosCellg, np * sizeof(float4));
  cudaMalloc(&Dcellg, np * sizeof(unsigned int));

  // Force computation arrays
  cudaMalloc(&Aceg, np * sizeof(float3));
  cudaMalloc(&ViscDtg, np * sizeof(float));
  cudaMalloc(&Arg, np * sizeof(float));

  // Symplectic arrays (if needed)
  if(Config.stepMethod == STEP_Symplectic) {
    cudaMalloc(&PosxyPreg, np * sizeof(double2));
    cudaMalloc(&PoszPreg, np * sizeof(double));
    cudaMalloc(&VelrhoPreg, np * sizeof(float4));
  }

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    throw std::runtime_error(std::string("GPU memory allocation failed: ") + cudaGetErrorString(err));
  }
}

void DsphStepEngine::FreeGpuMemory() {
  if(Idpg) { cudaFree(Idpg); Idpg = nullptr; }
  if(Codeg) { cudaFree(Codeg); Codeg = nullptr; }
  if(Posxyg) { cudaFree(Posxyg); Posxyg = nullptr; }
  if(Poszg) { cudaFree(Poszg); Poszg = nullptr; }
  if(Velrhog) { cudaFree(Velrhog); Velrhog = nullptr; }
  if(PosCellg) { cudaFree(PosCellg); PosCellg = nullptr; }
  if(Dcellg) { cudaFree(Dcellg); Dcellg = nullptr; }
  if(Aceg) { cudaFree(Aceg); Aceg = nullptr; }
  if(ViscDtg) { cudaFree(ViscDtg); ViscDtg = nullptr; }
  if(Arg) { cudaFree(Arg); Arg = nullptr; }
  if(ShiftPosfsg) { cudaFree(ShiftPosfsg); ShiftPosfsg = nullptr; }
  if(PosxyPreg) { cudaFree(PosxyPreg); PosxyPreg = nullptr; }
  if(PoszPreg) { cudaFree(PoszPreg); PoszPreg = nullptr; }
  if(VelrhoPreg) { cudaFree(VelrhoPreg); VelrhoPreg = nullptr; }

  if(ArraysGpu) { delete ArraysGpu; ArraysGpu = nullptr; }
  if(CellDiv) { delete CellDiv; CellDiv = nullptr; }
}

//==============================================================================
// Initialization
//==============================================================================
void DsphStepEngine::ComputeConstants() {
  double dp = Config.dp;

  // Kernel constants
  double coefH, coefK;
  if(Config.kernel == KERNEL_Wendland) {
    coefH = fsph::GetKernelWendlandFactorH();
    coefK = fsph::GetKernelWendlandFactorK();
  } else {
    coefH = fsph::GetKernelCubicFactorH();
    coefK = fsph::GetKernelCubicFactorK();
  }

  Config.kernelH = float(coefH * sqrt(3.0 * dp * dp));
  Config.kernelSize = float(coefK * Config.kernelH);

  // Mass calculation
  double volume = dp * dp * dp;
  Config.massFluid = float(Config.rho0 * volume);
  Config.massBound = Config.massFluid;

  // Speed of sound (if not set)
  if(Config.cs0 <= 0) {
    double domainHeight = Config.domainMax.z - Config.domainMin.z;
    double vMax = sqrt(2.0 * fabs(Config.gravity.z) * domainHeight);
    Config.cs0 = float(10.0 * std::max(vMax, 1.0));
  }

  // Cell size for neighbor search
  Scell = Config.kernelSize;
}

void DsphStepEngine::InitializeCellDivision() {
  // Calculate domain with margin
  double margin = Config.kernelSize * 4.0;  // Safety margin
  MapPosMin = Config.domainMin - TDouble3(margin);
  MapPosMax = Config.domainMax + TDouble3(margin);

  // Calculate cell grid
  tdouble3 mapSize = MapPosMax - MapPosMin;
  MapCells.x = unsigned(ceil(mapSize.x / Scell));
  MapCells.y = unsigned(ceil(mapSize.y / Scell));
  MapCells.z = unsigned(ceil(mapSize.z / Scell));

  // Encode cell code for spatial hashing
  unsigned int ncx = MapCells.x;
  unsigned int ncy = MapCells.y;
  unsigned int ncz = MapCells.z;

  // Bits needed for each dimension
  unsigned int bx = 0, by = 0, bz = 0;
  while((1u << bx) < ncx) bx++;
  while((1u << by) < ncy) by++;
  while((1u << bz) < ncz) bz++;

  CellCode = (bx << 20) | (by << 10) | bz;

  // Create cell division object
  if(CellDiv) delete CellDiv;
  CellDiv = new JCellDivGpuSingle();
  // Note: Full initialization would require more setup
  // This is simplified for the DLL API
}

bool DsphStepEngine::Initialize(const StDsphEngineConfig& config,
                                 const double* fluidPos, const double* fluidVel, unsigned int npf,
                                 const double* boundPos, const double* boundNormals, unsigned int npb,
                                 cudaStream_t stream) {
  if(Initialized) Shutdown();

  try {
    Config = config;
    Npf = npf;
    Npb = npb;
    Np = npf + npb;

    // Setup stream
    if(stream) {
      Stream = stream;
      OwnsStream = false;
    } else {
      cudaStreamCreate(&Stream);
      OwnsStream = true;
    }

    // Compute SPH constants
    ComputeConstants();

    // Initialize cell division
    InitializeCellDivision();

    // Allocate GPU memory
    AllocateGpuMemory(Np);

    // Prepare host data for upload
    std::vector<unsigned int> idp(Np);
    std::vector<typecode> code(Np);
    std::vector<double2> posxy(Np);
    std::vector<double> posz(Np);
    std::vector<float4> velrho(Np);

    unsigned int idx = 0;

    // Boundary particles first (indices 0 to Npb-1)
    for(unsigned int i = 0; i < Npb; i++) {
      idp[idx] = idx;
      code[idx] = CODE_TYPE_FIXED;  // Fixed boundary

      posxy[idx].x = boundPos[i * 3 + 0];
      posxy[idx].y = boundPos[i * 3 + 1];
      posz[idx] = boundPos[i * 3 + 2];

      velrho[idx].x = 0.0f;
      velrho[idx].y = 0.0f;
      velrho[idx].z = 0.0f;
      velrho[idx].w = Config.rho0;

      idx++;
    }

    // Fluid particles (indices Npb to Np-1)
    for(unsigned int i = 0; i < Npf; i++) {
      idp[idx] = idx;
      code[idx] = CODE_TYPE_FLUID;

      posxy[idx].x = fluidPos[i * 3 + 0];
      posxy[idx].y = fluidPos[i * 3 + 1];
      posz[idx] = fluidPos[i * 3 + 2];

      if(fluidVel) {
        velrho[idx].x = float(fluidVel[i * 3 + 0]);
        velrho[idx].y = float(fluidVel[i * 3 + 1]);
        velrho[idx].z = float(fluidVel[i * 3 + 2]);
      } else {
        velrho[idx].x = 0.0f;
        velrho[idx].y = 0.0f;
        velrho[idx].z = 0.0f;
      }
      velrho[idx].w = Config.rho0;

      idx++;
    }

    // Upload to GPU
    cudaMemcpyAsync(Idpg, idp.data(), Np * sizeof(unsigned int), cudaMemcpyHostToDevice, Stream);
    cudaMemcpyAsync(Codeg, code.data(), Np * sizeof(typecode), cudaMemcpyHostToDevice, Stream);
    cudaMemcpyAsync(Posxyg, posxy.data(), Np * sizeof(double2), cudaMemcpyHostToDevice, Stream);
    cudaMemcpyAsync(Poszg, posz.data(), Np * sizeof(double), cudaMemcpyHostToDevice, Stream);
    cudaMemcpyAsync(Velrhog, velrho.data(), Np * sizeof(float4), cudaMemcpyHostToDevice, Stream);

    // Initialize cell positions
    cudaMemsetAsync(PosCellg, 0, Np * sizeof(float4), Stream);
    cudaMemsetAsync(Dcellg, 0, Np * sizeof(unsigned int), Stream);

    // Initialize accelerations to gravity for fluid, zero for boundary
    cudaMemsetAsync(Aceg, 0, Np * sizeof(float3), Stream);

    cudaStreamSynchronize(Stream);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
      throw std::runtime_error(std::string("Initialization failed: ") + cudaGetErrorString(err));
    }

    TimeStep = 0.0;
    StepCount = 0;
    LastDt = 0.0;
    VerletStep = 0;
    Initialized = true;

    return true;
  }
  catch(const std::exception& e) {
    FreeGpuMemory();
    return false;
  }
}

//==============================================================================
// Cell Division
//==============================================================================
void DsphStepEngine::RunCellDivide(bool updatePeriodic) {
  // TODO: Implement proper cell division using JCellDivGpuSingle
  // This requires reordering particles based on spatial location
  // For now, we update PosCell which is used for neighbor search
  UpdatePosCell();
}

void DsphStepEngine::UpdatePosCell() {
  // Update position within cell for neighbor search
  // This kernel computes the relative position within each particle's cell
  cusphs::UpdatePosCell(Np, MapPosMin, Scell,
                        Posxyg, Poszg, PosCellg, Stream);
}

void DsphStepEngine::InitAcceleration() {
  // Initialize accelerations: gravity for fluid, zero for boundary
  cusphs::InitAceGravity(Np, Npb, Config.gravity, Aceg, Stream);
  // For 2D mode, reset Y acceleration to zero
  if(Config.simulate2D) {
    cusphs::Resety(Npf, Npb, Aceg, Stream);
  }
}

//==============================================================================
// Force Computation
//==============================================================================
void DsphStepEngine::PreInteraction() {
  // Allocate/prepare temporary arrays for force computation
  cudaMemsetAsync(ViscDtg, 0, Np * sizeof(float), Stream);
  cudaMemsetAsync(Arg, 0, Np * sizeof(float), Stream);
  InitAcceleration();
}

void DsphStepEngine::PostInteraction() {
  // Cleanup after force computation (if needed)
}

void DsphStepEngine::ComputeForces() {
  // TODO: Implement full force computation using cusph::Interaction_Forces
  // This requires setting up StInterParmsg structure with all parameters
  // and calling the appropriate kernel based on configuration

  // For now, this is a placeholder that just applies gravity
  // The actual implementation would call:
  // cusph::Interaction_Forces(interactionParams);

  // Simplified: just keep gravity acceleration (already initialized)
}

//==============================================================================
// Timestep Computation
//==============================================================================
double DsphStepEngine::ComputeDt() {
  // CFL condition: dt = CFL * h / (cs + vmax)
  double h = Config.kernelH;
  double cs = Config.cs0;
  double cfl = Config.cfl;

  // TODO: Compute actual vmax from particle velocities using reduction
  // For now, use a conservative estimate
  double vmax = 10.0;

  double dt = cfl * h / (cs + vmax);

  // Additional constraints
  double dtVisc = 0.125 * h * h / (Config.viscoValue + 1e-10);
  dt = std::min(dt, dtVisc);

  return dt;
}

double DsphStepEngine::ComputeTimeStep() {
  if(!Initialized) return 0.001;
  return ComputeDt();
}

//==============================================================================
// Position/Velocity Update
//==============================================================================
void DsphStepEngine::ComputeVerletStep(double dt) {
  // Verlet integration using simplified kernel (gravity-only forces)
  dsphker::SimpleVerletUpdate(
    Np, Npb, Aceg, Config.gravity, dt,
    Posxyg, Poszg, Velrhog, Config.rho0, Stream);

  TimeStep += dt;
  StepCount++;
  LastDt = dt;
  VerletStep++;
}

void DsphStepEngine::ComputeSymplecticPredictor(double dt) {
  // Symplectic predictor step (half step)
  double dtm = dt * 0.5;

  // Save current state to Pre arrays
  cudaMemcpyAsync(PosxyPreg, Posxyg, Np * sizeof(double2), cudaMemcpyDeviceToDevice, Stream);
  cudaMemcpyAsync(PoszPreg, Poszg, Np * sizeof(double), cudaMemcpyDeviceToDevice, Stream);
  cudaMemcpyAsync(VelrhoPreg, Velrhog, Np * sizeof(float4), cudaMemcpyDeviceToDevice, Stream);

  // Compute predictor step
  dsphker::SimpleSymplecticPre(
    Np, Npb, Aceg, Config.gravity, dtm,
    PosxyPreg, PoszPreg, VelrhoPreg,
    Posxyg, Poszg, Velrhog, Config.rho0, Stream);
}

void DsphStepEngine::ComputeSymplecticCorrector(double dt) {
  // Symplectic corrector step
  double dtm = dt * 0.5;

  // Compute corrector step
  dsphker::SimpleSymplecticCor(
    Np, Npb, Aceg, Config.gravity, dtm, dt,
    PosxyPreg, PoszPreg, VelrhoPreg,
    Posxyg, Poszg, Velrhog, Config.rho0, Stream);

  TimeStep += dt;
  StepCount++;
  LastDt = dt;
}

//==============================================================================
// Stepping
//==============================================================================
bool DsphStepEngine::Step(double dt) {
  if(!StepAsync(dt)) return false;
  Synchronize();
  return true;
}

bool DsphStepEngine::StepAsync(double dt) {
  if(!Initialized) return false;
  if(dt <= 0) return false;

  try {
    // 1. Cell division (spatial sorting)
    RunCellDivide(true);

    // 2. Prepare for force computation
    PreInteraction();

    // 3. Compute forces
    ComputeForces();

    // 4. Position/velocity update
    if(Config.stepMethod == STEP_Verlet) {
      ComputeVerletStep(dt);
    } else {
      // Symplectic scheme
      ComputeSymplecticPredictor(dt);
      RunCellDivide(true);  // Reorganize after predictor
      PreInteraction();
      ComputeForces();
      ComputeSymplecticCorrector(dt);
    }

    // 5. Post-step cleanup
    PostInteraction();

    // 6. Copy to external buffer if configured
    if(ExtBuffer.enabled) {
      CopyToExternalBufferInternal();
    }

    return true;
  }
  catch(...) {
    return false;
  }
}

void DsphStepEngine::Synchronize() {
  if(Stream) {
    cudaStreamSynchronize(Stream);
  }
}

//==============================================================================
// External Buffer
//==============================================================================
void DsphStepEngine::SetExternalBuffer(void* devicePtr, unsigned int maxParticles, bool fluidOnly) {
  ExtBuffer.devicePtr = devicePtr;
  ExtBuffer.maxParticles = maxParticles;
  ExtBuffer.writeFluidOnly = fluidOnly;
  ExtBuffer.enabled = (devicePtr != nullptr && maxParticles > 0);
}

void DsphStepEngine::ClearExternalBuffer() {
  ExtBuffer.devicePtr = nullptr;
  ExtBuffer.maxParticles = 0;
  ExtBuffer.enabled = false;
}

void DsphStepEngine::CopyToExternalBuffer() {
  if(!Initialized || !ExtBuffer.enabled) return;
  CopyToExternalBufferInternal();
  Synchronize();
}

void DsphStepEngine::CopyToExternalBufferInternal() {
  if(!ExtBuffer.enabled) return;

  unsigned int count = ExtBuffer.writeFluidOnly ? Npf : Np;
  unsigned int offset = ExtBuffer.writeFluidOnly ? Npb : 0;

  if(count > ExtBuffer.maxParticles) {
    count = ExtBuffer.maxParticles;
  }

  // Copy particle data to external buffer in interleaved format:
  // posX, posY, posZ, density, velX, velY, velZ, pressure (32 bytes per particle)
  dsphker::CopyToExternalBuffer(
    count,
    Posxyg + offset,
    Poszg + offset,
    Velrhog + offset,
    Config.rho0,
    Config.cs0,
    Config.gamma,
    static_cast<float*>(ExtBuffer.devicePtr),
    Stream);
}

//==============================================================================
// Data Access
//==============================================================================
void DsphStepEngine::GetPositions(float* outPositions, unsigned int count) {
  if(!Initialized || !outPositions || count == 0) return;
  if(count > Npf) count = Npf;

  // Download from GPU and convert to interleaved format
  std::vector<double2> posxy(count);
  std::vector<double> posz(count);

  cudaMemcpyAsync(posxy.data(), Posxyg + Npb, count * sizeof(double2), cudaMemcpyDeviceToHost, Stream);
  cudaMemcpyAsync(posz.data(), Poszg + Npb, count * sizeof(double), cudaMemcpyDeviceToHost, Stream);
  cudaStreamSynchronize(Stream);

  for(unsigned int i = 0; i < count; i++) {
    outPositions[i * 3 + 0] = float(posxy[i].x);
    outPositions[i * 3 + 1] = float(posxy[i].y);
    outPositions[i * 3 + 2] = float(posz[i]);
  }
}

void DsphStepEngine::GetVelocities(float* outVelocities, unsigned int count) {
  if(!Initialized || !outVelocities || count == 0) return;
  if(count > Npf) count = Npf;

  std::vector<float4> velrho(count);
  cudaMemcpyAsync(velrho.data(), Velrhog + Npb, count * sizeof(float4), cudaMemcpyDeviceToHost, Stream);
  cudaStreamSynchronize(Stream);

  for(unsigned int i = 0; i < count; i++) {
    outVelocities[i * 3 + 0] = velrho[i].x;
    outVelocities[i * 3 + 1] = velrho[i].y;
    outVelocities[i * 3 + 2] = velrho[i].z;
  }
}

void DsphStepEngine::GetDensities(float* outDensities, unsigned int count) {
  if(!Initialized || !outDensities || count == 0) return;
  if(count > Npf) count = Npf;

  std::vector<float4> velrho(count);
  cudaMemcpyAsync(velrho.data(), Velrhog + Npb, count * sizeof(float4), cudaMemcpyDeviceToHost, Stream);
  cudaStreamSynchronize(Stream);

  for(unsigned int i = 0; i < count; i++) {
    outDensities[i] = velrho[i].w;
  }
}

//==============================================================================
// Reset / Shutdown
//==============================================================================
void DsphStepEngine::Reset() {
  TimeStep = 0.0;
  StepCount = 0;
  LastDt = 0.0;
  VerletStep = 0;
  // Note: Does not reset particle data - would need re-initialization
}

void DsphStepEngine::Shutdown() {
  FreeGpuMemory();

  if(OwnsStream && Stream) {
    cudaStreamDestroy(Stream);
  }
  Stream = nullptr;
  OwnsStream = false;

  Initialized = false;
  Np = Npb = Npf = 0;
  TimeStep = 0.0;
  StepCount = 0;
}

#endif // _WITHGPU
