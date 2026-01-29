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
#include "JSphGpu_ker.h"
#include "JSphGpu_cte.h"
#include "JSphGpuSimple_ker.h"
#include "JCellDivGpuSingle_ker.h"
#include "JReduSum_ker.h"
#include "JDsDcellDef.h"
#include "JDsDcell.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <vector>

//==============================================================================
// Constructor / Destructor
//==============================================================================
DsphStepEngine::DsphStepEngine()
  : Initialized(false)
  , Stream(nullptr)
  , OwnsStream(false)
  , Np(0), Npb(0), NpbOk(0), Npf(0)
  , DomCellCode(0)
  , Scell(0), PosCellSize(0)
  , CellDivSingle(nullptr)
  , Idpg(nullptr), Codeg(nullptr)
  , Posxyg(nullptr), Poszg(nullptr)
  , Velrhog(nullptr), PosCellg(nullptr), Dcellg(nullptr)
  , Aceg(nullptr), ViscDtg(nullptr), Arg(nullptr), Deltag(nullptr), ShiftPosfsg(nullptr)
  , Movxyg(nullptr), Movzg(nullptr)
  , PosxyPreg(nullptr), PoszPreg(nullptr), VelrhoPreg(nullptr)
  , VelrhoM1g(nullptr)
  , AuxMemg(nullptr), AuxMemSize(0)
  , TimeStep(0), StepCount(0), LastDt(0), VerletStep(0)
  , ViscDtMax(0), AceMax(0)
  , BoundaryCount(0)
  , BoundLocalPosg(nullptr), BoundLocalNormg(nullptr)
  , BoundWorldPosg(nullptr), BoundWorldNormg(nullptr)
  , BoundVelg(nullptr)
  , BoundForcesg(nullptr), BoundTorquesg(nullptr)
  , FluidTypeCount(0), FluidTypeg(nullptr)
{
  MapPosMin = MapPosMax = TDouble3(0);
  MapRealPosMin = TDouble3(0);
  MapRealSize = TDouble3(0);
  MapCells = TUint3(0);
  DomPosMin = TDouble3(0);
  DivData = DivDataGpuNull();
}

DsphStepEngine::~DsphStepEngine() {
  Shutdown();
}

//==============================================================================
// Memory Management
//==============================================================================
void DsphStepEngine::AllocateGpuMemory(unsigned int np) {
  FreeGpuMemory();

  // Core arrays
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

  // Delta-SPH (if using DDT)
  if(Config.density != DDT_None) {
    cudaMalloc(&Deltag, np * sizeof(float));
  }

  // Movement arrays
  cudaMalloc(&Movxyg, np * sizeof(double2));
  cudaMalloc(&Movzg, np * sizeof(double));

  // Symplectic arrays
  if(Config.stepMethod == STEP_Symplectic) {
    cudaMalloc(&PosxyPreg, np * sizeof(double2));
    cudaMalloc(&PoszPreg, np * sizeof(double));
    cudaMalloc(&VelrhoPreg, np * sizeof(float4));
  }

  // Verlet arrays
  if(Config.stepMethod == STEP_Verlet) {
    cudaMalloc(&VelrhoM1g, np * sizeof(float4));
  }

  // Auxiliary memory for reductions
  AuxMemSize = cusph::ReduMaxFloatSize(np);
  cudaMalloc(&AuxMemg, AuxMemSize * sizeof(float));

  // Per-particle fluid type ID (only for fluid particles, allocated for all)
  cudaMalloc(&FluidTypeg, np * sizeof(unsigned char));
  cudaMemset(FluidTypeg, 0, np * sizeof(unsigned char));  // Default to fluid type 0

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    FreeGpuMemory();
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
  if(Deltag) { cudaFree(Deltag); Deltag = nullptr; }
  if(ShiftPosfsg) { cudaFree(ShiftPosfsg); ShiftPosfsg = nullptr; }
  if(Movxyg) { cudaFree(Movxyg); Movxyg = nullptr; }
  if(Movzg) { cudaFree(Movzg); Movzg = nullptr; }
  if(PosxyPreg) { cudaFree(PosxyPreg); PosxyPreg = nullptr; }
  if(PoszPreg) { cudaFree(PoszPreg); PoszPreg = nullptr; }
  if(VelrhoPreg) { cudaFree(VelrhoPreg); VelrhoPreg = nullptr; }
  if(VelrhoM1g) { cudaFree(VelrhoM1g); VelrhoM1g = nullptr; }
  if(AuxMemg) { cudaFree(AuxMemg); AuxMemg = nullptr; }
  if(FluidTypeg) { cudaFree(FluidTypeg); FluidTypeg = nullptr; }

  // Free boundary arrays
  if(BoundLocalPosg) { cudaFree(BoundLocalPosg); BoundLocalPosg = nullptr; }
  if(BoundLocalNormg) { cudaFree(BoundLocalNormg); BoundLocalNormg = nullptr; }
  if(BoundWorldPosg) { cudaFree(BoundWorldPosg); BoundWorldPosg = nullptr; }
  if(BoundWorldNormg) { cudaFree(BoundWorldNormg); BoundWorldNormg = nullptr; }
  if(BoundVelg) { cudaFree(BoundVelg); BoundVelg = nullptr; }
  if(BoundForcesg) { cudaFree(BoundForcesg); BoundForcesg = nullptr; }
  if(BoundTorquesg) { cudaFree(BoundTorquesg); BoundTorquesg = nullptr; }

  if(CellDivSingle) { delete CellDivSingle; CellDivSingle = nullptr; }
}

//==============================================================================
// Constant Computation
//==============================================================================
void DsphStepEngine::ComputeConstants() {
  double dp = Config.dp;

  // Kernel constants
  if(Config.kernel == KERNEL_Wendland) {
    Config.kwend = fsph::GetKernelWendland(Config.simulate2D ? 2 : 3, float(dp));
    Config.kernelH = Config.kwend.h;
    Config.kernelSize = Config.kwend.kernelsize;
  } else {
    Config.kcubic = fsph::GetKernelCubic(Config.simulate2D ? 2 : 3, float(dp));
    Config.kernelH = Config.kcubic.h;
    Config.kernelSize = Config.kcubic.kernelsize;
  }
  Config.kernelSize2 = Config.kernelSize * Config.kernelSize;

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

  // Pressure constant B = rho0 * cs0^2 / gamma
  Config.cteB = Config.rho0 * Config.cs0 * Config.cs0 / Config.gamma;

  // Eta^2 for viscosity
  Config.eta2 = (0.1f * Config.kernelH) * (0.1f * Config.kernelH);

  // Movement limit
  Config.movLimit = Config.kernelSize * 0.9f;

  // DDT constants
  Config.ddtkh = Config.ddtValue * Config.kernelSize;
  Config.ddtgz = Config.rho0 * Config.gravity.z / Config.cteB;

  // Cell size for neighbor search
  Scell = Config.kernelSize;
  PosCellSize = Config.kernelSize;
}

//==============================================================================
// Cell Division Setup
//==============================================================================
void DsphStepEngine::SetupCellDivision() {
  // Calculate domain with margin
  double margin = Config.kernelSize * 4.0;
  MapPosMin = Config.domainMin - TDouble3(margin);
  MapPosMax = Config.domainMax + TDouble3(margin);
  MapRealPosMin = MapPosMin;
  MapRealSize = MapPosMax - MapPosMin;

  // Calculate cell grid
  MapCells.x = unsigned(ceil((MapPosMax.x - MapPosMin.x) / Scell));
  MapCells.y = unsigned(ceil((MapPosMax.y - MapPosMin.y) / Scell));
  MapCells.z = unsigned(ceil((MapPosMax.z - MapPosMin.z) / Scell));

  // Encode cell code for spatial hashing
  unsigned int bx = 0, by = 0, bz = 0;
  while((1u << bx) < MapCells.x) bx++;
  while((1u << by) < MapCells.y) by++;
  while((1u << bz) < MapCells.z) bz++;
  DomCellCode = DCEL_GetCode(bx, by, bz);

  DomPosMin = MapPosMin;

  // Create cell division object
  if(CellDivSingle) delete CellDivSingle;
  CellDivSingle = new JCellDivGpuSingle(
    true,                           // stable
    false,                          // floating
    0,                              // periactive
    Config.kernelSize2,             // kernelsize2
    PosCellSize,                    // poscellsize
    false,                          // celldomfixed
    CELLMODE_Full,                  // cellmode
    Scell,                          // scell
    MapPosMin, MapPosMax, MapCells, // map definition
    0,                              // casenbound
    0,                              // casenfixed
    Npb,                            // casenpb
    ""                              // dirout
  );
}

//==============================================================================
// Upload Constants to GPU
//==============================================================================
void DsphStepEngine::UploadConstants() {
  StCteInteraction ctes;
  memset(&ctes, 0, sizeof(StCteInteraction));

  // Set mass particle values
  SetCtegMass(ctes, Config.massBound, Config.massFluid);

  // Set distance values
  SetCtegKsize(ctes, Config.kernelH, Config.kernelSize, Config.kernelSize2,
               PosCellSize, Config.eta2, float(Config.dp), Scell, Config.movLimit);

  // Wendland constants (always computed)
  SetCtegKerWendland(ctes, Config.kwend);

  // Cubic constants if using cubic kernel
  if(Config.kernel == KERNEL_Cubic) {
    SetCtegKerCubic(ctes, Config.kcubic);
  }

  // Set density and pressure values
  SetCtegRho(ctes, Config.rho0, 1.0f / Config.rho0, Config.gamma, Config.cs0, Config.cteB);

  // Set DDT values
  SetCtegDdt(ctes, Config.ddtkh, Config.ddtgz);

  // Set boundary options
  SetCtegOpts(ctes, unsigned(Config.boundary));

  // Set periodic (none for now)
  SetCtegPeriodic(ctes, 0, TDouble3(0), TDouble3(0), TDouble3(0));

  // Set map definition
  SetCtegMap(ctes, MapRealPosMin, MapRealSize);

  // Set domain
  SetCtegDomain(ctes, MGDIV_Z, DomCellCode, DomPosMin);

  // Upload to GPU constant memory
  cusph::CteInteractionUp(&ctes);
}

//==============================================================================
// Initialization
//==============================================================================
bool DsphStepEngine::Initialize(const StDsphEngineConfig& config,
                                 const double* fluidPos, const double* fluidVel, unsigned int npf,
                                 const double* boundPos, const double* boundNormals, unsigned int npb,
                                 cudaStream_t stream) {
  if(Initialized) Shutdown();

  try {
    Config = config;
    Npf = npf;
    Npb = npb;
    NpbOk = npb;
    Np = npf + npb;

    if(Np == 0) {
      throw std::runtime_error("No particles provided");
    }

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

    // Setup cell division
    SetupCellDivision();

    // Allocate GPU memory
    AllocateGpuMemory(Np);

    // Upload constants to GPU
    UploadConstants();

    // Prepare host data for upload
    std::vector<unsigned int> idp(Np);
    std::vector<typecode> code(Np);
    std::vector<double2> posxy(Np);
    std::vector<double> posz(Np);
    std::vector<float4> velrho(Np);
    std::vector<unsigned int> dcell(Np);

    unsigned int idx = 0;

    // Boundary particles first (indices 0 to Npb-1)
    for(unsigned int i = 0; i < Npb; i++) {
      idp[idx] = idx;
      code[idx] = CODE_SetType(0, CODE_TYPE_FIXED);  // Fixed boundary

      posxy[idx].x = boundPos[i * 3 + 0];
      posxy[idx].y = boundPos[i * 3 + 1];
      posz[idx] = boundPos[i * 3 + 2];

      velrho[idx].x = 0.0f;
      velrho[idx].y = 0.0f;
      velrho[idx].z = 0.0f;
      velrho[idx].w = Config.rho0;

      // Compute initial cell
      double dx = posxy[idx].x - DomPosMin.x;
      double dy = posxy[idx].y - DomPosMin.y;
      double dz = posz[idx] - DomPosMin.z;
      unsigned cx = unsigned(dx / Scell);
      unsigned cy = unsigned(dy / Scell);
      unsigned cz = unsigned(dz / Scell);
      dcell[idx] = DCEL_Cell(DomCellCode, cx, cy, cz);

      idx++;
    }

    // Fluid particles (indices Npb to Np-1)
    for(unsigned int i = 0; i < Npf; i++) {
      idp[idx] = idx;
      code[idx] = CODE_SetType(0, CODE_TYPE_FLUID);

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

      // Compute initial cell
      double dx = posxy[idx].x - DomPosMin.x;
      double dy = posxy[idx].y - DomPosMin.y;
      double dz = posz[idx] - DomPosMin.z;
      unsigned cx = unsigned(dx / Scell);
      unsigned cy = unsigned(dy / Scell);
      unsigned cz = unsigned(dz / Scell);
      dcell[idx] = DCEL_Cell(DomCellCode, cx, cy, cz);

      idx++;
    }

    // Upload to GPU
    cudaMemcpyAsync(Idpg, idp.data(), Np * sizeof(unsigned int), cudaMemcpyHostToDevice, Stream);
    cudaMemcpyAsync(Codeg, code.data(), Np * sizeof(typecode), cudaMemcpyHostToDevice, Stream);
    cudaMemcpyAsync(Posxyg, posxy.data(), Np * sizeof(double2), cudaMemcpyHostToDevice, Stream);
    cudaMemcpyAsync(Poszg, posz.data(), Np * sizeof(double), cudaMemcpyHostToDevice, Stream);
    cudaMemcpyAsync(Velrhog, velrho.data(), Np * sizeof(float4), cudaMemcpyHostToDevice, Stream);
    cudaMemcpyAsync(Dcellg, dcell.data(), Np * sizeof(unsigned int), cudaMemcpyHostToDevice, Stream);

    // Initialize PosCell
    cudaMemsetAsync(PosCellg, 0, Np * sizeof(float4), Stream);

    // Initialize accelerations
    cudaMemsetAsync(Aceg, 0, Np * sizeof(float3), Stream);

    // Initialize Verlet previous velocities
    if(VelrhoM1g) {
      cudaMemcpyAsync(VelrhoM1g, velrho.data(), Np * sizeof(float4), cudaMemcpyHostToDevice, Stream);
    }

    cudaStreamSynchronize(Stream);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
      throw std::runtime_error(std::string("Initialization failed: ") + cudaGetErrorString(err));
    }

    TimeStep = 0.0;
    StepCount = 0;
    LastDt = 0.0;
    VerletStep = 0;
    ViscDtMax = 0;
    AceMax = 0;

    // Initialize default fluid type (type 0)
    FluidTypeCount = 1;
    FluidTypes[0].rho0 = Config.rho0;
    FluidTypes[0].viscosity = Config.viscoValue;
    FluidTypes[0].surfaceTension = 0.0f;
    FluidTypes[0].mass = Config.massFluid;
    FluidTypes[0].cs0 = Config.cs0;
    FluidTypes[0].cteB = Config.cteB;
    FluidTypes[0].active = true;

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
  DivData = DivDataGpuNull();

  // Run cell division
  CellDivSingle->Divide(Npb, Npf, 0, 0, false,
                        Dcellg, Codeg, Posxyg, Poszg, Idpg, nullptr);
  DivData = CellDivSingle->GetCellDivData();

  // Sort particle arrays
  SortParticleArrays();

  // Update particle count (in case some were excluded)
  Np = CellDivSingle->GetNpFinal();
  Npb = CellDivSingle->GetNpbFinal();
  Npf = Np - Npb;
}

void DsphStepEngine::SortParticleArrays() {
  // Allocate temporary arrays for sorting
  unsigned int* idpTmp = nullptr;
  typecode* codeTmp = nullptr;
  unsigned int* dcellTmp = nullptr;
  double2* posxyTmp = nullptr;
  double* poszTmp = nullptr;
  float4* velrhoTmp = nullptr;

  cudaMalloc(&idpTmp, Np * sizeof(unsigned int));
  cudaMalloc(&codeTmp, Np * sizeof(typecode));
  cudaMalloc(&dcellTmp, Np * sizeof(unsigned int));
  cudaMalloc(&posxyTmp, Np * sizeof(double2));
  cudaMalloc(&poszTmp, Np * sizeof(double));
  cudaMalloc(&velrhoTmp, Np * sizeof(float4));

  // Sort basic arrays
  CellDivSingle->SortBasicArrays(Idpg, Codeg, Dcellg, Posxyg, Poszg, Velrhog,
                                  idpTmp, codeTmp, dcellTmp, posxyTmp, poszTmp, velrhoTmp);

  // Swap pointers
  std::swap(Idpg, idpTmp);
  std::swap(Codeg, codeTmp);
  std::swap(Dcellg, dcellTmp);
  std::swap(Posxyg, posxyTmp);
  std::swap(Poszg, poszTmp);
  std::swap(Velrhog, velrhoTmp);

  // Free temporary arrays
  cudaFree(idpTmp);
  cudaFree(codeTmp);
  cudaFree(dcellTmp);
  cudaFree(posxyTmp);
  cudaFree(poszTmp);
  cudaFree(velrhoTmp);

  // Sort symplectic arrays if active
  if(Config.stepMethod == STEP_Symplectic && PosxyPreg) {
    double2* posxyPreTmp = nullptr;
    double* poszPreTmp = nullptr;
    float4* velrhoPreTmp = nullptr;

    cudaMalloc(&posxyPreTmp, Np * sizeof(double2));
    cudaMalloc(&poszPreTmp, Np * sizeof(double));
    cudaMalloc(&velrhoPreTmp, Np * sizeof(float4));

    CellDivSingle->SortDataArrays(PosxyPreg, PoszPreg, VelrhoPreg,
                                   posxyPreTmp, poszPreTmp, velrhoPreTmp);

    std::swap(PosxyPreg, posxyPreTmp);
    std::swap(PoszPreg, poszPreTmp);
    std::swap(VelrhoPreg, velrhoPreTmp);

    cudaFree(posxyPreTmp);
    cudaFree(poszPreTmp);
    cudaFree(velrhoPreTmp);
  }

  // Sort Verlet arrays if active
  if(Config.stepMethod == STEP_Verlet && VelrhoM1g) {
    float4* velrhoM1Tmp = nullptr;
    cudaMalloc(&velrhoM1Tmp, Np * sizeof(float4));
    CellDivSingle->SortDataArrays(VelrhoM1g, velrhoM1Tmp);
    std::swap(VelrhoM1g, velrhoM1Tmp);
    cudaFree(velrhoM1Tmp);
  }
}

//==============================================================================
// Pre/Post Interaction
//==============================================================================
void DsphStepEngine::PreInteraction_Forces() {
  // Initialize viscDt and Ar
  cudaMemsetAsync(ViscDtg, 0, Np * sizeof(float), Stream);
  cudaMemsetAsync(Arg, 0, Np * sizeof(float), Stream);
  if(Deltag) {
    cudaMemsetAsync(Deltag, 0, Np * sizeof(float), Stream);
  }

  // Initialize accelerations (gravity for fluid, zero for boundary)
  cusphs::InitAceGravity(Np, Npb, Config.gravity, Aceg, Stream);

  // Update PosCell
  cusphs::UpdatePosCell(Np, DomPosMin, PosCellSize, Posxyg, Poszg, PosCellg, Stream);
}

void DsphStepEngine::PosInteraction_Forces() {
  // For 2D simulations, zero the Y component of acceleration
  if(Config.simulate2D) {
    cusphs::Resety(Np - Npb, Npb, Aceg, Stream);
  }

  // Add Delta-SPH correction to Ar
  if(Deltag) {
    cusph::AddDelta(Np - Npb, Deltag + Npb, Arg + Npb, Stream);
  }

  cudaStreamSynchronize(Stream);

  // Compute max viscDt
  ViscDtMax = ComputeViscDtMax();

  // Compute max acceleration
  AceMax = ComputeAceMax();
}

//==============================================================================
// Force Computation
//==============================================================================
void DsphStepEngine::Interaction_Forces() {
  // Build interaction parameters
  const StInterParmsg parms = StrInterParmsg(
    Config.simulate2D,
    Config.kernel,
    FTMODE_None,                    // No floating bodies
    Config.visco,
    Config.density,
    SHIFT_None,                     // No shifting
    MDBC2_None,                     // No mDBC2
    false, false, false, false,     // shiftadv, corrector, aleform, ncpress
    Config.viscoValue * Config.viscoBoundFactor,  // viscob
    Config.viscoValue,              // viscof
    256, 256,                       // bsbound, bsfluid
    Np, Npb, NpbOk,                 // particle counts
    0, StepCount,                   // id, nstep
    DivData,                        // cell division data
    Dcellg,                         // dcell
    Posxyg, Poszg, PosCellg,        // positions
    Velrhog, Idpg, Codeg,           // velocities, ids, codes
    nullptr, nullptr, nullptr,      // boundmode, tangenvel, motionvel (mDBC)
    nullptr, nullptr,               // boundnormal, nopenshift (mDBC)
    nullptr,                        // ftomassp (floating)
    nullptr,                        // spstaurho2 (SPS)
    nullptr,                        // dengradcorr
    ViscDtg, Arg, Aceg,             // output: viscdt, ar, ace
    Deltag,                         // delta (DDT)
    nullptr,                        // sps2strain
    ShiftPosfsg,                    // shiftposfs
    nullptr, nullptr,               // fstype, shiftvel
    nullptr, nullptr, nullptr,      // psiclean arrays
    0.0f, false,                    // divcleankp, divclean
    Stream,                         // CUDA stream
    nullptr                         // kerinfo
  );

  cusph::Interaction_Forces(parms);
}

float DsphStepEngine::ComputeViscDtMax() {
  if(Np == 0) return 0;
  return cusph::ReduMaxFloat(Np, 0, ViscDtg, AuxMemg);
}

float DsphStepEngine::ComputeAceMax() {
  if(Np == 0) return 0;
  // Compute acceleration magnitude and find max
  cusph::ComputeAceMod(Np, Codeg, Aceg, ViscDtg);
  return cusph::ReduMaxFloat(Np, 0, ViscDtg, AuxMemg);
}

//==============================================================================
// Timestep Computation
//==============================================================================
double DsphStepEngine::ComputeDtVariable() {
  double dt = 1e10;

  // CFL condition: dt <= CFL * h / (cs + vmax)
  if(AceMax > 0) {
    double dtf = Config.cfl * sqrt(Config.kernelH / AceMax);
    dt = std::min(dt, dtf);
  }

  // Viscous condition
  if(ViscDtMax > 0) {
    double dtcv = Config.cfl * Config.kernelH / (Config.cs0 + ViscDtMax);
    dt = std::min(dt, dtcv);
  }

  // Minimum timestep based on CFL
  double dtMin = Config.cfl * Config.kernelH / (Config.cs0 + 10.0);
  dt = std::max(dt, dtMin);

  return dt;
}

double DsphStepEngine::ComputeTimeStep() {
  if(!Initialized) return 0.001;
  return ComputeDtVariable();
}

//==============================================================================
// Position/Velocity Update
//==============================================================================
void DsphStepEngine::ComputeStepVerlet(double dt) {
  double dt2 = dt * 2.0;

  // Update velocities and compute movement
  cusphs::ComputeStepVerlet(
    false, false, false,            // floating, shift, inout
    MDBC2_None,                     // mdbc2
    Np, Npb,
    VelrhoM1g,                      // velrho1 (t-dt)
    Velrhog,                        // velrho2 (t)
    nullptr,                        // boundmode
    Arg, Aceg,                      // ar, ace
    nullptr,                        // shiftposfs
    nullptr,                        // indirvel
    nullptr,                        // nopenshift
    dt, dt2,
    Config.rho0,
    Config.rho0 * 0.7f,             // rhopoutmin
    Config.rho0 * 1.3f,             // rhopoutmax
    Config.gravity,
    Codeg, Movxyg, Movzg,
    Velrhog,                        // output: velrhonew
    Stream
  );

  // Every 40 steps, reset Verlet by copying current to previous
  VerletStep++;
  if(VerletStep >= 40) {
    cudaMemcpyAsync(VelrhoM1g, Velrhog, Np * sizeof(float4), cudaMemcpyDeviceToDevice, Stream);
    VerletStep = 0;
  } else {
    // Swap VelrhoM1 with old Velrho would be done here
    // For simplicity, just copy
    cudaMemcpyAsync(VelrhoM1g, Velrhog, Np * sizeof(float4), cudaMemcpyDeviceToDevice, Stream);
  }

  TimeStep += dt;
  StepCount++;
  LastDt = dt;
}

void DsphStepEngine::ComputeStepSymplectic(double dt) {
  double dtm = dt * 0.5;

  // Save current state
  cudaMemcpyAsync(PosxyPreg, Posxyg, Np * sizeof(double2), cudaMemcpyDeviceToDevice, Stream);
  cudaMemcpyAsync(PoszPreg, Poszg, Np * sizeof(double), cudaMemcpyDeviceToDevice, Stream);
  cudaMemcpyAsync(VelrhoPreg, Velrhog, Np * sizeof(float4), cudaMemcpyDeviceToDevice, Stream);

  // Predictor step
  cusphs::ComputeStepSymplecticPre(
    false, false, false,            // floating, shift, inout
    MDBC2_None,
    Np, Npb,
    VelrhoPreg,                     // velrhopre
    nullptr,                        // boundmode
    Arg, Aceg,                      // ar, ace
    nullptr,                        // shiftposfs
    nullptr,                        // indirvel
    dtm,
    Config.rho0,
    Config.rho0 * 0.7f,             // rhopoutmin
    Config.rho0 * 1.3f,             // rhopoutmax
    Config.gravity,
    Codeg, Movxyg, Movzg, Velrhog,
    nullptr, nullptr, nullptr, false,  // psiclean arrays
    Stream
  );

  // Update positions
  ComputeStepPos(dt);

  // Cell division at predictor position
  RunCellDivide(false);

  // Recompute forces at predictor position
  PreInteraction_Forces();
  Interaction_Forces();
  PosInteraction_Forces();

  // Corrector step
  cusphs::ComputeStepSymplecticCor(
    false, false, false, false,     // floating, shift, shiftadv, inout
    MDBC2_None,
    Np, Npb,
    VelrhoPreg,                     // velrhopre
    nullptr,                        // boundmode
    Arg, Aceg,                      // ar, ace
    nullptr,                        // shiftposfs
    nullptr,                        // indirvel
    nullptr,                        // nopenshift
    nullptr,                        // shiftvel
    dtm, dt,
    Config.rho0,
    Config.rho0 * 0.7f,             // rhopoutmin
    Config.rho0 * 1.3f,             // rhopoutmax
    Config.gravity,
    Codeg, Movxyg, Movzg, Velrhog,
    nullptr, nullptr, nullptr, false,  // psiclean arrays
    Stream
  );

  TimeStep += dt;
  StepCount++;
  LastDt = dt;
}

void DsphStepEngine::ComputeStepPos(double dt) {
  // Update positions based on computed movement
  cusph::ComputeStepPos(0, false, Np, Npb, Movxyg, Movzg, Posxyg, Poszg, Dcellg, Codeg);
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

    // 2. Pre-interaction setup
    PreInteraction_Forces();

    // 3. Compute SPH forces
    Interaction_Forces();

    // 4. Post-interaction (reductions, cleanup)
    PosInteraction_Forces();

    // 4b. Accumulate forces on dynamic boundary objects
    if(BoundaryCount > 0) {
      AccumulateBoundaryForces();
    }

    // 5. Time integration
    if(Config.stepMethod == STEP_Verlet) {
      ComputeStepVerlet(dt);
    } else {
      ComputeStepSymplectic(dt);
    }

    // 6. Update positions
    ComputeStepPos(dt);

    // 7. Copy to external buffer if configured
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

void DsphStepEngine::GetAccelerations(float* outAccelerations, unsigned int count) {
  if(!Initialized || !outAccelerations || count == 0) return;
  if(count > Npf) count = Npf;

  // Copy accelerations from fluid particles (offset by Npb for boundary particles)
  std::vector<float3> ace(count);
  cudaMemcpyAsync(ace.data(), Aceg + Npb, count * sizeof(float3), cudaMemcpyDeviceToHost, Stream);
  cudaStreamSynchronize(Stream);

  // Convert float3 array to interleaved xyz
  for(unsigned int i = 0; i < count; i++) {
    outAccelerations[i * 3 + 0] = ace[i].x;
    outAccelerations[i * 3 + 1] = ace[i].y;
    outAccelerations[i * 3 + 2] = ace[i].z;
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
  ViscDtMax = 0;
  AceMax = 0;
}

void DsphStepEngine::Shutdown() {
  FreeGpuMemory();

  if(OwnsStream && Stream) {
    cudaStreamDestroy(Stream);
  }
  Stream = nullptr;
  OwnsStream = false;

  Initialized = false;
  Np = Npb = NpbOk = Npf = 0;
  TimeStep = 0.0;
  StepCount = 0;
  DivData = DivDataGpuNull();
  BoundaryCount = 0;
}

//==============================================================================
// Dynamic Boundary Object Methods
//==============================================================================

int DsphStepEngine::AddBoundaryObject(
    const float* localPositions,
    const float* localNormals,
    unsigned int particleCount,
    float mass,
    const float* inertia,
    const float* centerOfMass,
    bool isDynamic)
{
  if(!Initialized) return -1;
  if(BoundaryCount >= DSPH_MAX_BOUNDARIES) return -1;
  if(particleCount == 0 || !localPositions || !centerOfMass) return -1;

  // Allocate force/torque arrays on first boundary
  if(BoundaryCount == 0) {
    cudaMalloc(&BoundForcesg, DSPH_MAX_BOUNDARIES * sizeof(float3));
    cudaMalloc(&BoundTorquesg, DSPH_MAX_BOUNDARIES * sizeof(float3));
    if(!BoundForcesg || !BoundTorquesg) {
      if(BoundForcesg) { cudaFree(BoundForcesg); BoundForcesg = nullptr; }
      if(BoundTorquesg) { cudaFree(BoundTorquesg); BoundTorquesg = nullptr; }
      return -1;
    }
    // Zero the arrays
    cudaMemsetAsync(BoundForcesg, 0, DSPH_MAX_BOUNDARIES * sizeof(float3), Stream);
    cudaMemsetAsync(BoundTorquesg, 0, DSPH_MAX_BOUNDARIES * sizeof(float3), Stream);
  }

  unsigned int boundaryId = BoundaryCount;

  // Calculate total boundary particles needed (sum of all boundary objects + new one)
  unsigned int totalBoundParticles = particleCount;
  for(unsigned int i = 0; i < BoundaryCount; i++) {
    totalBoundParticles += BoundaryObjects[i].particleCount;
  }

  // Determine particle start index
  unsigned int particleStart = 0;
  for(unsigned int i = 0; i < BoundaryCount; i++) {
    particleStart += BoundaryObjects[i].particleCount;
  }

  // Reallocate boundary arrays if needed
  float3* newLocalPos = nullptr;
  float3* newLocalNorm = nullptr;
  float3* newWorldPos = nullptr;
  float3* newWorldNorm = nullptr;
  float3* newVel = nullptr;

  cudaMalloc(&newLocalPos, totalBoundParticles * sizeof(float3));
  cudaMalloc(&newLocalNorm, totalBoundParticles * sizeof(float3));
  cudaMalloc(&newWorldPos, totalBoundParticles * sizeof(float3));
  cudaMalloc(&newWorldNorm, totalBoundParticles * sizeof(float3));
  cudaMalloc(&newVel, totalBoundParticles * sizeof(float3));

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    if(newLocalPos) cudaFree(newLocalPos);
    if(newLocalNorm) cudaFree(newLocalNorm);
    if(newWorldPos) cudaFree(newWorldPos);
    if(newWorldNorm) cudaFree(newWorldNorm);
    if(newVel) cudaFree(newVel);
    return -1;
  }

  // Copy existing data if any
  if(BoundLocalPosg && particleStart > 0) {
    cudaMemcpyAsync(newLocalPos, BoundLocalPosg, particleStart * sizeof(float3), cudaMemcpyDeviceToDevice, Stream);
    cudaMemcpyAsync(newLocalNorm, BoundLocalNormg, particleStart * sizeof(float3), cudaMemcpyDeviceToDevice, Stream);
    cudaMemcpyAsync(newWorldPos, BoundWorldPosg, particleStart * sizeof(float3), cudaMemcpyDeviceToDevice, Stream);
    cudaMemcpyAsync(newWorldNorm, BoundWorldNormg, particleStart * sizeof(float3), cudaMemcpyDeviceToDevice, Stream);
    cudaMemcpyAsync(newVel, BoundVelg, particleStart * sizeof(float3), cudaMemcpyDeviceToDevice, Stream);
  }

  // Convert and upload new boundary particle data
  std::vector<float3> localPosData(particleCount);
  std::vector<float3> localNormData(particleCount);

  for(unsigned int i = 0; i < particleCount; i++) {
    localPosData[i] = make_float3(
      localPositions[i * 3 + 0],
      localPositions[i * 3 + 1],
      localPositions[i * 3 + 2]
    );

    if(localNormals) {
      localNormData[i] = make_float3(
        localNormals[i * 3 + 0],
        localNormals[i * 3 + 1],
        localNormals[i * 3 + 2]
      );
    } else {
      localNormData[i] = make_float3(0, 0, 1);  // Default normal
    }
  }

  cudaMemcpyAsync(newLocalPos + particleStart, localPosData.data(),
                  particleCount * sizeof(float3), cudaMemcpyHostToDevice, Stream);
  cudaMemcpyAsync(newLocalNorm + particleStart, localNormData.data(),
                  particleCount * sizeof(float3), cudaMemcpyHostToDevice, Stream);

  // Free old arrays and assign new ones
  if(BoundLocalPosg) cudaFree(BoundLocalPosg);
  if(BoundLocalNormg) cudaFree(BoundLocalNormg);
  if(BoundWorldPosg) cudaFree(BoundWorldPosg);
  if(BoundWorldNormg) cudaFree(BoundWorldNormg);
  if(BoundVelg) cudaFree(BoundVelg);

  BoundLocalPosg = newLocalPos;
  BoundLocalNormg = newLocalNorm;
  BoundWorldPosg = newWorldPos;
  BoundWorldNormg = newWorldNorm;
  BoundVelg = newVel;

  // Initialize boundary object data
  StDsphBoundaryObject& obj = BoundaryObjects[boundaryId];
  obj.particleStart = particleStart;
  obj.particleCount = particleCount;
  obj.mass = mass;
  if(inertia) {
    for(int i = 0; i < 6; i++) obj.inertia[i] = inertia[i];
  }
  obj.position = make_float3(centerOfMass[0], centerOfMass[1], centerOfMass[2]);
  obj.velocity = make_float3(0, 0, 0);
  obj.orientation = make_float4(0, 0, 0, 1);  // Identity quaternion
  obj.angularVelocity = make_float3(0, 0, 0);
  obj.accumulatedForce = make_float3(0, 0, 0);
  obj.accumulatedTorque = make_float3(0, 0, 0);
  obj.isDynamic = isDynamic;
  obj.isActive = true;

  BoundaryCount++;

  // Transform particles to initial world positions
  TransformBoundaryParticles(boundaryId);

  cudaStreamSynchronize(Stream);

  return (int)boundaryId;
}

void DsphStepEngine::TransformBoundaryParticles(unsigned int boundaryId) {
  if(boundaryId >= BoundaryCount) return;

  const StDsphBoundaryObject& obj = BoundaryObjects[boundaryId];
  if(!obj.isActive) return;

  dsphker::TransformBoundaryParticles(
    obj.particleCount,
    BoundLocalPosg + obj.particleStart,
    BoundLocalNormg + obj.particleStart,
    obj.position,
    obj.orientation,
    obj.velocity,
    obj.angularVelocity,
    BoundWorldPosg + obj.particleStart,
    BoundWorldNormg + obj.particleStart,
    BoundVelg + obj.particleStart,
    Stream
  );
}

void DsphStepEngine::AccumulateBoundaryForces() {
  if(BoundaryCount == 0 || !BoundForcesg || !BoundTorquesg) return;
  if(!CellDivSingle) return;

  // Get cell division data for neighbor search (int2: .x=begin, .y=end)
  const int2* beginEndCell = CellDivSingle->GetBeginCell();
  if(!beginEndCell) return;

  // Zero force/torque accumulators for all boundaries
  dsphker::ZeroFloat3Array(BoundForcesg, BoundaryCount, Stream);
  dsphker::ZeroFloat3Array(BoundTorquesg, BoundaryCount, Stream);

  // For each boundary, compute fluid forces using SPH interaction
  for(unsigned int i = 0; i < BoundaryCount; i++) {
    StDsphBoundaryObject& obj = BoundaryObjects[i];
    if(!obj.isActive) continue;

    // Compute forces from fluid onto this boundary
    dsphker::ComputeBoundaryFluidForces(
      obj.particleCount,
      BoundWorldPosg + obj.particleStart,
      BoundWorldNormg + obj.particleStart,
      obj.position,
      Np, Npb,
      Posxyg, Poszg, Velrhog,
      beginEndCell,
      DivData.cellfluid,
      DomCellCode,
      make_double3(DomPosMin.x, DomPosMin.y, DomPosMin.z),
      Scell,
      Config.kernelH,
      Config.kernelSize,
      Config.massFluid,
      Config.massBound,
      Config.rho0,
      Config.cs0,
      Config.gamma,
      BoundForcesg + i,
      BoundTorquesg + i,
      Stream
    );
  }

  // Copy results back to CPU (host) for each boundary
  cudaStreamSynchronize(Stream);

  std::vector<float3> forces(BoundaryCount);
  std::vector<float3> torques(BoundaryCount);
  cudaMemcpy(forces.data(), BoundForcesg, BoundaryCount * sizeof(float3), cudaMemcpyDeviceToHost);
  cudaMemcpy(torques.data(), BoundTorquesg, BoundaryCount * sizeof(float3), cudaMemcpyDeviceToHost);

  for(unsigned int i = 0; i < BoundaryCount; i++) {
    BoundaryObjects[i].accumulatedForce.x += forces[i].x;
    BoundaryObjects[i].accumulatedForce.y += forces[i].y;
    BoundaryObjects[i].accumulatedForce.z += forces[i].z;
    BoundaryObjects[i].accumulatedTorque.x += torques[i].x;
    BoundaryObjects[i].accumulatedTorque.y += torques[i].y;
    BoundaryObjects[i].accumulatedTorque.z += torques[i].z;
  }
}

bool DsphStepEngine::UpdateBoundaryState(
    unsigned int boundaryId,
    const float* position,
    const float* velocity,
    const float* orientation,
    const float* angularVelocity)
{
  if(boundaryId >= BoundaryCount) return false;

  StDsphBoundaryObject& obj = BoundaryObjects[boundaryId];
  if(!obj.isActive) return false;

  if(position) {
    obj.position = make_float3(position[0], position[1], position[2]);
  }
  if(velocity) {
    obj.velocity = make_float3(velocity[0], velocity[1], velocity[2]);
  }
  if(orientation) {
    obj.orientation = make_float4(orientation[0], orientation[1], orientation[2], orientation[3]);
  }
  if(angularVelocity) {
    obj.angularVelocity = make_float3(angularVelocity[0], angularVelocity[1], angularVelocity[2]);
  }

  // Re-transform particles to new world positions
  TransformBoundaryParticles(boundaryId);

  return true;
}

bool DsphStepEngine::GetBoundaryForces(
    unsigned int boundaryId,
    float* outForce,
    float* outTorque,
    bool clearAfterRead)
{
  if(boundaryId >= BoundaryCount) return false;

  StDsphBoundaryObject& obj = BoundaryObjects[boundaryId];
  if(!obj.isActive) return false;

  if(outForce) {
    outForce[0] = obj.accumulatedForce.x;
    outForce[1] = obj.accumulatedForce.y;
    outForce[2] = obj.accumulatedForce.z;
  }
  if(outTorque) {
    outTorque[0] = obj.accumulatedTorque.x;
    outTorque[1] = obj.accumulatedTorque.y;
    outTorque[2] = obj.accumulatedTorque.z;
  }

  if(clearAfterRead) {
    obj.accumulatedForce = make_float3(0, 0, 0);
    obj.accumulatedTorque = make_float3(0, 0, 0);
  }

  return true;
}

void DsphStepEngine::ClearBoundaryForces(unsigned int boundaryId) {
  if(boundaryId >= BoundaryCount) return;

  StDsphBoundaryObject& obj = BoundaryObjects[boundaryId];
  obj.accumulatedForce = make_float3(0, 0, 0);
  obj.accumulatedTorque = make_float3(0, 0, 0);
}

bool DsphStepEngine::GetBoundaryState(
    unsigned int boundaryId,
    float* outPosition,
    float* outVelocity,
    float* outOrientation,
    float* outAngularVelocity)
{
  if(boundaryId >= BoundaryCount) return false;

  const StDsphBoundaryObject& obj = BoundaryObjects[boundaryId];
  if(!obj.isActive) return false;

  if(outPosition) {
    outPosition[0] = obj.position.x;
    outPosition[1] = obj.position.y;
    outPosition[2] = obj.position.z;
  }
  if(outVelocity) {
    outVelocity[0] = obj.velocity.x;
    outVelocity[1] = obj.velocity.y;
    outVelocity[2] = obj.velocity.z;
  }
  if(outOrientation) {
    outOrientation[0] = obj.orientation.x;
    outOrientation[1] = obj.orientation.y;
    outOrientation[2] = obj.orientation.z;
    outOrientation[3] = obj.orientation.w;
  }
  if(outAngularVelocity) {
    outAngularVelocity[0] = obj.angularVelocity.x;
    outAngularVelocity[1] = obj.angularVelocity.y;
    outAngularVelocity[2] = obj.angularVelocity.z;
  }

  return true;
}

//==============================================================================
// Multiple Fluid Type Methods
//==============================================================================

int DsphStepEngine::CreateFluidType(float rho0, float viscosity, float surfaceTension) {
  if(FluidTypeCount >= DSPH_MAX_FLUID_TYPES) return -1;

  unsigned int typeId = FluidTypeCount;
  StDsphFluidType& ft = FluidTypes[typeId];

  ft.rho0 = rho0;
  ft.viscosity = viscosity;
  ft.surfaceTension = surfaceTension;
  ft.active = true;

  // Compute derived properties using same formulas as main config
  double volume = Config.dp * Config.dp * Config.dp;
  ft.mass = float(rho0 * volume);

  // Speed of sound (proportional to reference density ratio)
  ft.cs0 = Config.cs0 * sqrt(rho0 / Config.rho0);

  // Pressure constant B = rho0 * cs0^2 / gamma
  ft.cteB = rho0 * ft.cs0 * ft.cs0 / Config.gamma;

  FluidTypeCount++;
  return (int)typeId;
}

bool DsphStepEngine::SetFluidTypeDensity(unsigned int fluidTypeId, float rho0) {
  if(fluidTypeId >= FluidTypeCount) return false;
  if(!FluidTypes[fluidTypeId].active) return false;

  StDsphFluidType& ft = FluidTypes[fluidTypeId];
  ft.rho0 = rho0;

  // Recompute derived properties
  double volume = Config.dp * Config.dp * Config.dp;
  ft.mass = float(rho0 * volume);
  ft.cs0 = Config.cs0 * sqrt(rho0 / Config.rho0);
  ft.cteB = rho0 * ft.cs0 * ft.cs0 / Config.gamma;

  return true;
}

bool DsphStepEngine::SetFluidTypeViscosity(unsigned int fluidTypeId, float viscosity) {
  if(fluidTypeId >= FluidTypeCount) return false;
  if(!FluidTypes[fluidTypeId].active) return false;

  FluidTypes[fluidTypeId].viscosity = viscosity;
  return true;
}

bool DsphStepEngine::GetFluidTypeProperties(unsigned int fluidTypeId, float* outRho0,
                                            float* outViscosity, float* outSurfaceTension) {
  if(fluidTypeId >= FluidTypeCount) return false;
  if(!FluidTypes[fluidTypeId].active) return false;

  const StDsphFluidType& ft = FluidTypes[fluidTypeId];
  if(outRho0) *outRho0 = ft.rho0;
  if(outViscosity) *outViscosity = ft.viscosity;
  if(outSurfaceTension) *outSurfaceTension = ft.surfaceTension;

  return true;
}

bool DsphStepEngine::AddFluidParticlesTyped(
    const double* positions,
    const double* velocities,
    unsigned int count,
    unsigned int fluidTypeId)
{
  // This method is for adding particles before initialization
  // For now, we don't support dynamic particle addition
  // Return false to indicate this requires re-initialization
  if(Initialized) return false;
  if(fluidTypeId >= DSPH_MAX_FLUID_TYPES) return false;

  // Placeholder - full implementation would need to track pending particles
  // and incorporate them during Initialize()
  return false;
}

bool DsphStepEngine::SetParticleFluidType(unsigned int particleIndex, unsigned int fluidTypeId) {
  if(!Initialized) return false;
  if(particleIndex >= Npf) return false;
  if(fluidTypeId >= FluidTypeCount) return false;
  if(!FluidTypes[fluidTypeId].active) return false;

  // Update fluid type for this particle on GPU
  unsigned char typeVal = (unsigned char)fluidTypeId;

  // FluidTypeg array is for all particles, fluid particles start at Npb
  cudaMemcpyAsync(FluidTypeg + Npb + particleIndex, &typeVal, sizeof(unsigned char),
                  cudaMemcpyHostToDevice, Stream);

  return true;
}

unsigned int DsphStepEngine::GetParticleFluidType(unsigned int particleIndex) {
  if(!Initialized || particleIndex >= Npf) return 0;

  unsigned char typeVal = 0;
  cudaMemcpy(&typeVal, FluidTypeg + Npb + particleIndex, sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  return (unsigned int)typeVal;
}

#endif // _WITHGPU
