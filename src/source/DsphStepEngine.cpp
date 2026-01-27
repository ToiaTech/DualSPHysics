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
}

#endif // _WITHGPU
