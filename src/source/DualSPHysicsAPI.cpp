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

/// \file DualSPHysicsAPI.cpp \brief Implementation of C-compatible API for DualSPHysics DLL.

#include "DualSPHysicsAPI.h"
#include "JAppInfo.h"
#include "JLog2.h"
#include "JException.h"
#include "JSphCfgRun.h"
#include "JSph.h"
#include "DualSphDef.h"
#include "Functions.h"
#include "FunSphKernelsCfg.h"

#ifdef _WITHGPU
  #include "JSphGpuSingle.h"
  #include "FunctionsCuda.h"
  #include <cuda_runtime.h>
#endif

#include <string>
#include <cstring>
#include <mutex>
#include <vector>
#include <cmath>

//==============================================================================
// Internal state
//==============================================================================
namespace {
  bool g_Initialized = false;
  std::string g_LastError;
  std::mutex g_Mutex;
  std::string g_VersionStr;
  std::string g_FullNameStr;
  std::string g_FeaturesStr;

  void SetError(const std::string& error) {
    std::lock_guard<std::mutex> lock(g_Mutex);
    g_LastError = error;
  }

  void ClearErrorInternal() {
    std::lock_guard<std::mutex> lock(g_Mutex);
    g_LastError.clear();
  }
}

//==============================================================================
// Simulation Configuration Structure
//==============================================================================
struct DsphSimConfig {
  // Domain
  double domainMinX, domainMinY, domainMinZ;
  double domainMaxX, domainMaxY, domainMaxZ;
  bool domainSet;

  // Basic parameters
  double dp;                    // Particle spacing
  double gravity[3];            // Gravity vector
  int kernelType;               // DSPH_KERNEL_*
  int viscoType;                // DSPH_VISCO_*
  double viscoValue;            // Viscosity coefficient
  double viscoBoundFactor;      // Boundary viscosity factor
  int stepMethod;               // DSPH_STEP_*
  double cfl;                   // CFL number
  int boundaryMethod;           // DSPH_BOUNDARY_*
  int ddtType;                  // DSPH_DDT_*
  double ddtValue;              // DDT coefficient
  double rho0;                  // Reference density
  double speedOfSound;          // Speed of sound (0 = auto)
  bool simulate2D;              // 2D mode
  double simulate2DPosY;        // Y position for 2D

  DsphSimConfig() { Reset(); }

  void Reset() {
    domainMinX = domainMinY = domainMinZ = 0.0;
    domainMaxX = domainMaxY = domainMaxZ = 1.0;
    domainSet = false;

    dp = 0.01;
    gravity[0] = 0.0; gravity[1] = 0.0; gravity[2] = -9.81;
    kernelType = DSPH_KERNEL_WENDLAND;
    viscoType = DSPH_VISCO_ARTIFICIAL;
    viscoValue = 0.01;
    viscoBoundFactor = 1.0;
    stepMethod = DSPH_STEP_SYMPLECTIC;
    cfl = 0.2;
    boundaryMethod = DSPH_BOUNDARY_DBC;
    ddtType = DSPH_DDT_FOURTAKAS;
    ddtValue = 0.1;
    rho0 = 1000.0;
    speedOfSound = 0.0;  // Auto-calculate
    simulate2D = false;
    simulate2DPosY = 0.0;
  }
};

//==============================================================================
// Particle Storage Structure (pre-preparation)
//==============================================================================
struct DsphParticleStorage {
  std::vector<double> fluidPositions;     // x,y,z,x,y,z,...
  std::vector<double> fluidVelocities;    // vx,vy,vz,...
  std::vector<double> boundaryPositions;  // x,y,z,...
  std::vector<double> boundaryNormals;    // nx,ny,nz,...

  unsigned int FluidCount() const { return static_cast<unsigned int>(fluidPositions.size() / 3); }
  unsigned int BoundaryCount() const { return static_cast<unsigned int>(boundaryPositions.size() / 3); }

  void Clear() {
    fluidPositions.clear();
    fluidVelocities.clear();
    boundaryPositions.clear();
    boundaryNormals.clear();
  }
};

//==============================================================================
// External Buffer Configuration
//==============================================================================
struct DsphExternalBuffer {
  void* cudaDevicePtr;
  unsigned int maxParticles;
  bool writeFluidOnly;
  bool enabled;

  DsphExternalBuffer() : cudaDevicePtr(nullptr), maxParticles(0),
                         writeFluidOnly(true), enabled(false) {}
};

//==============================================================================
// Simulation Handle Structure
//==============================================================================
struct DsphSimulation_ {
  int deviceType;
  int gpuId;

  // Configuration (pre-preparation)
  DsphSimConfig config;
  DsphParticleStorage particles;
  DsphExternalBuffer externalBuffer;

  // Runtime state
  bool prepared;
  double simulationTime;
  unsigned int stepCount;

  // CUDA stream (GPU only)
  void* cudaStream;
  bool ownsStream;

  // Internal DualSPHysics objects (created on Prepare)
  // These will be managed by the actual simulation engine
  // For now, we store the computed SPH constants
  float kernelH;
  float kernelSize;
  float massFluid;
  float massBound;
  float cs0;  // Speed of sound

  // GPU particle arrays (pointers to GPU memory)
#ifdef _WITHGPU
  double* gpu_posX;
  double* gpu_posY;
  double* gpu_posZ;
  float* gpu_velX;
  float* gpu_velY;
  float* gpu_velZ;
  float* gpu_rho;
  unsigned int* gpu_idp;
  unsigned int allocatedParticles;
#endif

  unsigned int totalParticles;
  unsigned int fluidParticles;
  unsigned int boundaryParticles;

  DsphSimulation_() :
    deviceType(DSPH_DEVICE_CPU),
    gpuId(0),
    prepared(false),
    simulationTime(0.0),
    stepCount(0),
    cudaStream(nullptr),
    ownsStream(false),
    kernelH(0),
    kernelSize(0),
    massFluid(0),
    massBound(0),
    cs0(0),
#ifdef _WITHGPU
    gpu_posX(nullptr),
    gpu_posY(nullptr),
    gpu_posZ(nullptr),
    gpu_velX(nullptr),
    gpu_velY(nullptr),
    gpu_velZ(nullptr),
    gpu_rho(nullptr),
    gpu_idp(nullptr),
    allocatedParticles(0),
#endif
    totalParticles(0),
    fluidParticles(0),
    boundaryParticles(0)
  {}

  ~DsphSimulation_() {
    FreeGpuMemory();
    if(ownsStream && cudaStream) {
#ifdef _WITHGPU
      cudaStreamDestroy(static_cast<cudaStream_t>(cudaStream));
#endif
    }
  }

  void FreeGpuMemory() {
#ifdef _WITHGPU
    if(gpu_posX) { cudaFree(gpu_posX); gpu_posX = nullptr; }
    if(gpu_posY) { cudaFree(gpu_posY); gpu_posY = nullptr; }
    if(gpu_posZ) { cudaFree(gpu_posZ); gpu_posZ = nullptr; }
    if(gpu_velX) { cudaFree(gpu_velX); gpu_velX = nullptr; }
    if(gpu_velY) { cudaFree(gpu_velY); gpu_velY = nullptr; }
    if(gpu_velZ) { cudaFree(gpu_velZ); gpu_velZ = nullptr; }
    if(gpu_rho) { cudaFree(gpu_rho); gpu_rho = nullptr; }
    if(gpu_idp) { cudaFree(gpu_idp); gpu_idp = nullptr; }
    allocatedParticles = 0;
#endif
  }
};

//==============================================================================
// Version Information
//==============================================================================
DUALSPH_CAPI const char* DsphGetVersion(void) {
  if(g_VersionStr.empty()) {
    g_VersionStr = "5.4.355";
  }
  return g_VersionStr.c_str();
}

DUALSPH_CAPI const char* DsphGetFullName(void) {
  if(g_FullNameStr.empty()) {
    g_FullNameStr = "DualSPHysics5 v5.4.355";
  }
  return g_FullNameStr.c_str();
}

DUALSPH_CAPI int DsphHasGpuSupport(void) {
#ifdef _WITHGPU
  return 1;
#else
  return 0;
#endif
}

DUALSPH_CAPI const char* DsphGetFeatures(void) {
  if(g_FeaturesStr.empty()) {
    g_FeaturesStr = JSph::GetFeatureList();
  }
  return g_FeaturesStr.c_str();
}

//==============================================================================
// Library Initialization
//==============================================================================
DUALSPH_CAPI int DsphInitialize(void) {
  std::lock_guard<std::mutex> lock(g_Mutex);

  if(g_Initialized) {
    g_LastError = "Library already initialized";
    return DSPH_ERROR_ALREADY_INIT;
  }

  try {
    g_Initialized = true;
    g_LastError.clear();
    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    g_LastError = std::string("Initialization failed: ") + e.what();
    return DSPH_ERROR_UNKNOWN;
  }
}

DUALSPH_CAPI int DsphShutdown(void) {
  std::lock_guard<std::mutex> lock(g_Mutex);

  if(!g_Initialized) {
    g_LastError = "Library not initialized";
    return DSPH_ERROR_NOT_INIT;
  }

  g_Initialized = false;
  g_LastError.clear();
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphIsInitialized(void) {
  return g_Initialized ? 1 : 0;
}

//==============================================================================
// GPU Information
//==============================================================================
DUALSPH_CAPI int DsphGetGpuCount(void) {
#ifdef _WITHGPU
  try {
    return fcuda::GetCudaDevicesCount();
  }
  catch(...) {
    return 0;
  }
#else
  return 0;
#endif
}

DUALSPH_CAPI int DsphGetGpuInfo(int gpuId, char* nameBuffer, int nameBufferSize,
                                int* computeCapMajor, int* computeCapMinor,
                                int* totalMemoryMB) {
#ifdef _WITHGPU
  try {
    int deviceCount = fcuda::GetCudaDevicesCount();
    if(gpuId < 0 || gpuId >= deviceCount) {
      SetError("Invalid GPU ID");
      return DSPH_ERROR_INVALID_PARAM;
    }

    fcuda::StCudaDeviceInfo info = fcuda::GetCudaDeviceInfo(gpuId);

    if(nameBuffer && nameBufferSize > 0) {
      strncpy(nameBuffer, info.name.c_str(), nameBufferSize - 1);
      nameBuffer[nameBufferSize - 1] = '\0';
    }

    if(computeCapMajor) *computeCapMajor = info.ccmajor;
    if(computeCapMinor) *computeCapMinor = info.ccminor;
    if(totalMemoryMB) *totalMemoryMB = static_cast<int>(info.globalmem / (1024 * 1024));

    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    SetError(std::string("Failed to get GPU info: ") + e.what());
    return DSPH_ERROR_UNKNOWN;
  }
#else
  SetError("GPU support not compiled");
  return DSPH_ERROR_NO_GPU;
#endif
}

//==============================================================================
// Simulation Lifecycle
//==============================================================================
DUALSPH_CAPI int DsphCreateSimulation(int deviceType, int gpuId, DsphSimHandle* outHandle) {
  if(!g_Initialized) {
    SetError("Library not initialized");
    return DSPH_ERROR_NOT_INIT;
  }

  if(!outHandle) {
    SetError("Invalid output handle pointer");
    return DSPH_ERROR_INVALID_PARAM;
  }

  if(deviceType != DSPH_DEVICE_CPU && deviceType != DSPH_DEVICE_GPU) {
    SetError("Invalid device type");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifndef _WITHGPU
  if(deviceType == DSPH_DEVICE_GPU) {
    SetError("GPU support not compiled");
    return DSPH_ERROR_NO_GPU;
  }
#endif

  try {
    DsphSimulation_* sim = new DsphSimulation_();
    sim->deviceType = deviceType;
    sim->gpuId = gpuId;

#ifdef _WITHGPU
    if(deviceType == DSPH_DEVICE_GPU) {
      cudaSetDevice(gpuId);
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      sim->cudaStream = stream;
      sim->ownsStream = true;
    }
#endif

    *outHandle = sim;
    ClearErrorInternal();
    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    SetError(std::string("Failed to create simulation: ") + e.what());
    return DSPH_ERROR_MEMORY;
  }
}

DUALSPH_CAPI int DsphDestroySimulation(DsphSimHandle handle) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

  try {
    delete handle;
    ClearErrorInternal();
    return DSPH_SUCCESS;
  }
  catch(...) {
    SetError("Failed to destroy simulation");
    return DSPH_ERROR_UNKNOWN;
  }
}

DUALSPH_CAPI int DsphResetSimulation(DsphSimHandle handle) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->FreeGpuMemory();
  handle->particles.Clear();
  handle->prepared = false;
  handle->simulationTime = 0.0;
  handle->stepCount = 0;
  handle->totalParticles = 0;
  handle->fluidParticles = 0;
  handle->boundaryParticles = 0;

  return DSPH_SUCCESS;
}

//==============================================================================
// Simulation Configuration
//==============================================================================
DUALSPH_CAPI int DsphSetDomain(DsphSimHandle handle,
                               double minX, double minY, double minZ,
                               double maxX, double maxY, double maxZ) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }

  handle->config.domainMinX = minX;
  handle->config.domainMinY = minY;
  handle->config.domainMinZ = minZ;
  handle->config.domainMaxX = maxX;
  handle->config.domainMaxY = maxY;
  handle->config.domainMaxZ = maxZ;
  handle->config.domainSet = true;

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetParticleSpacing(DsphSimHandle handle, double dp) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(dp <= 0) {
    SetError("Particle spacing must be positive");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->config.dp = dp;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetKernel(DsphSimHandle handle, int kernelType) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(kernelType != DSPH_KERNEL_CUBIC && kernelType != DSPH_KERNEL_WENDLAND) {
    SetError("Invalid kernel type");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->config.kernelType = kernelType;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetGravity(DsphSimHandle handle, double gx, double gy, double gz) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }

  handle->config.gravity[0] = gx;
  handle->config.gravity[1] = gy;
  handle->config.gravity[2] = gz;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetViscosity(DsphSimHandle handle, int viscoType, double viscoValue) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(viscoType < DSPH_VISCO_ARTIFICIAL || viscoType > DSPH_VISCO_LAMINAR_SPS) {
    SetError("Invalid viscosity type");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->config.viscoType = viscoType;
  handle->config.viscoValue = viscoValue;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetViscoBoundFactor(DsphSimHandle handle, double factor) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }

  handle->config.viscoBoundFactor = factor;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetTimeStepMethod(DsphSimHandle handle, int method) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(method != DSPH_STEP_VERLET && method != DSPH_STEP_SYMPLECTIC) {
    SetError("Invalid time step method");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->config.stepMethod = method;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetCFL(DsphSimHandle handle, double cfl) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(cfl <= 0 || cfl > 1.0) {
    SetError("CFL must be between 0 and 1");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->config.cfl = cfl;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetBoundaryMethod(DsphSimHandle handle, int method) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(method != DSPH_BOUNDARY_DBC && method != DSPH_BOUNDARY_MDBC) {
    SetError("Invalid boundary method");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->config.boundaryMethod = method;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetDensityDiffusion(DsphSimHandle handle, int ddtType, double ddtValue) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }

  handle->config.ddtType = ddtType;
  handle->config.ddtValue = ddtValue;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetReferenceDensity(DsphSimHandle handle, double rho0) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(rho0 <= 0) {
    SetError("Reference density must be positive");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->config.rho0 = rho0;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetSpeedOfSound(DsphSimHandle handle, double speedOfSound) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }

  handle->config.speedOfSound = speedOfSound;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSet2DMode(DsphSimHandle handle, int enable, double yPosition) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot modify configuration after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }

  handle->config.simulate2D = (enable != 0);
  handle->config.simulate2DPosY = yPosition;
  return DSPH_SUCCESS;
}

//==============================================================================
// Particle Definition
//==============================================================================
DUALSPH_CAPI int DsphAddFluidParticles(DsphSimHandle handle,
                                        const double* positions,
                                        const double* velocities,
                                        unsigned int count) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot add particles after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(!positions || count == 0) {
    SetError("Invalid positions array");
    return DSPH_ERROR_INVALID_PARAM;
  }

  try {
    size_t offset = handle->particles.fluidPositions.size();
    handle->particles.fluidPositions.resize(offset + count * 3);
    std::memcpy(handle->particles.fluidPositions.data() + offset,
                positions, count * 3 * sizeof(double));

    handle->particles.fluidVelocities.resize(offset + count * 3);
    if(velocities) {
      std::memcpy(handle->particles.fluidVelocities.data() + offset,
                  velocities, count * 3 * sizeof(double));
    } else {
      std::memset(handle->particles.fluidVelocities.data() + offset,
                  0, count * 3 * sizeof(double));
    }

    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    SetError(std::string("Failed to add fluid particles: ") + e.what());
    return DSPH_ERROR_MEMORY;
  }
}

DUALSPH_CAPI int DsphAddBoundaryParticles(DsphSimHandle handle,
                                           const double* positions,
                                           const double* normals,
                                           unsigned int count) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot add particles after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(!positions || count == 0) {
    SetError("Invalid positions array");
    return DSPH_ERROR_INVALID_PARAM;
  }

  try {
    size_t offset = handle->particles.boundaryPositions.size();
    handle->particles.boundaryPositions.resize(offset + count * 3);
    std::memcpy(handle->particles.boundaryPositions.data() + offset,
                positions, count * 3 * sizeof(double));

    handle->particles.boundaryNormals.resize(offset + count * 3);
    if(normals) {
      std::memcpy(handle->particles.boundaryNormals.data() + offset,
                  normals, count * 3 * sizeof(double));
    } else {
      std::memset(handle->particles.boundaryNormals.data() + offset,
                  0, count * 3 * sizeof(double));
    }

    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    SetError(std::string("Failed to add boundary particles: ") + e.what());
    return DSPH_ERROR_MEMORY;
  }
}

DUALSPH_CAPI int DsphAddFluidBlock(DsphSimHandle handle,
                                    double minX, double minY, double minZ,
                                    double maxX, double maxY, double maxZ,
                                    double velX, double velY, double velZ) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot add particles after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }
  if(handle->config.dp <= 0) {
    SetError("Particle spacing (dp) must be set before adding fluid blocks");
    return DSPH_ERROR_INVALID_STATE;
  }

  try {
    double dp = handle->config.dp;
    std::vector<double> positions;
    std::vector<double> velocities;

    // Generate particles in a grid
    for(double z = minZ + dp/2; z < maxZ; z += dp) {
      for(double y = minY + dp/2; y < maxY; y += dp) {
        for(double x = minX + dp/2; x < maxX; x += dp) {
          positions.push_back(x);
          positions.push_back(y);
          positions.push_back(z);
          velocities.push_back(velX);
          velocities.push_back(velY);
          velocities.push_back(velZ);
        }
      }
    }

    if(positions.empty()) {
      return DSPH_SUCCESS;  // Empty block is valid
    }

    return DsphAddFluidParticles(handle, positions.data(), velocities.data(),
                                  static_cast<unsigned int>(positions.size() / 3));
  }
  catch(const std::exception& e) {
    SetError(std::string("Failed to add fluid block: ") + e.what());
    return DSPH_ERROR_MEMORY;
  }
}

DUALSPH_CAPI int DsphGetDefinedParticleCounts(DsphSimHandle handle,
                                               unsigned int* outFluidCount,
                                               unsigned int* outBoundaryCount) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

  if(outFluidCount) *outFluidCount = handle->particles.FluidCount();
  if(outBoundaryCount) *outBoundaryCount = handle->particles.BoundaryCount();

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphClearParticles(DsphSimHandle handle) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Cannot clear particles after preparation");
    return DSPH_ERROR_ALREADY_PREPARED;
  }

  handle->particles.Clear();
  return DSPH_SUCCESS;
}

//==============================================================================
// External Buffer Configuration
//==============================================================================
DUALSPH_CAPI int DsphSetExternalParticleBuffer(DsphSimHandle handle,
                                                void* cudaDevicePtr,
                                                unsigned int maxParticles,
                                                int writeFluidOnly) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!cudaDevicePtr || maxParticles == 0) {
    SetError("Invalid buffer parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->externalBuffer.cudaDevicePtr = cudaDevicePtr;
  handle->externalBuffer.maxParticles = maxParticles;
  handle->externalBuffer.writeFluidOnly = (writeFluidOnly != 0);
  handle->externalBuffer.enabled = true;

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphClearExternalParticleBuffer(DsphSimHandle handle) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->externalBuffer.cudaDevicePtr = nullptr;
  handle->externalBuffer.maxParticles = 0;
  handle->externalBuffer.enabled = false;

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphGetCudaStream(DsphSimHandle handle, void** outCudaStream) {
  if(!handle || !outCudaStream) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  *outCudaStream = handle->cudaStream;
  return DSPH_SUCCESS;
#else
  SetError("GPU support not compiled");
  return DSPH_ERROR_NO_GPU;
#endif
}

DUALSPH_CAPI int DsphSetCudaStream(DsphSimHandle handle, void* cudaStream) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  // Free old stream if we own it
  if(handle->ownsStream && handle->cudaStream) {
    cudaStreamDestroy(static_cast<cudaStream_t>(handle->cudaStream));
  }

  handle->cudaStream = cudaStream;
  handle->ownsStream = false;
  return DSPH_SUCCESS;
#else
  SetError("GPU support not compiled");
  return DSPH_ERROR_NO_GPU;
#endif
}

//==============================================================================
// Simulation Preparation
//==============================================================================
DUALSPH_CAPI int DsphPrepare(DsphSimHandle handle) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(handle->prepared) {
    SetError("Simulation already prepared");
    return DSPH_ERROR_ALREADY_PREPARED;
  }

  try {
    const DsphSimConfig& cfg = handle->config;

    // Validate configuration
    if(handle->particles.FluidCount() == 0) {
      SetError("No fluid particles defined");
      return DSPH_ERROR_INVALID_STATE;
    }

    // Calculate SPH constants
    double dp = cfg.dp;
    double coefH = (cfg.kernelType == DSPH_KERNEL_WENDLAND) ?
                   fsph::GetKernelWendlandFactorH() : fsph::GetKernelCubicFactorH();
    double coefK = (cfg.kernelType == DSPH_KERNEL_WENDLAND) ?
                   fsph::GetKernelWendlandFactorK() : fsph::GetKernelCubicFactorK();

    handle->kernelH = static_cast<float>(coefH * sqrt(3.0 * dp * dp));
    handle->kernelSize = static_cast<float>(coefK * handle->kernelH);

    // Mass calculation (assuming uniform particle distribution)
    double volume = dp * dp * dp;
    handle->massFluid = static_cast<float>(cfg.rho0 * volume);
    handle->massBound = handle->massFluid;

    // Speed of sound
    if(cfg.speedOfSound > 0) {
      handle->cs0 = static_cast<float>(cfg.speedOfSound);
    } else {
      // Auto-calculate based on expected max velocity
      // Using 10x safety factor on expected gravity-driven velocity
      double domainHeight = cfg.domainMaxZ - cfg.domainMinZ;
      double vMax = sqrt(2.0 * fabs(cfg.gravity[2]) * domainHeight);
      handle->cs0 = static_cast<float>(10.0 * std::max(vMax, 1.0));
    }

    // Store particle counts
    handle->fluidParticles = handle->particles.FluidCount();
    handle->boundaryParticles = handle->particles.BoundaryCount();
    handle->totalParticles = handle->fluidParticles + handle->boundaryParticles;

#ifdef _WITHGPU
    if(handle->deviceType == DSPH_DEVICE_GPU) {
      cudaSetDevice(handle->gpuId);

      // Allocate GPU arrays
      unsigned int np = handle->totalParticles;
      handle->allocatedParticles = np;

      cudaMalloc(&handle->gpu_posX, np * sizeof(double));
      cudaMalloc(&handle->gpu_posY, np * sizeof(double));
      cudaMalloc(&handle->gpu_posZ, np * sizeof(double));
      cudaMalloc(&handle->gpu_velX, np * sizeof(float));
      cudaMalloc(&handle->gpu_velY, np * sizeof(float));
      cudaMalloc(&handle->gpu_velZ, np * sizeof(float));
      cudaMalloc(&handle->gpu_rho, np * sizeof(float));
      cudaMalloc(&handle->gpu_idp, np * sizeof(unsigned int));

      // Upload boundary particles first (they have lower indices)
      // Then fluid particles
      std::vector<double> posX(np), posY(np), posZ(np);
      std::vector<float> velX(np), velY(np), velZ(np);
      std::vector<float> rho(np, static_cast<float>(cfg.rho0));
      std::vector<unsigned int> idp(np);

      unsigned int idx = 0;

      // Boundary particles
      for(unsigned int i = 0; i < handle->boundaryParticles; i++) {
        posX[idx] = handle->particles.boundaryPositions[i * 3 + 0];
        posY[idx] = handle->particles.boundaryPositions[i * 3 + 1];
        posZ[idx] = handle->particles.boundaryPositions[i * 3 + 2];
        velX[idx] = 0.0f;
        velY[idx] = 0.0f;
        velZ[idx] = 0.0f;
        idp[idx] = idx;
        idx++;
      }

      // Fluid particles
      for(unsigned int i = 0; i < handle->fluidParticles; i++) {
        posX[idx] = handle->particles.fluidPositions[i * 3 + 0];
        posY[idx] = handle->particles.fluidPositions[i * 3 + 1];
        posZ[idx] = handle->particles.fluidPositions[i * 3 + 2];
        velX[idx] = static_cast<float>(handle->particles.fluidVelocities[i * 3 + 0]);
        velY[idx] = static_cast<float>(handle->particles.fluidVelocities[i * 3 + 1]);
        velZ[idx] = static_cast<float>(handle->particles.fluidVelocities[i * 3 + 2]);
        idp[idx] = idx;
        idx++;
      }

      cudaStream_t stream = static_cast<cudaStream_t>(handle->cudaStream);
      cudaMemcpyAsync(handle->gpu_posX, posX.data(), np * sizeof(double), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(handle->gpu_posY, posY.data(), np * sizeof(double), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(handle->gpu_posZ, posZ.data(), np * sizeof(double), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(handle->gpu_velX, velX.data(), np * sizeof(float), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(handle->gpu_velY, velY.data(), np * sizeof(float), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(handle->gpu_velZ, velZ.data(), np * sizeof(float), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(handle->gpu_rho, rho.data(), np * sizeof(float), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(handle->gpu_idp, idp.data(), np * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);

      cudaStreamSynchronize(stream);
    }
#endif

    handle->prepared = true;
    handle->simulationTime = 0.0;
    handle->stepCount = 0;

    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    SetError(std::string("Preparation failed: ") + e.what());
    return DSPH_ERROR_SIMULATION;
  }
}

DUALSPH_CAPI int DsphIsPrepared(DsphSimHandle handle) {
  if(!handle) return 0;
  return handle->prepared ? 1 : 0;
}

//==============================================================================
// Simulation Stepping
//==============================================================================
DUALSPH_CAPI int DsphComputeTimeStep(DsphSimHandle handle, double* outDt) {
  if(!handle || !outDt) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }

  // CFL condition: dt = CFL * h / (cs + vmax)
  // For now, use a simplified estimate
  double h = handle->kernelH;
  double cs = handle->cs0;
  double cfl = handle->config.cfl;

  // TODO: Compute actual vmax from particle velocities
  double vmax = 1.0;  // Placeholder

  *outDt = cfl * h / (cs + vmax);

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphStep(DsphSimHandle handle, double dt) {
  int result = DsphStepAsync(handle, dt);
  if(result != DSPH_SUCCESS) return result;
  return DsphSynchronize(handle);
}

DUALSPH_CAPI int DsphStepAsync(DsphSimHandle handle, double dt) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }
  if(dt <= 0) {
    SetError("Time step must be positive");
    return DSPH_ERROR_INVALID_PARAM;
  }

  try {
    // TODO: Implement actual SPH step using DualSPHysics kernels
    // This requires deeper integration with the JSphGpu internals
    // For now, this is a placeholder that updates time

    handle->simulationTime += dt;
    handle->stepCount++;

    // Copy to external buffer if configured
    if(handle->externalBuffer.enabled) {
      DsphCopyToExternalBuffer(handle);
    }

    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    SetError(std::string("Step failed: ") + e.what());
    return DSPH_ERROR_SIMULATION;
  }
}

DUALSPH_CAPI int DsphSynchronize(DsphSimHandle handle) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->cudaStream) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(handle->cudaStream));
  }
#endif

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphGetSimulationTime(DsphSimHandle handle, double* outTime) {
  if(!handle || !outTime) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }

  *outTime = handle->simulationTime;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphGetStepCount(DsphSimHandle handle, unsigned int* outSteps) {
  if(!handle || !outSteps) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }

  *outSteps = handle->stepCount;
  return DSPH_SUCCESS;
}

//==============================================================================
// Particle Data Access
//==============================================================================
DUALSPH_CAPI int DsphGetParticleCount(DsphSimHandle handle, unsigned int* outCount) {
  if(!handle || !outCount) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }

  *outCount = handle->fluidParticles;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphGetPositions(DsphSimHandle handle, float* outPositions, unsigned int count) {
  if(!handle || !outPositions) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }
  if(count > handle->fluidParticles) {
    SetError("Count exceeds particle count");
    return DSPH_ERROR_BUFFER_TOO_SMALL;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU) {
    // Download from GPU - need to convert from separate arrays to interleaved
    std::vector<double> posX(count), posY(count), posZ(count);
    unsigned int offset = handle->boundaryParticles;  // Fluid particles start after boundary

    cudaStream_t stream = static_cast<cudaStream_t>(handle->cudaStream);
    cudaMemcpyAsync(posX.data(), handle->gpu_posX + offset, count * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(posY.data(), handle->gpu_posY + offset, count * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(posZ.data(), handle->gpu_posZ + offset, count * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for(unsigned int i = 0; i < count; i++) {
      outPositions[i * 3 + 0] = static_cast<float>(posX[i]);
      outPositions[i * 3 + 1] = static_cast<float>(posY[i]);
      outPositions[i * 3 + 2] = static_cast<float>(posZ[i]);
    }
  }
#endif

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphGetPositionsDouble(DsphSimHandle handle, double* outPositions, unsigned int count) {
  if(!handle || !outPositions) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }
  if(count > handle->fluidParticles) {
    SetError("Count exceeds particle count");
    return DSPH_ERROR_BUFFER_TOO_SMALL;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU) {
    std::vector<double> posX(count), posY(count), posZ(count);
    unsigned int offset = handle->boundaryParticles;

    cudaStream_t stream = static_cast<cudaStream_t>(handle->cudaStream);
    cudaMemcpyAsync(posX.data(), handle->gpu_posX + offset, count * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(posY.data(), handle->gpu_posY + offset, count * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(posZ.data(), handle->gpu_posZ + offset, count * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for(unsigned int i = 0; i < count; i++) {
      outPositions[i * 3 + 0] = posX[i];
      outPositions[i * 3 + 1] = posY[i];
      outPositions[i * 3 + 2] = posZ[i];
    }
  }
#endif

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphGetVelocities(DsphSimHandle handle, float* outVelocities, unsigned int count) {
  if(!handle || !outVelocities) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }
  if(count > handle->fluidParticles) {
    SetError("Count exceeds particle count");
    return DSPH_ERROR_BUFFER_TOO_SMALL;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU) {
    std::vector<float> velX(count), velY(count), velZ(count);
    unsigned int offset = handle->boundaryParticles;

    cudaStream_t stream = static_cast<cudaStream_t>(handle->cudaStream);
    cudaMemcpyAsync(velX.data(), handle->gpu_velX + offset, count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(velY.data(), handle->gpu_velY + offset, count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(velZ.data(), handle->gpu_velZ + offset, count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for(unsigned int i = 0; i < count; i++) {
      outVelocities[i * 3 + 0] = velX[i];
      outVelocities[i * 3 + 1] = velY[i];
      outVelocities[i * 3 + 2] = velZ[i];
    }
  }
#endif

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphGetDensities(DsphSimHandle handle, float* outDensities, unsigned int count) {
  if(!handle || !outDensities) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }
  if(count > handle->fluidParticles) {
    SetError("Count exceeds particle count");
    return DSPH_ERROR_BUFFER_TOO_SMALL;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU) {
    unsigned int offset = handle->boundaryParticles;
    cudaStream_t stream = static_cast<cudaStream_t>(handle->cudaStream);
    cudaMemcpyAsync(outDensities, handle->gpu_rho + offset, count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
#endif

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphCopyToExternalBuffer(DsphSimHandle handle) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }
  if(!handle->externalBuffer.enabled) {
    SetError("External buffer not configured");
    return DSPH_ERROR_INVALID_STATE;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU) {
    unsigned int count = handle->externalBuffer.writeFluidOnly ?
                         handle->fluidParticles : handle->totalParticles;

    if(count > handle->externalBuffer.maxParticles) {
      SetError("External buffer too small");
      return DSPH_ERROR_BUFFER_TOO_SMALL;
    }

    // TODO: Implement efficient GPU kernel to copy data to external buffer
    // in DsphParticleData format (interleaved pos, density, vel, pressure)
    // For now, this is a placeholder

    // The actual implementation would use a CUDA kernel like:
    // copyToExternalBuffer<<<blocks, threads, 0, stream>>>(
    //     externalBuffer, posX, posY, posZ, velX, velY, velZ, rho, count);
  }
#endif

  return DSPH_SUCCESS;
}

//==============================================================================
// Error Handling
//==============================================================================
DUALSPH_CAPI const char* DsphGetLastError(void) {
  return g_LastError.c_str();
}

DUALSPH_CAPI void DsphClearError(void) {
  ClearErrorInternal();
}
