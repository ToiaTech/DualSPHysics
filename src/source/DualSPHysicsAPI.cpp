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
  #include "DsphStepEngine.h"
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
// Fluid Type Definition (pre-preparation)
//==============================================================================
struct DsphFluidTypeDef {
  float density;
  float viscosity;
  float surfaceTension;
  bool active;

  DsphFluidTypeDef() : density(1000.0f), viscosity(0.01f), surfaceTension(0.0f), active(false) {}
};

struct DsphFluidTypeStorage {
  DsphFluidTypeDef types[DSPH_MAX_FLUID_TYPES];
  unsigned int count;

  DsphFluidTypeStorage() : count(1) {
    // Initialize default fluid type 0
    types[0].density = 1000.0f;
    types[0].viscosity = 0.01f;
    types[0].surfaceTension = 0.0f;
    types[0].active = true;
  }

  void Clear() {
    count = 1;
    types[0].active = true;
    for(int i = 1; i < DSPH_MAX_FLUID_TYPES; i++) {
      types[i].active = false;
    }
  }
};

//==============================================================================
// Particle Storage Structure (pre-preparation)
//==============================================================================
struct DsphParticleStorage {
  std::vector<double> fluidPositions;     // x,y,z,x,y,z,...
  std::vector<double> fluidVelocities;    // vx,vy,vz,...
  std::vector<unsigned char> fluidTypes;  // Per-particle fluid type ID
  std::vector<double> boundaryPositions;  // x,y,z,...
  std::vector<double> boundaryNormals;    // nx,ny,nz,...

  unsigned int FluidCount() const { return static_cast<unsigned int>(fluidPositions.size() / 3); }
  unsigned int BoundaryCount() const { return static_cast<unsigned int>(boundaryPositions.size() / 3); }

  void Clear() {
    fluidPositions.clear();
    fluidVelocities.clear();
    fluidTypes.clear();
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
  DsphFluidTypeStorage fluidTypes;
  DsphExternalBuffer externalBuffer;

  // Runtime state
  bool prepared;
  double simulationTime;
  unsigned int stepCount;

  // CUDA stream (GPU only)
  void* cudaStream;
  bool ownsStream;

  // SPH Step Engine (manages GPU simulation)
#ifdef _WITHGPU
  DsphStepEngine* stepEngine;
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
#ifdef _WITHGPU
    stepEngine(nullptr),
#endif
    totalParticles(0),
    fluidParticles(0),
    boundaryParticles(0)
  {}

  ~DsphSimulation_() {
#ifdef _WITHGPU
    if(stepEngine) {
      delete stepEngine;
      stepEngine = nullptr;
    }
#endif
    if(ownsStream && cudaStream) {
#ifdef _WITHGPU
      cudaStreamDestroy(static_cast<cudaStream_t>(cudaStream));
#endif
    }
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

#ifdef _WITHGPU
  if(handle->stepEngine) {
    handle->stepEngine->Shutdown();
    delete handle->stepEngine;
    handle->stepEngine = nullptr;
  }
#endif

  handle->particles.Clear();
  handle->fluidTypes.Clear();
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
                                        unsigned int count,
                                        int fluidType) {
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
  if(fluidType < 0 || (unsigned int)fluidType >= handle->fluidTypes.count) {
    SetError("Invalid fluid type ID - create type first with DsphCreateFluidType");
    return DSPH_ERROR_INVALID_FLUID_TYPE;
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

    // Store fluid type for all particles
    size_t typeOffset = handle->particles.fluidTypes.size();
    handle->particles.fluidTypes.resize(typeOffset + count, static_cast<unsigned char>(fluidType));

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
                                    double velX, double velY, double velZ,
                                    int fluidType) {
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
                                  static_cast<unsigned int>(positions.size() / 3), fluidType);
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

    // Store particle counts
    handle->fluidParticles = handle->particles.FluidCount();
    handle->boundaryParticles = handle->particles.BoundaryCount();
    handle->totalParticles = handle->fluidParticles + handle->boundaryParticles;

#ifdef _WITHGPU
    if(handle->deviceType == DSPH_DEVICE_GPU) {
      cudaSetDevice(handle->gpuId);

      // Create step engine configuration
      StDsphEngineConfig engineConfig;
      engineConfig.domainMin = TDouble3(cfg.domainMinX, cfg.domainMinY, cfg.domainMinZ);
      engineConfig.domainMax = TDouble3(cfg.domainMaxX, cfg.domainMaxY, cfg.domainMaxZ);
      engineConfig.dp = cfg.dp;
      engineConfig.gravity = TFloat3(float(cfg.gravity[0]), float(cfg.gravity[1]), float(cfg.gravity[2]));
      engineConfig.kernel = (cfg.kernelType == DSPH_KERNEL_WENDLAND) ? KERNEL_Wendland : KERNEL_Cubic;
      engineConfig.visco = (cfg.viscoType == DSPH_VISCO_ARTIFICIAL) ? VISCO_Artificial :
                           (cfg.viscoType == DSPH_VISCO_LAMINAR) ? VISCO_LaminarSPS : VISCO_LaminarSPS;
      engineConfig.viscoValue = float(cfg.viscoValue);
      engineConfig.viscoBoundFactor = float(cfg.viscoBoundFactor);
      engineConfig.stepMethod = (cfg.stepMethod == DSPH_STEP_VERLET) ? STEP_Verlet : STEP_Symplectic;
      engineConfig.cfl = cfg.cfl;
      engineConfig.boundary = (cfg.boundaryMethod == DSPH_BOUNDARY_MDBC) ? BC_MDBC : BC_DBC;
      engineConfig.density = (TpDensity)cfg.ddtType;
      engineConfig.ddtValue = float(cfg.ddtValue);
      engineConfig.rho0 = float(cfg.rho0);
      engineConfig.cs0 = float(cfg.speedOfSound);
      engineConfig.simulate2D = cfg.simulate2D;
      engineConfig.simulate2DPosY = cfg.simulate2DPosY;

      // Create and initialize step engine
      handle->stepEngine = new DsphStepEngine();

      cudaStream_t stream = static_cast<cudaStream_t>(handle->cudaStream);

      bool success = handle->stepEngine->Initialize(
        engineConfig,
        handle->particles.fluidPositions.data(),
        handle->particles.fluidVelocities.empty() ? nullptr : handle->particles.fluidVelocities.data(),
        handle->fluidParticles,
        handle->particles.boundaryPositions.empty() ? nullptr : handle->particles.boundaryPositions.data(),
        handle->particles.boundaryNormals.empty() ? nullptr : handle->particles.boundaryNormals.data(),
        handle->boundaryParticles,
        stream
      );

      if(!success) {
        delete handle->stepEngine;
        handle->stepEngine = nullptr;
        SetError("Failed to initialize step engine");
        return DSPH_ERROR_SIMULATION;
      }

      // Set external buffer if configured
      if(handle->externalBuffer.enabled) {
        handle->stepEngine->SetExternalBuffer(
          handle->externalBuffer.cudaDevicePtr,
          handle->externalBuffer.maxParticles,
          handle->externalBuffer.writeFluidOnly
        );
      }

      // Update fluid type 0 properties if they were customized
      if(handle->fluidTypes.types[0].active) {
        handle->stepEngine->SetFluidTypeDensity(0, handle->fluidTypes.types[0].density);
        handle->stepEngine->SetFluidTypeViscosity(0, handle->fluidTypes.types[0].viscosity);
      }

      // Create additional fluid types in step engine (type 0 already exists)
      for(unsigned int i = 1; i < handle->fluidTypes.count; i++) {
        const DsphFluidTypeDef& ft = handle->fluidTypes.types[i];
        handle->stepEngine->CreateFluidType(ft.density, ft.viscosity, ft.surfaceTension);
      }

      // Upload per-particle fluid types if any were set
      if(!handle->particles.fluidTypes.empty()) {
        handle->stepEngine->SetAllParticleFluidTypes(
          handle->particles.fluidTypes.data(),
          static_cast<unsigned int>(handle->particles.fluidTypes.size())
        );
      }
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

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    *outDt = handle->stepEngine->ComputeTimeStep();
    return DSPH_SUCCESS;
  }
#endif

  // CPU fallback (simplified estimate)
  *outDt = 0.0001;  // Conservative default
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
#ifdef _WITHGPU
    if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
      bool success = handle->stepEngine->StepAsync(dt);
      if(!success) {
        SetError("Step engine step failed");
        return DSPH_ERROR_SIMULATION;
      }

      // Update local counters from engine
      handle->simulationTime = handle->stepEngine->GetSimulationTime();
      handle->stepCount = handle->stepEngine->GetStepCount();

      // Copy to external buffer if configured (async)
      if(handle->externalBuffer.enabled) {
        handle->stepEngine->CopyToExternalBuffer();
      }

      return DSPH_SUCCESS;
    }
#endif

    // CPU fallback (not implemented)
    SetError("CPU simulation not yet implemented");
    return DSPH_ERROR_NOT_IMPLEMENTED;
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
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    handle->stepEngine->Synchronize();
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
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    handle->stepEngine->GetPositions(outPositions, count);
    return DSPH_SUCCESS;
  }
#endif

  SetError("CPU simulation not yet implemented");
  return DSPH_ERROR_NOT_IMPLEMENTED;
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
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    // Get positions as float and convert to double
    std::vector<float> posFloat(count * 3);
    handle->stepEngine->GetPositions(posFloat.data(), count);
    for(unsigned int i = 0; i < count * 3; i++) {
      outPositions[i] = static_cast<double>(posFloat[i]);
    }
    return DSPH_SUCCESS;
  }
#endif

  SetError("CPU simulation not yet implemented");
  return DSPH_ERROR_NOT_IMPLEMENTED;
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
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    handle->stepEngine->GetVelocities(outVelocities, count);
    return DSPH_SUCCESS;
  }
#endif

  SetError("CPU simulation not yet implemented");
  return DSPH_ERROR_NOT_IMPLEMENTED;
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
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    handle->stepEngine->GetDensities(outDensities, count);
    return DSPH_SUCCESS;
  }
#endif

  SetError("CPU simulation not yet implemented");
  return DSPH_ERROR_NOT_IMPLEMENTED;
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
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    handle->stepEngine->CopyToExternalBuffer();
    return DSPH_SUCCESS;
  }
#endif

  SetError("CPU simulation not yet implemented");
  return DSPH_ERROR_NOT_IMPLEMENTED;
}

//==============================================================================
// Particle Acceleration Access
//==============================================================================
DUALSPH_CAPI int DsphGetAccelerations(DsphSimHandle handle, float* outAccelerations, unsigned int count) {
  if(!handle || !outAccelerations) {
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
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    handle->stepEngine->GetAccelerations(outAccelerations, count);
    return DSPH_SUCCESS;
  }
#endif

  SetError("CPU simulation not yet implemented");
  return DSPH_ERROR_NOT_IMPLEMENTED;
}

//==============================================================================
// Dynamic Boundary Management
//==============================================================================
DUALSPH_CAPI int DsphAddDynamicBoundary(
    DsphSimHandle handle,
    const float* positions,
    const float* normals,
    unsigned int count,
    float mass,
    const float* inertia,
    const float* centerOfMass,
    int isDynamic)
{
  if(!handle || !positions || count == 0 || !centerOfMass) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->prepared && handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    // After preparation: use step engine's dynamic boundary support
    int boundaryId = handle->stepEngine->AddBoundaryObject(
      positions,
      normals,
      count,
      mass,
      inertia,
      centerOfMass,
      isDynamic != 0
    );

    if(boundaryId < 0) {
      SetError("Failed to add dynamic boundary (limit reached or allocation failed)");
      return DSPH_ERROR_BOUNDARY_LIMIT;
    }

    return boundaryId;
  }
#endif

  // Before preparation: store as static boundary particles
  // Dynamic boundaries should be added AFTER preparation for full functionality
  if(!handle->prepared) {
    // Convert float positions to double for existing API
    std::vector<double> posDouble(count * 3);
    std::vector<double> normDouble(count * 3);
    for(unsigned int i = 0; i < count * 3; i++) {
      posDouble[i] = static_cast<double>(positions[i]);
      normDouble[i] = normals ? static_cast<double>(normals[i]) : 0.0;
    }

    int result = DsphAddBoundaryParticles(handle, posDouble.data(),
                                           normals ? normDouble.data() : nullptr, count);
    if(result != DSPH_SUCCESS) return result;

    // Return 0 as static boundary ID (no dynamic features)
    return 0;
  }

  SetError("GPU not available for dynamic boundary");
  return DSPH_ERROR_NO_GPU;
}

DUALSPH_CAPI int DsphGetBoundaryCount(DsphSimHandle handle) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->prepared && handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    return (int)handle->stepEngine->GetBoundaryCount();
  }
#endif

  // Fallback: return 1 if we have boundary particles, 0 otherwise
  return (handle->boundaryParticles > 0) ? 1 : 0;
}

DUALSPH_CAPI int DsphGetBoundaryForces(
    DsphSimHandle handle,
    int boundaryId,
    float* outForce,
    float* outTorque)
{
  if(!handle || !outForce || !outTorque) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(boundaryId < 0 || (unsigned int)boundaryId >= handle->stepEngine->GetBoundaryCount()) {
      SetError("Invalid boundary ID");
      return DSPH_ERROR_INVALID_BOUNDARY;
    }

    bool success = handle->stepEngine->GetBoundaryForces(
      (unsigned int)boundaryId,
      outForce,
      outTorque,
      true  // clearAfterRead
    );

    if(!success) {
      SetError("Failed to get boundary forces");
      return DSPH_ERROR_INVALID_BOUNDARY;
    }

    return DSPH_SUCCESS;
  }
#endif

  // No dynamic boundaries on CPU - return zeros
  outForce[0] = outForce[1] = outForce[2] = 0.0f;
  outTorque[0] = outTorque[1] = outTorque[2] = 0.0f;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphPeekBoundaryForces(
    DsphSimHandle handle,
    int boundaryId,
    float* outForce,
    float* outTorque)
{
  if(!handle || !outForce || !outTorque) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(boundaryId < 0 || (unsigned int)boundaryId >= handle->stepEngine->GetBoundaryCount()) {
      SetError("Invalid boundary ID");
      return DSPH_ERROR_INVALID_BOUNDARY;
    }

    bool success = handle->stepEngine->GetBoundaryForces(
      (unsigned int)boundaryId,
      outForce,
      outTorque,
      false  // don't clear after read
    );

    if(!success) {
      SetError("Failed to peek boundary forces");
      return DSPH_ERROR_INVALID_BOUNDARY;
    }

    return DSPH_SUCCESS;
  }
#endif

  // No dynamic boundaries on CPU - return zeros
  outForce[0] = outForce[1] = outForce[2] = 0.0f;
  outTorque[0] = outTorque[1] = outTorque[2] = 0.0f;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphClearBoundaryForces(DsphSimHandle handle, int boundaryId) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->prepared && handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(boundaryId >= 0 && (unsigned int)boundaryId < handle->stepEngine->GetBoundaryCount()) {
      handle->stepEngine->ClearBoundaryForces((unsigned int)boundaryId);
    }
  }
#endif

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphUpdateBoundaryState(
    DsphSimHandle handle,
    int boundaryId,
    const float* position,
    const float* velocity,
    const float* orientation,
    const float* angularVelocity)
{
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(boundaryId < 0 || (unsigned int)boundaryId >= handle->stepEngine->GetBoundaryCount()) {
      SetError("Invalid boundary ID");
      return DSPH_ERROR_INVALID_BOUNDARY;
    }

    bool success = handle->stepEngine->UpdateBoundaryState(
      (unsigned int)boundaryId,
      position,
      velocity,
      orientation,
      angularVelocity
    );

    if(!success) {
      SetError("Failed to update boundary state");
      return DSPH_ERROR_INVALID_BOUNDARY;
    }

    return DSPH_SUCCESS;
  }
#endif

  SetError("Dynamic boundary state updates require GPU");
  return DSPH_ERROR_NO_GPU;
}

DUALSPH_CAPI int DsphUpdateBoundaryStateMatrix(
    DsphSimHandle handle,
    int boundaryId,
    const float* transform,
    const float* velocity,
    const float* angularVelocity)
{
  if(!handle || !transform) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }

  // Extract position from 4x4 matrix (last column)
  float position[3] = { transform[12], transform[13], transform[14] };

  // Convert rotation matrix to quaternion
  // Using Shepperd's method for numerical stability
  float m00 = transform[0], m01 = transform[4], m02 = transform[8];
  float m10 = transform[1], m11 = transform[5], m12 = transform[9];
  float m20 = transform[2], m21 = transform[6], m22 = transform[10];

  float trace = m00 + m11 + m22;
  float orientation[4];

  if(trace > 0) {
    float s = 0.5f / sqrtf(trace + 1.0f);
    orientation[3] = 0.25f / s;  // w
    orientation[0] = (m21 - m12) * s;  // x
    orientation[1] = (m02 - m20) * s;  // y
    orientation[2] = (m10 - m01) * s;  // z
  } else if(m00 > m11 && m00 > m22) {
    float s = 2.0f * sqrtf(1.0f + m00 - m11 - m22);
    orientation[3] = (m21 - m12) / s;
    orientation[0] = 0.25f * s;
    orientation[1] = (m01 + m10) / s;
    orientation[2] = (m02 + m20) / s;
  } else if(m11 > m22) {
    float s = 2.0f * sqrtf(1.0f + m11 - m00 - m22);
    orientation[3] = (m02 - m20) / s;
    orientation[0] = (m01 + m10) / s;
    orientation[1] = 0.25f * s;
    orientation[2] = (m12 + m21) / s;
  } else {
    float s = 2.0f * sqrtf(1.0f + m22 - m00 - m11);
    orientation[3] = (m10 - m01) / s;
    orientation[0] = (m02 + m20) / s;
    orientation[1] = (m12 + m21) / s;
    orientation[2] = 0.25f * s;
  }

  // Normalize quaternion
  float len = sqrtf(orientation[0]*orientation[0] + orientation[1]*orientation[1] +
                    orientation[2]*orientation[2] + orientation[3]*orientation[3]);
  if(len > 0) {
    orientation[0] /= len;
    orientation[1] /= len;
    orientation[2] /= len;
    orientation[3] /= len;
  }

  return DsphUpdateBoundaryState(handle, boundaryId, position, velocity,
                                  orientation, angularVelocity);
}

DUALSPH_CAPI int DsphGetBoundaryState(
    DsphSimHandle handle,
    int boundaryId,
    float* outPosition,
    float* outVelocity,
    float* outOrientation,
    float* outAngularVelocity)
{
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->prepared && handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(boundaryId < 0 || (unsigned int)boundaryId >= handle->stepEngine->GetBoundaryCount()) {
      SetError("Invalid boundary ID");
      return DSPH_ERROR_INVALID_BOUNDARY;
    }

    bool success = handle->stepEngine->GetBoundaryState(
      (unsigned int)boundaryId,
      outPosition,
      outVelocity,
      outOrientation,
      outAngularVelocity
    );

    if(!success) {
      SetError("Failed to get boundary state");
      return DSPH_ERROR_INVALID_BOUNDARY;
    }

    return DSPH_SUCCESS;
  }
#endif

  // Return identity state for CPU fallback
  if(outPosition) { outPosition[0] = outPosition[1] = outPosition[2] = 0.0f; }
  if(outVelocity) { outVelocity[0] = outVelocity[1] = outVelocity[2] = 0.0f; }
  if(outOrientation) {
    outOrientation[0] = outOrientation[1] = outOrientation[2] = 0.0f;
    outOrientation[3] = 1.0f;  // Identity quaternion
  }
  if(outAngularVelocity) {
    outAngularVelocity[0] = outAngularVelocity[1] = outAngularVelocity[2] = 0.0f;
  }

  return DSPH_SUCCESS;
}

//==============================================================================
// Multiple Fluid Type Support
//==============================================================================
DUALSPH_CAPI int DsphCreateFluidType(
    DsphSimHandle handle,
    float density,
    float viscosity,
    float surfaceTension)
{
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(density <= 0) {
    SetError("Density must be positive");
    return DSPH_ERROR_INVALID_PARAM;
  }

  // Before preparation: store in handle's fluid type storage
  if(!handle->prepared) {
    if(handle->fluidTypes.count >= DSPH_MAX_FLUID_TYPES) {
      SetError("Maximum fluid type limit reached");
      return DSPH_ERROR_FLUID_TYPE_LIMIT;
    }

    unsigned int typeId = handle->fluidTypes.count;
    DsphFluidTypeDef& ft = handle->fluidTypes.types[typeId];
    ft.density = density;
    ft.viscosity = viscosity;
    ft.surfaceTension = surfaceTension;
    ft.active = true;
    handle->fluidTypes.count++;

    return (int)typeId;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    // After preparation: use step engine's multi-fluid support
    int typeId = handle->stepEngine->CreateFluidType(density, viscosity, surfaceTension);
    if(typeId < 0) {
      SetError("Failed to create fluid type (limit reached)");
      return DSPH_ERROR_FLUID_TYPE_LIMIT;
    }
    return typeId;
  }
#endif

  SetError("Multi-fluid requires GPU after preparation");
  return DSPH_ERROR_NO_GPU;
}

DUALSPH_CAPI int DsphGetFluidTypeCount(DsphSimHandle handle) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->prepared && handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    return (int)handle->stepEngine->GetFluidTypeCount();
  }
#endif

  // Before preparation: return count from fluid type storage
  return (int)handle->fluidTypes.count;
}

DUALSPH_CAPI int DsphGetFluidTypeParticleCount(
    DsphSimHandle handle,
    int fluidType,
    unsigned int* outCount)
{
  if(!handle || !outCount) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->prepared && handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(fluidType < 0 || (unsigned int)fluidType >= handle->stepEngine->GetFluidTypeCount()) {
      SetError("Invalid fluid type ID");
      return DSPH_ERROR_INVALID_FLUID_TYPE;
    }

    *outCount = handle->stepEngine->GetFluidTypeParticleCount((unsigned int)fluidType);
    return DSPH_SUCCESS;
  }
#endif

  // Before preparation or CPU: only type 0, return all fluid particles
  if(fluidType != 0) {
    SetError("Invalid fluid type");
    return DSPH_ERROR_INVALID_FLUID_TYPE;
  }

  *outCount = handle->fluidParticles;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetFluidTypeViscosity(
    DsphSimHandle handle,
    int fluidType,
    float viscosity)
{
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->prepared && handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(fluidType < 0 || (unsigned int)fluidType >= handle->stepEngine->GetFluidTypeCount()) {
      SetError("Invalid fluid type ID");
      return DSPH_ERROR_INVALID_FLUID_TYPE;
    }

    if(!handle->stepEngine->SetFluidTypeViscosity((unsigned int)fluidType, viscosity)) {
      SetError("Failed to set fluid type viscosity");
      return DSPH_ERROR_INVALID_FLUID_TYPE;
    }
    return DSPH_SUCCESS;
  }
#endif

  // Before preparation: update config
  if(!handle->prepared && fluidType == 0) {
    handle->config.viscoValue = viscosity;
    return DSPH_SUCCESS;
  }

  if(fluidType != 0) {
    SetError("Invalid fluid type");
    return DSPH_ERROR_INVALID_FLUID_TYPE;
  }

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetFluidTypeDensity(
    DsphSimHandle handle,
    int fluidType,
    float density)
{
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(density <= 0) {
    SetError("Density must be positive");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->prepared && handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(fluidType < 0 || (unsigned int)fluidType >= handle->stepEngine->GetFluidTypeCount()) {
      SetError("Invalid fluid type ID");
      return DSPH_ERROR_INVALID_FLUID_TYPE;
    }

    if(!handle->stepEngine->SetFluidTypeDensity((unsigned int)fluidType, density)) {
      SetError("Failed to set fluid type density");
      return DSPH_ERROR_INVALID_FLUID_TYPE;
    }
    return DSPH_SUCCESS;
  }
#endif

  // Before preparation: update config for type 0
  if(!handle->prepared && fluidType == 0) {
    handle->config.rho0 = density;
    return DSPH_SUCCESS;
  }

  if(fluidType != 0) {
    SetError("Invalid fluid type");
    return DSPH_ERROR_INVALID_FLUID_TYPE;
  }

  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetParticleFluidType(
    DsphSimHandle handle,
    unsigned int particleIndex,
    int fluidType)
{
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(fluidType < 0 || (unsigned int)fluidType >= handle->stepEngine->GetFluidTypeCount()) {
      SetError("Invalid fluid type ID");
      return DSPH_ERROR_INVALID_FLUID_TYPE;
    }

    if(!handle->stepEngine->SetParticleFluidType(particleIndex, (unsigned int)fluidType)) {
      SetError("Failed to set particle fluid type (invalid particle index)");
      return DSPH_ERROR_INVALID_PARAM;
    }
    return DSPH_SUCCESS;
  }
#endif

  SetError("Particle fluid types require GPU");
  return DSPH_ERROR_NO_GPU;
}

DUALSPH_CAPI int DsphGetParticleFluidType(
    DsphSimHandle handle,
    unsigned int particleIndex,
    int* outFluidType)
{
  if(!handle || !outFluidType) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }
  if(!handle->prepared) {
    SetError("Simulation not prepared");
    return DSPH_ERROR_NOT_PREPARED;
  }

#ifdef _WITHGPU
  if(handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    *outFluidType = (int)handle->stepEngine->GetParticleFluidType(particleIndex);
    return DSPH_SUCCESS;
  }
#endif

  // CPU fallback: all particles are type 0
  *outFluidType = 0;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphGetFluidTypeProperties(
    DsphSimHandle handle,
    int fluidType,
    float* outDensity,
    float* outViscosity,
    float* outSurfaceTension)
{
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

#ifdef _WITHGPU
  if(handle->prepared && handle->deviceType == DSPH_DEVICE_GPU && handle->stepEngine) {
    if(fluidType < 0 || (unsigned int)fluidType >= handle->stepEngine->GetFluidTypeCount()) {
      SetError("Invalid fluid type ID");
      return DSPH_ERROR_INVALID_FLUID_TYPE;
    }

    if(!handle->stepEngine->GetFluidTypeProperties((unsigned int)fluidType,
                                                    outDensity, outViscosity, outSurfaceTension)) {
      SetError("Failed to get fluid type properties");
      return DSPH_ERROR_INVALID_FLUID_TYPE;
    }
    return DSPH_SUCCESS;
  }
#endif

  // Before preparation or CPU: return config values for type 0
  if(fluidType != 0) {
    SetError("Invalid fluid type");
    return DSPH_ERROR_INVALID_FLUID_TYPE;
  }

  if(outDensity) *outDensity = (float)handle->config.rho0;
  if(outViscosity) *outViscosity = (float)handle->config.viscoValue;
  if(outSurfaceTension) *outSurfaceTension = 0.0f;

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
