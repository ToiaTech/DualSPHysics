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
#include "JSphCpuSingle.h"
#include "JSph.h"
#include "Functions.h"

#ifdef _WITHGPU
  #include "JSphGpuSingle.h"
  #include "FunctionsCuda.h"
#endif

#ifdef _WITHMR
  #include "JSphVResDriver.h"
  #include "JSphCpuSingle_VRes.h"
  #ifdef _WITHGPU
    #include "JSphGpuSingle_VRes.h"
  #endif
#endif

#include <string>
#include <cstring>
#include <mutex>

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

  // App info for DLL (separate from the global AppInfo used in main.cpp)
  JAppInfo* g_DllAppInfo = nullptr;

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
// Simulation handle structure
//==============================================================================
struct DsphSimulation_ {
  int deviceType;
  int gpuId;
  JSphCfgRun* cfg;
  JLog2* log;
  bool caseLoaded;
  std::string casePath;
  std::string outputDir;

  DsphSimulation_() : deviceType(DSPH_DEVICE_CPU), gpuId(0), cfg(nullptr),
                      log(nullptr), caseLoaded(false) {}

  ~DsphSimulation_() {
    delete log;
    delete cfg;
  }
};

//==============================================================================
// Version information
//==============================================================================
DUALSPH_CAPI const char* DsphGetVersion(void) {
  if(g_VersionStr.empty()) {
    g_VersionStr = "v5.4.355";  // Match main.cpp version
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
// Library initialization
//==============================================================================
DUALSPH_CAPI int DsphInitialize(void) {
  std::lock_guard<std::mutex> lock(g_Mutex);

  if(g_Initialized) {
    g_LastError = "Library already initialized";
    return DSPH_ERROR_ALREADY_INIT;
  }

  try {
    // Create DLL-specific app info
    g_DllAppInfo = new JAppInfo("DualSPHysics5", "v5.4.355", "08-04-2025");
    g_Initialized = true;
    g_LastError.clear();
    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    g_LastError = std::string("Initialization failed: ") + e.what();
    return DSPH_ERROR_UNKNOWN;
  }
  catch(...) {
    g_LastError = "Initialization failed: Unknown error";
    return DSPH_ERROR_UNKNOWN;
  }
}

DUALSPH_CAPI int DsphShutdown(void) {
  std::lock_guard<std::mutex> lock(g_Mutex);

  if(!g_Initialized) {
    g_LastError = "Library not initialized";
    return DSPH_ERROR_NOT_INIT;
  }

  try {
    delete g_DllAppInfo;
    g_DllAppInfo = nullptr;
    g_Initialized = false;
    g_LastError.clear();
    return DSPH_SUCCESS;
  }
  catch(...) {
    g_LastError = "Shutdown failed: Unknown error";
    return DSPH_ERROR_UNKNOWN;
  }
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
  catch(...) {
    SetError("Failed to get GPU info: Unknown error");
    return DSPH_ERROR_UNKNOWN;
  }
#else
  SetError("GPU support not compiled");
  return DSPH_ERROR_NO_GPU;
#endif
}

//==============================================================================
// Simulation management
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
    sim->cfg = new JSphCfgRun();
    sim->log = new JLog2();
    sim->caseLoaded = false;

    *outHandle = sim;
    ClearErrorInternal();
    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    SetError(std::string("Failed to create simulation: ") + e.what());
    return DSPH_ERROR_MEMORY;
  }
  catch(...) {
    SetError("Failed to create simulation: Unknown error");
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
    SetError("Failed to destroy simulation: Unknown error");
    return DSPH_ERROR_UNKNOWN;
  }
}

DUALSPH_CAPI int DsphLoadCase(DsphSimHandle handle, const char* casePath, const char* outputDir) {
  if(!g_Initialized) {
    SetError("Library not initialized");
    return DSPH_ERROR_NOT_INIT;
  }

  if(!handle || !casePath) {
    SetError("Invalid parameters");
    return DSPH_ERROR_INVALID_PARAM;
  }

  try {
    handle->casePath = casePath;
    handle->outputDir = outputDir ? outputDir : "";

    // Configure the JSphCfgRun
    JSphCfgRun* cfg = handle->cfg;
    cfg->Reset();

    // Set device configuration
    cfg->Cpu = (handle->deviceType == DSPH_DEVICE_CPU);
    cfg->Gpu = (handle->deviceType == DSPH_DEVICE_GPU);
    cfg->GpuId = handle->gpuId;

    // Set case name
    cfg->CaseName = casePath;

    // Set output directory
    if(outputDir && strlen(outputDir) > 0) {
      cfg->DirOut = outputDir;
    }

    // Set feature list
    cfg->SetFeatureList(JSph::GetFeatureList());

    handle->caseLoaded = true;
    ClearErrorInternal();
    return DSPH_SUCCESS;
  }
  catch(const std::exception& e) {
    SetError(std::string("Failed to load case: ") + e.what());
    return DSPH_ERROR_FILE_NOT_FOUND;
  }
  catch(...) {
    SetError("Failed to load case: Unknown error");
    return DSPH_ERROR_UNKNOWN;
  }
}

DUALSPH_CAPI int DsphRunSimulation(DsphSimHandle handle) {
  if(!g_Initialized) {
    SetError("Library not initialized");
    return DSPH_ERROR_NOT_INIT;
  }

  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

  if(!handle->caseLoaded) {
    SetError("No case loaded");
    return DSPH_ERROR_INVALID_PARAM;
  }

  try {
    std::string appname = DsphGetFullName();
    JSphCfgRun* cfg = handle->cfg;
    JLog2* log = handle->log;

    // Initialize log
    if(!handle->outputDir.empty()) {
      log->Init(handle->outputDir + "/Run.out");
    }

    // Run simulation based on device type
    if(cfg->Cpu) {
      // CPU simulation
#ifdef _WITHMR
      if(cfg->VRes) {
        JSphVResDriver<JSphCpuSingle_VRes, stinterparmscb> sph;
        sph.Run(appname, cfg, log);
      } else {
        JSphCpuSingle sph;
        sph.Run(appname, cfg, log);
      }
#else
      JSphCpuSingle sph;
      sph.Run(appname, cfg, log);
#endif
    }
    else {
      // GPU simulation
#ifdef _WITHGPU
#ifdef _WITHMR
      if(cfg->VRes) {
        JSphVResDriver<JSphGpuSingle_VRes, StInterParmsbg> sph;
        sph.Run(appname, cfg, log);
      } else {
        JSphGpuSingle sph;
        sph.Run(appname, cfg, log);
      }
#else
      JSphGpuSingle sph;
      sph.Run(appname, cfg, log);
#endif
#else
      SetError("GPU support not compiled");
      return DSPH_ERROR_NO_GPU;
#endif
    }

    ClearErrorInternal();
    return DSPH_SUCCESS;
  }
  catch(const char* cad) {
    SetError(std::string("Simulation error: ") + cad);
    return DSPH_ERROR_SIMULATION;
  }
  catch(const std::string& e) {
    SetError(std::string("Simulation error: ") + e);
    return DSPH_ERROR_SIMULATION;
  }
  catch(const JException& e) {
    SetError(std::string("Simulation error: ") + e.what());
    return DSPH_ERROR_SIMULATION;
  }
  catch(const std::exception& e) {
    SetError(std::string("Simulation error: ") + e.what());
    return DSPH_ERROR_SIMULATION;
  }
  catch(...) {
    SetError("Simulation error: Unknown exception");
    return DSPH_ERROR_SIMULATION;
  }
}

//==============================================================================
// Simulation configuration
//==============================================================================
DUALSPH_CAPI int DsphSetTimeMax(DsphSimHandle handle, double timeMax) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

  if(timeMax <= 0) {
    SetError("TimeMax must be positive");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->cfg->TimeMax = timeMax;
  return DSPH_SUCCESS;
}

DUALSPH_CAPI int DsphSetTimePart(DsphSimHandle handle, double timePart) {
  if(!handle) {
    SetError("Invalid simulation handle");
    return DSPH_ERROR_INVALID_PARAM;
  }

  if(timePart <= 0) {
    SetError("TimePart must be positive");
    return DSPH_ERROR_INVALID_PARAM;
  }

  handle->cfg->TimePart = timePart;
  return DSPH_SUCCESS;
}

//==============================================================================
// Error handling
//==============================================================================
DUALSPH_CAPI const char* DsphGetLastError(void) {
  return g_LastError.c_str();
}

DUALSPH_CAPI void DsphClearError(void) {
  ClearErrorInternal();
}
