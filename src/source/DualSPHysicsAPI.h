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

/// \file DualSPHysicsAPI.h \brief C-compatible API for DualSPHysics DLL.

#ifndef _DualSPHysicsAPI_
#define _DualSPHysicsAPI_

#include "DualSPHysicsLib.h"

//==============================================================================
// Error codes
//==============================================================================
#define DSPH_SUCCESS              0
#define DSPH_ERROR_UNKNOWN       -1
#define DSPH_ERROR_INVALID_PARAM -2
#define DSPH_ERROR_NOT_INIT      -3
#define DSPH_ERROR_ALREADY_INIT  -4
#define DSPH_ERROR_NO_GPU        -5
#define DSPH_ERROR_FILE_NOT_FOUND -6
#define DSPH_ERROR_SIMULATION    -7
#define DSPH_ERROR_MEMORY        -8

//==============================================================================
// Device types
//==============================================================================
#define DSPH_DEVICE_CPU  0
#define DSPH_DEVICE_GPU  1

//==============================================================================
// Opaque handle types
//==============================================================================
typedef struct DsphSimulation_* DsphSimHandle;

//==============================================================================
// Version information
//==============================================================================

/// Get the library version string.
/// @return Version string (e.g., "v5.4.355")
DUALSPH_CAPI const char* DsphGetVersion(void);

/// Get the full library name with version.
/// @return Full name string (e.g., "DualSPHysics5 v5.4.355")
DUALSPH_CAPI const char* DsphGetFullName(void);

/// Check if GPU support is available in this build.
/// @return 1 if GPU support is available, 0 otherwise
DUALSPH_CAPI int DsphHasGpuSupport(void);

/// Get the list of available features.
/// @return Feature list string
DUALSPH_CAPI const char* DsphGetFeatures(void);

//==============================================================================
// Library initialization
//==============================================================================

/// Initialize the DualSPHysics library. Must be called before any other functions.
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphInitialize(void);

/// Shutdown the DualSPHysics library and release all resources.
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphShutdown(void);

/// Check if the library is initialized.
/// @return 1 if initialized, 0 otherwise
DUALSPH_CAPI int DsphIsInitialized(void);

//==============================================================================
// GPU Information
//==============================================================================

/// Get the number of available CUDA GPUs.
/// @return Number of GPUs, or 0 if no GPUs available or GPU support not compiled
DUALSPH_CAPI int DsphGetGpuCount(void);

/// Get information about a specific GPU.
/// @param gpuId GPU index (0-based)
/// @param nameBuffer Buffer to receive GPU name (can be NULL)
/// @param nameBufferSize Size of nameBuffer
/// @param computeCapMajor Pointer to receive compute capability major version (can be NULL)
/// @param computeCapMinor Pointer to receive compute capability minor version (can be NULL)
/// @param totalMemoryMB Pointer to receive total memory in MB (can be NULL)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetGpuInfo(int gpuId, char* nameBuffer, int nameBufferSize,
                                int* computeCapMajor, int* computeCapMinor,
                                int* totalMemoryMB);

//==============================================================================
// Simulation management
//==============================================================================

/// Create a new simulation instance.
/// @param deviceType DSPH_DEVICE_CPU or DSPH_DEVICE_GPU
/// @param gpuId GPU device ID (ignored for CPU, use 0 for default GPU)
/// @param outHandle Pointer to receive the simulation handle
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphCreateSimulation(int deviceType, int gpuId, DsphSimHandle* outHandle);

/// Destroy a simulation instance and release its resources.
/// @param handle Simulation handle
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphDestroySimulation(DsphSimHandle handle);

/// Load simulation configuration from an XML case file.
/// @param handle Simulation handle
/// @param casePath Path to the case XML file (without _Def.xml suffix)
/// @param outputDir Output directory path (can be NULL for default)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphLoadCase(DsphSimHandle handle, const char* casePath, const char* outputDir);

/// Run the full simulation (blocking call).
/// @param handle Simulation handle
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphRunSimulation(DsphSimHandle handle);

//==============================================================================
// Simulation configuration (call after DsphLoadCase, before DsphRunSimulation)
//==============================================================================

/// Set the maximum simulation time.
/// @param handle Simulation handle
/// @param timeMax Maximum simulation time in seconds
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetTimeMax(DsphSimHandle handle, double timeMax);

/// Set the time interval for saving output files.
/// @param handle Simulation handle
/// @param timePart Time interval between PART outputs in seconds
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetTimePart(DsphSimHandle handle, double timePart);

//==============================================================================
// Error handling
//==============================================================================

/// Get the last error message.
/// @return Error message string, or empty string if no error
DUALSPH_CAPI const char* DsphGetLastError(void);

/// Clear the last error.
DUALSPH_CAPI void DsphClearError(void);

#endif
