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
///
/// This API provides programmatic control over SPH simulations, designed for
/// integration with external engines (game engines, visualization tools, etc.).
///
/// Key features:
/// - Programmatic simulation setup (no XML files required)
/// - Fine-grained stepping control for engine integration
/// - External CUDA buffer injection for zero-copy GPU interop
/// - CUDA stream access for external synchronization (D3D12 fences, etc.)

#ifndef _DualSPHysicsAPI_
#define _DualSPHysicsAPI_

#include "DualSPHysicsLib.h"

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Error Codes
//==============================================================================
#define DSPH_SUCCESS               0
#define DSPH_ERROR_UNKNOWN        -1
#define DSPH_ERROR_INVALID_PARAM  -2
#define DSPH_ERROR_NOT_INIT       -3
#define DSPH_ERROR_ALREADY_INIT   -4
#define DSPH_ERROR_NO_GPU         -5
#define DSPH_ERROR_FILE_NOT_FOUND -6
#define DSPH_ERROR_SIMULATION     -7
#define DSPH_ERROR_MEMORY         -8
#define DSPH_ERROR_NOT_PREPARED   -9
#define DSPH_ERROR_ALREADY_PREPARED -10
#define DSPH_ERROR_INVALID_STATE  -11
#define DSPH_ERROR_BUFFER_TOO_SMALL -12
#define DSPH_ERROR_NOT_IMPLEMENTED -13
#define DSPH_ERROR_INVALID_BOUNDARY -14
#define DSPH_ERROR_INVALID_FLUID_TYPE -15
#define DSPH_ERROR_BOUNDARY_LIMIT -16
#define DSPH_ERROR_FLUID_TYPE_LIMIT -17
#define DSPH_ERROR_MESH_INVALID -18
#define DSPH_ERROR_SAMPLING_FAILED -19

//==============================================================================
// Device Types
//==============================================================================
#define DSPH_DEVICE_CPU  0
#define DSPH_DEVICE_GPU  1

//==============================================================================
// Kernel Types
//==============================================================================
#define DSPH_KERNEL_CUBIC     0
#define DSPH_KERNEL_WENDLAND  1
#define DSPH_KERNEL_POLY6     2   // Good for density estimation
#define DSPH_KERNEL_SPIKY     3   // Good for pressure gradient

//==============================================================================
// Viscosity Types
//==============================================================================
#define DSPH_VISCO_ARTIFICIAL   0
#define DSPH_VISCO_LAMINAR      1
#define DSPH_VISCO_LAMINAR_SPS  2

//==============================================================================
// Time Step Methods
//==============================================================================
#define DSPH_STEP_VERLET     0
#define DSPH_STEP_SYMPLECTIC 1

//==============================================================================
// Boundary Methods
//==============================================================================
#define DSPH_BOUNDARY_DBC   0   // Dynamic Boundary Condition
#define DSPH_BOUNDARY_MDBC  1   // Modified Dynamic Boundary Condition

//==============================================================================
// Density Diffusion Term
//==============================================================================
#define DSPH_DDT_NONE       0
#define DSPH_DDT_MOLTENI    1
#define DSPH_DDT_FOURTAKAS  2
#define DSPH_DDT_FOURTAKAS_FULL 3

//==============================================================================
// Opaque Handle Types
//==============================================================================
typedef struct DsphSimulation_* DsphSimHandle;

//==============================================================================
// Particle Data Structure (for external buffer output)
// 32 bytes per particle, 16-byte aligned for GPU efficiency
//==============================================================================
#pragma pack(push, 1)
typedef struct DsphParticleData {
    float posX, posY, posZ;    // Position (12 bytes)
    float density;              // Density (4 bytes)
    float velX, velY, velZ;    // Velocity (12 bytes)
    float pressure;             // Pressure (4 bytes)
} DsphParticleData;             // Total: 32 bytes
#pragma pack(pop)

//==============================================================================
// Version Information
//==============================================================================

/// Get the library version string.
/// @return Version string (e.g., "5.4.355")
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
// Library Initialization
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
// Simulation Lifecycle
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

/// Reset simulation to initial state (keeps configuration, clears particles).
/// @param handle Simulation handle
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphResetSimulation(DsphSimHandle handle);

//==============================================================================
// Simulation Configuration (call before DsphPrepare)
//==============================================================================

/// Set the simulation domain bounds.
/// @param handle Simulation handle
/// @param minX, minY, minZ Lower corner of domain
/// @param maxX, maxY, maxZ Upper corner of domain
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetDomain(DsphSimHandle handle,
                               double minX, double minY, double minZ,
                               double maxX, double maxY, double maxZ);

/// Set the initial particle spacing (dp).
/// This determines the resolution of the simulation.
/// @param handle Simulation handle
/// @param dp Particle spacing in world units
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetParticleSpacing(DsphSimHandle handle, double dp);

/// Set the SPH kernel type.
/// @param handle Simulation handle
/// @param kernelType DSPH_KERNEL_CUBIC or DSPH_KERNEL_WENDLAND
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetKernel(DsphSimHandle handle, int kernelType);

/// Set gravitational acceleration.
/// @param handle Simulation handle
/// @param gx, gy, gz Gravity vector components (e.g., 0, 0, -9.81)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetGravity(DsphSimHandle handle, double gx, double gy, double gz);

/// Set viscosity model and value.
/// @param handle Simulation handle
/// @param viscoType DSPH_VISCO_ARTIFICIAL, DSPH_VISCO_LAMINAR, or DSPH_VISCO_LAMINAR_SPS
/// @param viscoValue Viscosity coefficient
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetViscosity(DsphSimHandle handle, int viscoType, double viscoValue);

/// Set viscosity factor for boundary interactions.
/// @param handle Simulation handle
/// @param factor Multiplier for viscosity at boundaries (default 1.0)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetViscoBoundFactor(DsphSimHandle handle, double factor);

/// Set time integration method.
/// @param handle Simulation handle
/// @param method DSPH_STEP_VERLET or DSPH_STEP_SYMPLECTIC
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetTimeStepMethod(DsphSimHandle handle, int method);

/// Set the CFL number for adaptive time stepping.
/// @param handle Simulation handle
/// @param cfl CFL coefficient (default 0.2, lower = more stable but slower)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetCFL(DsphSimHandle handle, double cfl);

/// Set boundary condition method.
/// @param handle Simulation handle
/// @param method DSPH_BOUNDARY_DBC or DSPH_BOUNDARY_MDBC
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetBoundaryMethod(DsphSimHandle handle, int method);

/// Set density diffusion term (DDT) for pressure noise reduction.
/// @param handle Simulation handle
/// @param ddtType DSPH_DDT_NONE, DSPH_DDT_MOLTENI, DSPH_DDT_FOURTAKAS, or DSPH_DDT_FOURTAKAS_FULL
/// @param ddtValue DDT coefficient (default 0.1)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetDensityDiffusion(DsphSimHandle handle, int ddtType, double ddtValue);

/// Set fluid reference density.
/// @param handle Simulation handle
/// @param rho0 Reference density in kg/m^3 (default 1000 for water)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetReferenceDensity(DsphSimHandle handle, double rho0);

/// Set speed of sound coefficient.
/// Higher values = stiffer fluid, more stable but smaller timesteps.
/// @param handle Simulation handle
/// @param speedOfSound Speed of sound (default calculated from expected max velocity)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetSpeedOfSound(DsphSimHandle handle, double speedOfSound);

/// Enable or disable 2D simulation mode (forces in Y axis are zeroed).
/// @param handle Simulation handle
/// @param enable 1 to enable 2D mode, 0 for full 3D
/// @param yPosition Y coordinate for the 2D plane (only used if enable=1)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSet2DMode(DsphSimHandle handle, int enable, double yPosition);

//==============================================================================
// Particle Definition (call before DsphPrepare)
//==============================================================================

/// Add fluid particles to the simulation.
/// @param handle Simulation handle
/// @param positions Array of positions [x0,y0,z0, x1,y1,z1, ...] (count*3 doubles)
/// @param velocities Array of velocities [vx0,vy0,vz0, ...] (count*3 doubles), can be NULL for zero velocity
/// @param count Number of particles to add
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphAddFluidParticles(DsphSimHandle handle,
                                        const double* positions,
                                        const double* velocities,
                                        unsigned int count);

/// Add boundary particles to the simulation.
/// For mDBC, normals should point into the fluid domain.
/// @param handle Simulation handle
/// @param positions Array of positions [x0,y0,z0, x1,y1,z1, ...] (count*3 doubles)
/// @param normals Array of normals [nx0,ny0,nz0, ...] (count*3 doubles), can be NULL for DBC
/// @param count Number of particles to add
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphAddBoundaryParticles(DsphSimHandle handle,
                                           const double* positions,
                                           const double* normals,
                                           unsigned int count);

/// Add a rectangular block of fluid particles.
/// Particles are automatically generated with the configured spacing (dp).
/// @param handle Simulation handle
/// @param minX, minY, minZ Lower corner of fluid block
/// @param maxX, maxY, maxZ Upper corner of fluid block
/// @param velX, velY, velZ Initial velocity for all particles in block
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphAddFluidBlock(DsphSimHandle handle,
                                    double minX, double minY, double minZ,
                                    double maxX, double maxY, double maxZ,
                                    double velX, double velY, double velZ);

/// Get the total number of particles currently defined.
/// @param handle Simulation handle
/// @param outFluidCount Pointer to receive fluid particle count (can be NULL)
/// @param outBoundaryCount Pointer to receive boundary particle count (can be NULL)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetDefinedParticleCounts(DsphSimHandle handle,
                                               unsigned int* outFluidCount,
                                               unsigned int* outBoundaryCount);

/// Clear all defined particles (does not affect configuration).
/// @param handle Simulation handle
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphClearParticles(DsphSimHandle handle);

//==============================================================================
// External Buffer Configuration (for CUDA-D3D12/Vulkan interop)
//==============================================================================

/// Set an external CUDA buffer for particle data output.
/// DualSPHysics will write particle data directly to this buffer after each step.
/// The buffer must remain valid for the lifetime of the simulation.
///
/// This enables zero-copy integration with graphics APIs:
/// 1. Create D3D12/Vulkan buffer with shared flags
/// 2. Import as CUDA external memory (cudaImportExternalMemory)
/// 3. Get mapped device pointer (cudaExternalMemoryGetMappedBuffer)
/// 4. Pass that pointer here
/// 5. After DsphStep, the buffer contains particle data for rendering
///
/// @param handle Simulation handle
/// @param cudaDevicePtr CUDA device pointer from cudaExternalMemoryGetMappedBuffer
/// @param maxParticles Maximum particles the buffer can hold
/// @param writeFluidOnly If 1, only fluid particles are written (not boundaries)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetExternalParticleBuffer(DsphSimHandle handle,
                                                void* cudaDevicePtr,
                                                unsigned int maxParticles,
                                                int writeFluidOnly);

/// Disable external buffer output (use internal buffers).
/// @param handle Simulation handle
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphClearExternalParticleBuffer(DsphSimHandle handle);

/// Get the CUDA stream used by the simulation.
/// Use this for external synchronization (e.g., CUDA-D3D12 fences).
/// @param handle Simulation handle
/// @param outCudaStream Pointer to receive cudaStream_t (cast to void*)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetCudaStream(DsphSimHandle handle, void** outCudaStream);

/// Set an external CUDA stream for the simulation to use.
/// The caller is responsible for the stream's lifetime.
/// @param handle Simulation handle
/// @param cudaStream cudaStream_t to use (cast to void*)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSetCudaStream(DsphSimHandle handle, void* cudaStream);

//==============================================================================
// Simulation Preparation
//==============================================================================

/// Prepare the simulation for execution.
/// This allocates GPU memory, builds spatial data structures, and computes
/// initial values. Must be called after configuration and particle setup,
/// before stepping.
/// @param handle Simulation handle
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphPrepare(DsphSimHandle handle);

/// Check if simulation is prepared and ready for stepping.
/// @param handle Simulation handle
/// @return 1 if prepared, 0 otherwise
DUALSPH_CAPI int DsphIsPrepared(DsphSimHandle handle);

//==============================================================================
// Simulation Stepping
//==============================================================================

/// Compute the recommended timestep based on CFL condition.
/// Call this each frame to get a stable dt, or use a fixed dt if preferred.
/// @param handle Simulation handle
/// @param outDt Pointer to receive recommended timestep in seconds
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphComputeTimeStep(DsphSimHandle handle, double* outDt);

/// Perform a single simulation step (synchronous).
/// Blocks until the step is complete.
/// @param handle Simulation handle
/// @param dt Timestep in seconds
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphStep(DsphSimHandle handle, double dt);

/// Perform a single simulation step (asynchronous).
/// Returns immediately; the step executes on the CUDA stream.
/// Use DsphSynchronize() or external fence to wait for completion.
/// @param handle Simulation handle
/// @param dt Timestep in seconds
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphStepAsync(DsphSimHandle handle, double dt);

/// Wait for all pending asynchronous operations to complete.
/// @param handle Simulation handle
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphSynchronize(DsphSimHandle handle);

/// Get the current simulation time.
/// @param handle Simulation handle
/// @param outTime Pointer to receive current simulation time in seconds
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetSimulationTime(DsphSimHandle handle, double* outTime);

/// Get the number of simulation steps performed.
/// @param handle Simulation handle
/// @param outSteps Pointer to receive step count
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetStepCount(DsphSimHandle handle, unsigned int* outSteps);

//==============================================================================
// Particle Data Access (when not using external buffer)
//==============================================================================

/// Get the current number of active fluid particles.
/// @param handle Simulation handle
/// @param outCount Pointer to receive particle count
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetParticleCount(DsphSimHandle handle, unsigned int* outCount);

/// Copy particle positions from GPU to a CPU buffer.
/// @param handle Simulation handle
/// @param outPositions Buffer to receive positions [x0,y0,z0, x1,y1,z1, ...] (count*3 floats)
/// @param count Number of particles to copy (must not exceed particle count)
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetPositions(DsphSimHandle handle, float* outPositions, unsigned int count);

/// Copy particle positions from GPU to a CPU buffer (double precision).
/// @param handle Simulation handle
/// @param outPositions Buffer to receive positions [x0,y0,z0, x1,y1,z1, ...] (count*3 doubles)
/// @param count Number of particles to copy
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetPositionsDouble(DsphSimHandle handle, double* outPositions, unsigned int count);

/// Copy particle velocities from GPU to a CPU buffer.
/// @param handle Simulation handle
/// @param outVelocities Buffer to receive velocities [vx0,vy0,vz0, ...] (count*3 floats)
/// @param count Number of particles to copy
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetVelocities(DsphSimHandle handle, float* outVelocities, unsigned int count);

/// Copy particle densities from GPU to a CPU buffer.
/// @param handle Simulation handle
/// @param outDensities Buffer to receive densities (count floats)
/// @param count Number of particles to copy
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetDensities(DsphSimHandle handle, float* outDensities, unsigned int count);

/// Copy all particle data to the external buffer (if set).
/// This is called automatically after DsphStep/DsphStepAsync, but can be
/// called manually if needed.
/// @param handle Simulation handle
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphCopyToExternalBuffer(DsphSimHandle handle);

//==============================================================================
// Particle Acceleration Access
//==============================================================================

/// Copy particle accelerations from GPU to a CPU buffer.
/// Accelerations are the total (pressure + viscosity + gravity + external).
/// @param handle Simulation handle
/// @param outAccelerations Buffer to receive accelerations [ax0,ay0,az0, ...] (count*3 floats)
/// @param count Number of particles to copy
/// @return DSPH_SUCCESS on success, error code otherwise
DUALSPH_CAPI int DsphGetAccelerations(DsphSimHandle handle, float* outAccelerations, unsigned int count);

//==============================================================================
// Dynamic Boundary (Floating Body) Management
//==============================================================================

/// Add a dynamic boundary object (floating body) to the simulation.
/// For kinematic boundaries (isDynamic=false), call DsphUpdateBoundaryState()
/// each frame. Forces are computed and can be retrieved via DsphGetBoundaryForces().
///
/// @param handle Simulation handle
/// @param positions Boundary particle positions (x,y,z triplets) - count*3 floats
/// @param normals Surface normals at each particle (x,y,z triplets) - count*3 floats
/// @param count Number of boundary particles
/// @param mass Total mass of the boundary object (kg)
/// @param inertia Moment of inertia tensor (Ixx, Iyy, Izz, Ixy, Ixz, Iyz) - 6 floats
/// @param centerOfMass Center of mass position (x, y, z) - 3 floats
/// @param isDynamic If true, forces affect object motion; if false, kinematic control
/// @return Boundary handle (>= 0) on success, negative error code on failure
DUALSPH_CAPI int DsphAddDynamicBoundary(
    DsphSimHandle handle,
    const float* positions,
    const float* normals,
    unsigned int count,
    float mass,
    const float* inertia,
    const float* centerOfMass,
    int isDynamic
);

/// Get the number of dynamic boundary objects in the simulation.
/// @param handle Simulation handle
/// @return Number of boundaries (>= 0), or negative error code
DUALSPH_CAPI int DsphGetBoundaryCount(DsphSimHandle handle);

/// Retrieve accumulated forces and torques on a boundary object.
/// Forces are computed during DsphStep() from fluid-boundary interactions.
/// Forces are cleared after retrieval to prevent double-counting.
/// Torques are computed about the boundary's center of mass.
///
/// @param handle Simulation handle
/// @param boundaryId Boundary handle from DsphAddDynamicBoundary()
/// @param outForce Output: accumulated force vector (Fx, Fy, Fz) in Newtons - 3 floats
/// @param outTorque Output: accumulated torque vector (Tx, Ty, Tz) in N*m - 3 floats
/// @return DSPH_SUCCESS on success, error code on failure
DUALSPH_CAPI int DsphGetBoundaryForces(
    DsphSimHandle handle,
    int boundaryId,
    float* outForce,
    float* outTorque
);

/// Get forces without clearing the accumulator.
/// Useful when multiple systems need to read forces (e.g., haptics + logging).
/// Caller must manually call DsphClearBoundaryForces() when done.
DUALSPH_CAPI int DsphPeekBoundaryForces(
    DsphSimHandle handle,
    int boundaryId,
    float* outForce,
    float* outTorque
);

/// Manually clear accumulated forces on a boundary.
/// @param handle Simulation handle
/// @param boundaryId Boundary handle
/// @return DSPH_SUCCESS on success, error code on failure
DUALSPH_CAPI int DsphClearBoundaryForces(
    DsphSimHandle handle,
    int boundaryId
);

/// Update the kinematic state of a boundary object.
/// Call this before DsphStep() each frame for kinematic boundaries.
///
/// @param handle Simulation handle
/// @param boundaryId Boundary handle from DsphAddDynamicBoundary()
/// @param position New center of mass position (x, y, z) - 3 floats
/// @param velocity Linear velocity (vx, vy, vz) in m/s - 3 floats
/// @param orientation Rotation quaternion (qx, qy, qz, qw) normalized - 4 floats
/// @param angularVelocity Angular velocity (wx, wy, wz) in rad/s - 3 floats
/// @return DSPH_SUCCESS on success, error code on failure
DUALSPH_CAPI int DsphUpdateBoundaryState(
    DsphSimHandle handle,
    int boundaryId,
    const float* position,
    const float* velocity,
    const float* orientation,
    const float* angularVelocity
);

/// Update boundary state with a 4x4 transformation matrix.
/// Alternative to quaternion-based update for systems using matrices.
///
/// @param handle Simulation handle
/// @param boundaryId Boundary handle
/// @param transform 4x4 column-major transformation matrix - 16 floats
/// @param velocity Linear velocity (3 floats)
/// @param angularVelocity Angular velocity (3 floats)
/// @return DSPH_SUCCESS on success, error code on failure
DUALSPH_CAPI int DsphUpdateBoundaryStateMatrix(
    DsphSimHandle handle,
    int boundaryId,
    const float* transform,
    const float* velocity,
    const float* angularVelocity
);

/// Get current state of a boundary object.
/// @param handle Simulation handle
/// @param boundaryId Boundary handle
/// @param outPosition Current position (3 floats, can be NULL)
/// @param outVelocity Current velocity (3 floats, can be NULL)
/// @param outOrientation Current orientation quaternion (4 floats, can be NULL)
/// @param outAngularVelocity Current angular velocity (3 floats, can be NULL)
/// @return DSPH_SUCCESS on success, error code on failure
DUALSPH_CAPI int DsphGetBoundaryState(
    DsphSimHandle handle,
    int boundaryId,
    float* outPosition,
    float* outVelocity,
    float* outOrientation,
    float* outAngularVelocity
);

//==============================================================================
// Multiple Fluid Type Support (Experimental)
//==============================================================================

/// Create a new fluid type with specific properties.
/// Call this before adding particles. Cannot create new types after DsphPrepare().
///
/// @param handle Simulation handle
/// @param density Rest density in kg/m^3 (typical: water=1000, oil=900)
/// @param viscosity Dynamic viscosity in Pa*s (typical: water=0.001, oil=0.1)
/// @param surfaceTension Surface tension coefficient in N/m (typical: water=0.0728)
/// @return Fluid type handle (>= 0) on success, negative error code on failure
///
/// @note Multi-fluid interaction is experimental and may not be fully stable.
DUALSPH_CAPI int DsphCreateFluidType(
    DsphSimHandle handle,
    float density,
    float viscosity,
    float surfaceTension
);

/// Add fluid particles of a specific type.
/// @param handle Simulation handle
/// @param fluidType Fluid type handle from DsphCreateFluidType()
/// @param positions Particle positions (x,y,z triplets) - count*3 floats
/// @param velocities Initial velocities (x,y,z triplets), or NULL for zero - count*3 floats
/// @param count Number of particles
/// @return DSPH_SUCCESS on success, error code on failure
DUALSPH_CAPI int DsphAddFluidParticlesTyped(
    DsphSimHandle handle,
    int fluidType,
    const float* positions,
    const float* velocities,
    unsigned int count
);

/// Get number of fluid types in the simulation.
/// @param handle Simulation handle
/// @return Number of fluid types (>= 0), or negative error code
DUALSPH_CAPI int DsphGetFluidTypeCount(DsphSimHandle handle);

/// Get particle count for a specific fluid type.
/// @param handle Simulation handle
/// @param fluidType Fluid type handle
/// @param outCount Pointer to receive particle count
/// @return DSPH_SUCCESS on success, error code on failure
DUALSPH_CAPI int DsphGetFluidTypeParticleCount(
    DsphSimHandle handle,
    int fluidType,
    unsigned int* outCount
);

/// Set viscosity for a specific fluid type (can be called during simulation).
/// @param handle Simulation handle
/// @param fluidType Fluid type handle
/// @param viscosity New viscosity value in Pa*s
/// @return DSPH_SUCCESS on success, error code on failure
DUALSPH_CAPI int DsphSetFluidTypeViscosity(
    DsphSimHandle handle,
    int fluidType,
    float viscosity
);

//==============================================================================
// Error Handling
//==============================================================================

/// Get the last error message.
/// @return Error message string, or empty string if no error
DUALSPH_CAPI const char* DsphGetLastError(void);

/// Clear the last error.
DUALSPH_CAPI void DsphClearError(void);

#ifdef __cplusplus
}
#endif

#endif // _DualSPHysicsAPI_
