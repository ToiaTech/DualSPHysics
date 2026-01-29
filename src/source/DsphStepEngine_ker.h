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

/// \file DsphStepEngine_ker.h \brief Declares CUDA kernels for DsphStepEngine.

#ifndef _DsphStepEngine_ker_
#define _DsphStepEngine_ker_

#include "TypesDef.h"
#include <cuda_runtime.h>

/// CUDA kernels for DsphStepEngine DLL API.
namespace dsphker {

/// Copy particle data to external buffer in interleaved DsphParticleData format.
/// Output format: posX, posY, posZ, density, velX, velY, velZ, pressure (32 bytes per particle).
void CopyToExternalBuffer(
  unsigned count,
  const double2* posxy,
  const double* posz,
  const float4* velrho,
  float rho0,
  float cs0,
  float gamma,
  float* output,
  cudaStream_t stm = nullptr);

/// Simple Verlet position/velocity update (gravity-only forces).
void SimpleVerletUpdate(
  unsigned np,
  unsigned npb,
  const float3* ace,
  tfloat3 gravity,
  double dt,
  double2* posxy,
  double* posz,
  float4* velrho,
  float rho0,
  cudaStream_t stm = nullptr);

/// Simple Symplectic predictor step.
void SimpleSymplecticPre(
  unsigned np,
  unsigned npb,
  const float3* ace,
  tfloat3 gravity,
  double dtm,
  const double2* posxyPre,
  const double* poszPre,
  const float4* velrhoPre,
  double2* posxy,
  double* posz,
  float4* velrho,
  float rho0,
  cudaStream_t stm = nullptr);

/// Simple Symplectic corrector step.
void SimpleSymplecticCor(
  unsigned np,
  unsigned npb,
  const float3* ace,
  tfloat3 gravity,
  double dtm,
  double dt,
  const double2* posxyPre,
  const double* poszPre,
  const float4* velrhoPre,
  double2* posxy,
  double* posz,
  float4* velrho,
  float rho0,
  cudaStream_t stm = nullptr);

//==============================================================================
// Dynamic Boundary Transformation Kernels
//==============================================================================

/// Transform boundary particles from local (CoM-relative) to world coordinates.
/// Uses quaternion rotation + translation to compute world positions.
/// Also computes particle velocities from rigid body motion: v = v_linear + omega x r
void TransformBoundaryParticles(
  unsigned int count,
  const float3* localPos,         // Local positions (relative to CoM)
  const float3* localNorm,        // Local normals (optional, can be nullptr)
  float3 comPosition,             // Center of mass world position
  float4 orientation,             // Rotation quaternion (x,y,z,w)
  float3 linearVel,               // Linear velocity
  float3 angularVel,              // Angular velocity
  float3* worldPos,               // Output: world positions
  float3* worldNorm,              // Output: world normals (optional, can be nullptr)
  float3* particleVel,            // Output: particle velocities
  cudaStream_t stm = nullptr);

/// Accumulate fluid forces acting on a boundary object.
/// Reads from SPH acceleration array and computes total force/torque.
void AccumulateBoundaryForces(
  unsigned int particleStart,     // First particle index in boundary arrays
  unsigned int particleCount,     // Number of particles
  const float3* worldPos,         // World positions
  float3 comPosition,             // Center of mass
  const float3* ace,              // Acceleration array (from SPH)
  float particleMass,             // Mass per particle
  float3* outForce,               // Output: accumulated force (device ptr, single value)
  float3* outTorque,              // Output: accumulated torque (device ptr, single value)
  cudaStream_t stm = nullptr);

/// Compute fluid pressure forces on boundary particles using SPH interaction.
/// This performs neighbor search and computes pressure forces from fluid onto boundary.
/// Force formula: F = -m_b * sum_f[ m_f * (p_f/rho_f^2 + p_b/rho_b^2) * grad(W) ]
void ComputeBoundaryFluidForces(
  unsigned int boundaryCount,       // Number of boundary particles
  const float3* boundWorldPos,      // Boundary world positions
  const float3* boundWorldNorm,     // Boundary world normals
  float3 comPosition,               // Center of mass for torque calc
  // Fluid particle data
  unsigned int np,                  // Total particles
  unsigned int npb,                 // Boundary particles in main arrays
  const double2* fluidPosxy,        // Fluid positions XY
  const double* fluidPosz,          // Fluid position Z
  const float4* fluidVelrho,        // Fluid velocities + density
  // Cell division data
  const int* cellBegin,             // Cell begin indices
  unsigned int cellCode,            // Cell encoding
  double3 cellPosMin,               // Cell minimum position
  float cellSize,                   // Cell size
  // SPH parameters
  float kernelH,                    // Smoothing length
  float kernelSize,                 // Kernel support radius (2h)
  float massFluid,                  // Fluid particle mass
  float massBound,                  // Boundary particle mass
  float rho0,                       // Reference density
  float cs0,                        // Speed of sound
  float gamma,                      // Polytropic constant
  // Output (device pointers)
  float3* outForce,                 // Output: accumulated force
  float3* outTorque,                // Output: accumulated torque
  cudaStream_t stm = nullptr);

/// Zero a float3 value on device
void ZeroFloat3(float3* ptr, cudaStream_t stm = nullptr);

/// Zero an array of float3 values on device
void ZeroFloat3Array(float3* ptr, unsigned int count, cudaStream_t stm = nullptr);

} // namespace dsphker

#endif // _DsphStepEngine_ker_
