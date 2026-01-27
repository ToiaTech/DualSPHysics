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

} // namespace dsphker

#endif // _DsphStepEngine_ker_
