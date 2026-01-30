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

/// \file FunSphMultiFluid_iker.h \brief Declares GPU structures and functions for multi-fluid SPH support.

#ifndef _FunSphMultiFluid_iker_
#define _FunSphMultiFluid_iker_

#include <cuda_runtime_api.h>

#define MULTIFLUID_MAX_TYPES 16

/// Structure for fluid type properties on GPU.
/// Stored in constant memory for fast access during particle interactions.
struct StGpuFluidType {
  float rho0;             ///< Reference density [kg/m3]
  float viscosity;        ///< Viscosity coefficient
  float mass;             ///< Particle mass [kg]
  float cs0;              ///< Speed of sound [m/s]
  float cteB;             ///< Pressure constant B = rho0 * cs0^2 / gamma
};

/// Constant memory for fluid type properties.
/// Each CUDA compilation unit gets its own copy of constant memory.
__constant__ StGpuFluidType c_MultiFluidTypes[MULTIFLUID_MAX_TYPES];
__constant__ unsigned int c_MultiFluidTypeCount;

namespace cufsph {

//==============================================================================
/// Device function: Compute pressure using per-fluid parameters.
/// Uses Tait equation: P = B * ((rho/rho0)^gamma - 1)
//==============================================================================
__device__ __forceinline__ float ComputePressMultiFluid(float rho, float rho0, float cteB, float gamma) {
  return cteB * (powf(rho / rho0, gamma) - 1.0f);
}

//==============================================================================
/// Device function: Get pressure for a particle given its fluid type.
/// Looks up rho0 and cteB from constant memory.
//==============================================================================
__device__ __forceinline__ float ComputePressFromFluidType(float rho, unsigned char fluidType, float gamma) {
  const StGpuFluidType& ft = c_MultiFluidTypes[fluidType < c_MultiFluidTypeCount ? fluidType : 0];
  return ComputePressMultiFluid(rho, ft.rho0, ft.cteB, gamma);
}

//==============================================================================
/// Device function: Get averaged viscosity for a particle pair.
/// Uses arithmetic mean: visc_ij = (visc_i + visc_j) / 2
//==============================================================================
__device__ __forceinline__ float GetAveragedViscosity(unsigned char type1, unsigned char type2) {
  const unsigned char t1 = (type1 < c_MultiFluidTypeCount) ? type1 : 0;
  const unsigned char t2 = (type2 < c_MultiFluidTypeCount) ? type2 : 0;
  return 0.5f * (c_MultiFluidTypes[t1].viscosity + c_MultiFluidTypes[t2].viscosity);
}

//==============================================================================
/// Device function: Get mass for a particle given its fluid type.
//==============================================================================
__device__ __forceinline__ float GetFluidTypeMass(unsigned char fluidType) {
  return c_MultiFluidTypes[fluidType < c_MultiFluidTypeCount ? fluidType : 0].mass;
}

//==============================================================================
/// Device function: Get reference density for a particle given its fluid type.
//==============================================================================
__device__ __forceinline__ float GetFluidTypeRho0(unsigned char fluidType) {
  return c_MultiFluidTypes[fluidType < c_MultiFluidTypeCount ? fluidType : 0].rho0;
}

//==============================================================================
/// Device function: Get pressure constant B for a particle given its fluid type.
//==============================================================================
__device__ __forceinline__ float GetFluidTypeCteB(unsigned char fluidType) {
  return c_MultiFluidTypes[fluidType < c_MultiFluidTypeCount ? fluidType : 0].cteB;
}

} // namespace cufsph

//==============================================================================
// Host functions for uploading multi-fluid data to GPU
//==============================================================================
namespace cufsph {

/// Upload fluid type properties to GPU constant memory.
void UploadMultiFluidTypes(const StGpuFluidType* types, unsigned int count);

/// Upload fluid type properties from separate arrays.
void UploadMultiFluidTypesFromArrays(
  const float* rho0,
  const float* viscosity,
  const float* mass,
  const float* cs0,
  const float* cteB,
  unsigned int count);

/// Clear multi-fluid constant memory.
void ClearMultiFluidTypes();

} // namespace cufsph

#endif // _FunSphMultiFluid_iker_
