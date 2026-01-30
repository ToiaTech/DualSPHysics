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

/// \file FunSphMultiFluid_iker.cu \brief Implements multi-fluid constant memory and upload functions.

#include "FunSphMultiFluid_iker.h"
#include <cstring>

namespace cufsph {

//==============================================================================
/// Upload fluid type properties to GPU constant memory.
/// @param types Array of fluid type properties
/// @param count Number of fluid types (max MULTIFLUID_MAX_TYPES)
//==============================================================================
void UploadMultiFluidTypes(const StGpuFluidType* types, unsigned int count) {
  if(count > MULTIFLUID_MAX_TYPES) count = MULTIFLUID_MAX_TYPES;

  cudaMemcpyToSymbol(c_MultiFluidTypes, types, count * sizeof(StGpuFluidType));
  cudaMemcpyToSymbol(c_MultiFluidTypeCount, &count, sizeof(unsigned int));
}

//==============================================================================
/// Upload fluid type properties from separate arrays.
/// @param rho0 Reference density array
/// @param viscosity Viscosity coefficient array
/// @param mass Particle mass array
/// @param cs0 Speed of sound array
/// @param cteB Pressure constant B array
/// @param count Number of fluid types
//==============================================================================
void UploadMultiFluidTypesFromArrays(
  const float* rho0,
  const float* viscosity,
  const float* mass,
  const float* cs0,
  const float* cteB,
  unsigned int count)
{
  if(count > MULTIFLUID_MAX_TYPES) count = MULTIFLUID_MAX_TYPES;

  StGpuFluidType types[MULTIFLUID_MAX_TYPES];
  for(unsigned int i = 0; i < count; i++) {
    types[i].rho0 = rho0[i];
    types[i].viscosity = viscosity[i];
    types[i].mass = mass[i];
    types[i].cs0 = cs0[i];
    types[i].cteB = cteB[i];
  }

  UploadMultiFluidTypes(types, count);
}

//==============================================================================
/// Clear multi-fluid constant memory (set count to 0).
//==============================================================================
void ClearMultiFluidTypes() {
  unsigned int zero = 0;
  cudaMemcpyToSymbol(c_MultiFluidTypeCount, &zero, sizeof(unsigned int));
}

} // namespace cufsph
