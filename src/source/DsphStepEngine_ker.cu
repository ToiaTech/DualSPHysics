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

/// \file DsphStepEngine_ker.cu \brief CUDA kernels for DsphStepEngine.

#include "DsphStepEngine_ker.h"
#include <cuda_runtime.h>

namespace dsphker{

#define BSIZE 256

//==============================================================================
/// Returns grid size for simple kernels.
//==============================================================================
inline dim3 GetGridSize(unsigned n, unsigned blocksize) {
  dim3 sgrid;
  const unsigned nb = unsigned(n + blocksize - 1) / blocksize;
  sgrid.x = (nb <= 65535 ? nb : unsigned(ceil(sqrt(float(nb)))));
  sgrid.y = 1 + (nb - 1) / sgrid.x;
  sgrid.z = 1;
  return sgrid;
}

//==============================================================================
/// CUDA kernel: Copy particle data to external buffer in interleaved format.
/// Output format: posX, posY, posZ, density, velX, velY, velZ, pressure
/// Each particle is 32 bytes (8 floats).
//==============================================================================
__global__ void KerCopyToExternalBuffer(
  unsigned count,
  const double2* posxy,
  const double* posz,
  const float4* velrho,
  float rho0,
  float cs0,
  float gamma,
  float* output)
{
  const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
  if(p < count) {
    const double2 rposxy = posxy[p];
    const double rposz = posz[p];
    const float4 rvelrho = velrho[p];

    // Calculate pressure from density using Tait equation:
    // P = B * ((rho/rho0)^gamma - 1)
    // where B = rho0 * cs0^2 / gamma
    const float rho = rvelrho.w;
    const float B = rho0 * cs0 * cs0 / gamma;
    const float rhoRatio = rho / rho0;
    float pressure = B * (powf(rhoRatio, gamma) - 1.0f);
    if(pressure < 0.0f) pressure = 0.0f;  // Clamp negative pressures

    // Output stride: 8 floats per particle (32 bytes)
    unsigned out = p * 8;
    output[out + 0] = float(rposxy.x);  // posX
    output[out + 1] = float(rposxy.y);  // posY
    output[out + 2] = float(rposz);     // posZ
    output[out + 3] = rho;              // density
    output[out + 4] = rvelrho.x;        // velX
    output[out + 5] = rvelrho.y;        // velY
    output[out + 6] = rvelrho.z;        // velZ
    output[out + 7] = pressure;         // pressure
  }
}

//==============================================================================
/// Copies particle data to external buffer.
//==============================================================================
void CopyToExternalBuffer(
  unsigned count,
  const double2* posxy,
  const double* posz,
  const float4* velrho,
  float rho0,
  float cs0,
  float gamma,
  float* output,
  cudaStream_t stm)
{
  if(count > 0) {
    dim3 sgrid = GetGridSize(count, BSIZE);
    KerCopyToExternalBuffer<<<sgrid, BSIZE, 0, stm>>>(
      count, posxy, posz, velrho, rho0, cs0, gamma, output);
  }
}

//==============================================================================
/// CUDA kernel: Simple Verlet position update for fluid particles.
/// This is a simplified version that doesn't handle all the advanced features.
//==============================================================================
__global__ void KerSimpleVerletUpdate(
  unsigned np,
  unsigned npb,
  const float3* ace,
  float3 gravity,
  double dt,
  double2* posxy,
  double* posz,
  float4* velrho,
  float rho0)
{
  const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
  if(p >= np) return;

  if(p < npb) {
    // Boundary particles: just ensure density doesn't drop below rho0
    float4 vr = velrho[p];
    if(vr.w < rho0) vr.w = rho0;
    velrho[p] = make_float4(0, 0, 0, vr.w);
  }
  else {
    // Fluid particles: update position and velocity
    float4 vr = velrho[p];
    float3 a = ace[p];

    // Add gravity to acceleration
    float ax = a.x + gravity.x;
    float ay = a.y + gravity.y;
    float az = a.z + gravity.z;

    // Update velocity: v_new = v_old + a * dt
    float vx = vr.x + ax * float(dt);
    float vy = vr.y + ay * float(dt);
    float vz = vr.z + az * float(dt);

    // Update position: x_new = x_old + v_new * dt
    double2 rposxy = posxy[p];
    double rposz = posz[p];
    rposxy.x += double(vx) * dt;
    rposxy.y += double(vy) * dt;
    rposz += double(vz) * dt;

    // Store results
    posxy[p] = rposxy;
    posz[p] = rposz;
    velrho[p] = make_float4(vx, vy, vz, vr.w);
  }
}

//==============================================================================
/// Simple Verlet position update.
//==============================================================================
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
  cudaStream_t stm)
{
  if(np > 0) {
    dim3 sgrid = GetGridSize(np, BSIZE);
    KerSimpleVerletUpdate<<<sgrid, BSIZE, 0, stm>>>(
      np, npb, ace, make_float3(gravity.x, gravity.y, gravity.z),
      dt, posxy, posz, velrho, rho0);
  }
}

//==============================================================================
/// CUDA kernel: Simple Symplectic predictor step.
//==============================================================================
__global__ void KerSimpleSymplecticPre(
  unsigned np,
  unsigned npb,
  const float3* ace,
  float3 gravity,
  double dtm,
  const double2* posxyPre,
  const double* poszPre,
  const float4* velrhoPre,
  double2* posxy,
  double* posz,
  float4* velrho,
  float rho0)
{
  const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
  if(p >= np) return;

  float4 vrPre = velrhoPre[p];

  if(p < npb) {
    // Boundary particles
    if(vrPre.w < rho0) vrPre.w = rho0;
    velrho[p] = make_float4(0, 0, 0, vrPre.w);
  }
  else {
    // Fluid particles
    float3 a = ace[p];
    float ax = a.x + gravity.x;
    float ay = a.y + gravity.y;
    float az = a.z + gravity.z;

    // Predictor velocity: v = v0 + a * dtm
    float vx = vrPre.x + ax * float(dtm);
    float vy = vrPre.y + ay * float(dtm);
    float vz = vrPre.z + az * float(dtm);

    // Predictor position: x = x0 + v0 * dtm
    double2 rposxyPre = posxyPre[p];
    double rposzPre = poszPre[p];
    double2 rposxy;
    rposxy.x = rposxyPre.x + double(vrPre.x) * dtm;
    rposxy.y = rposxyPre.y + double(vrPre.y) * dtm;
    double rposz = rposzPre + double(vrPre.z) * dtm;

    posxy[p] = rposxy;
    posz[p] = rposz;
    velrho[p] = make_float4(vx, vy, vz, vrPre.w);
  }
}

//==============================================================================
/// Simple Symplectic predictor step.
//==============================================================================
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
  cudaStream_t stm)
{
  if(np > 0) {
    dim3 sgrid = GetGridSize(np, BSIZE);
    KerSimpleSymplecticPre<<<sgrid, BSIZE, 0, stm>>>(
      np, npb, ace, make_float3(gravity.x, gravity.y, gravity.z),
      dtm, posxyPre, poszPre, velrhoPre, posxy, posz, velrho, rho0);
  }
}

//==============================================================================
/// CUDA kernel: Simple Symplectic corrector step.
//==============================================================================
__global__ void KerSimpleSymplecticCor(
  unsigned np,
  unsigned npb,
  const float3* ace,
  float3 gravity,
  double dtm,
  double dt,
  const double2* posxyPre,
  const double* poszPre,
  const float4* velrhoPre,
  double2* posxy,
  double* posz,
  float4* velrho,
  float rho0)
{
  const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
  if(p >= np) return;

  float4 vrPre = velrhoPre[p];

  if(p < npb) {
    // Boundary particles
    if(vrPre.w < rho0) vrPre.w = rho0;
    velrho[p] = make_float4(0, 0, 0, vrPre.w);
  }
  else {
    // Fluid particles
    float3 a = ace[p];
    float ax = a.x + gravity.x;
    float ay = a.y + gravity.y;
    float az = a.z + gravity.z;

    // Corrector velocity: v_new = v0 + a * dt
    float vx = vrPre.x + ax * float(dt);
    float vy = vrPre.y + ay * float(dt);
    float vz = vrPre.z + az * float(dt);

    // Corrector position: x_new = x0 + (v0 + v_new) * dtm
    double2 rposxyPre = posxyPre[p];
    double rposzPre = poszPre[p];
    double2 rposxy;
    rposxy.x = rposxyPre.x + (double(vrPre.x) + double(vx)) * dtm;
    rposxy.y = rposxyPre.y + (double(vrPre.y) + double(vy)) * dtm;
    double rposz = rposzPre + (double(vrPre.z) + double(vz)) * dtm;

    posxy[p] = rposxy;
    posz[p] = rposz;
    velrho[p] = make_float4(vx, vy, vz, vrPre.w);
  }
}

//==============================================================================
/// Simple Symplectic corrector step.
//==============================================================================
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
  cudaStream_t stm)
{
  if(np > 0) {
    dim3 sgrid = GetGridSize(np, BSIZE);
    KerSimpleSymplecticCor<<<sgrid, BSIZE, 0, stm>>>(
      np, npb, ace, make_float3(gravity.x, gravity.y, gravity.z),
      dtm, dt, posxyPre, poszPre, velrhoPre, posxy, posz, velrho, rho0);
  }
}

} // namespace dsphker
