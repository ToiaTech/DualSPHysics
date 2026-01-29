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

//==============================================================================
// Dynamic Boundary Transformation Kernels
//==============================================================================

//------------------------------------------------------------------------------
/// Device function: Rotate vector by quaternion.
/// q = (x,y,z,w) where w is the scalar component.
/// Formula: v' = v + 2*w*(qv x v) + 2*(qv x (qv x v))
/// where qv = (x,y,z)
//------------------------------------------------------------------------------
__device__ float3 QuatRotate(float4 q, float3 v) {
  // Extract quaternion vector part
  float3 qv = make_float3(q.x, q.y, q.z);
  float qw = q.w;

  // Cross product: qv x v
  float3 t;
  t.x = qv.y * v.z - qv.z * v.y;
  t.y = qv.z * v.x - qv.x * v.z;
  t.z = qv.x * v.y - qv.y * v.x;

  // t = 2 * t
  t.x *= 2.0f;
  t.y *= 2.0f;
  t.z *= 2.0f;

  // Cross product: qv x t
  float3 t2;
  t2.x = qv.y * t.z - qv.z * t.y;
  t2.y = qv.z * t.x - qv.x * t.z;
  t2.z = qv.x * t.y - qv.y * t.x;

  // Result: v + w*t + t2
  float3 result;
  result.x = v.x + qw * t.x + t2.x;
  result.y = v.y + qw * t.y + t2.y;
  result.z = v.z + qw * t.z + t2.z;

  return result;
}

//------------------------------------------------------------------------------
/// Device function: Cross product of two float3 vectors.
//------------------------------------------------------------------------------
__device__ float3 Cross(float3 a, float3 b) {
  return make_float3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

//------------------------------------------------------------------------------
/// CUDA kernel: Transform boundary particles from local to world coordinates.
//------------------------------------------------------------------------------
__global__ void KerTransformBoundaryParticles(
  unsigned int count,
  const float3* localPos,
  const float3* localNorm,
  float3 comPosition,
  float4 orientation,
  float3 linearVel,
  float3 angularVel,
  float3* worldPos,
  float3* worldNorm,
  float3* particleVel)
{
  const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
  if(p >= count) return;

  // Read local position
  float3 lpos = localPos[p];

  // Rotate local position by orientation quaternion
  float3 rotatedPos = QuatRotate(orientation, lpos);

  // Translate to world coordinates
  float3 wpos;
  wpos.x = rotatedPos.x + comPosition.x;
  wpos.y = rotatedPos.y + comPosition.y;
  wpos.z = rotatedPos.z + comPosition.z;
  worldPos[p] = wpos;

  // Transform normal if provided
  if(localNorm && worldNorm) {
    float3 lnorm = localNorm[p];
    worldNorm[p] = QuatRotate(orientation, lnorm);
  }

  // Compute particle velocity: v = v_linear + omega x r
  // where r is the rotated position (relative to CoM)
  float3 vel = Cross(angularVel, rotatedPos);
  vel.x += linearVel.x;
  vel.y += linearVel.y;
  vel.z += linearVel.z;
  particleVel[p] = vel;
}

//------------------------------------------------------------------------------
/// Transform boundary particles from local to world coordinates.
//------------------------------------------------------------------------------
void TransformBoundaryParticles(
  unsigned int count,
  const float3* localPos,
  const float3* localNorm,
  float3 comPosition,
  float4 orientation,
  float3 linearVel,
  float3 angularVel,
  float3* worldPos,
  float3* worldNorm,
  float3* particleVel,
  cudaStream_t stm)
{
  if(count > 0) {
    dim3 sgrid = GetGridSize(count, BSIZE);
    KerTransformBoundaryParticles<<<sgrid, BSIZE, 0, stm>>>(
      count, localPos, localNorm, comPosition, orientation,
      linearVel, angularVel, worldPos, worldNorm, particleVel);
  }
}

//------------------------------------------------------------------------------
/// CUDA kernel: Accumulate forces on boundary particles.
/// Uses parallel reduction to sum forces and torques.
//------------------------------------------------------------------------------
__global__ void KerAccumulateBoundaryForces(
  unsigned int particleCount,
  const float3* worldPos,
  float3 comPosition,
  const float3* ace,
  float particleMass,
  float3* outForce,
  float3* outTorque)
{
  // Shared memory for reduction
  __shared__ float sfx[BSIZE];
  __shared__ float sfy[BSIZE];
  __shared__ float sfz[BSIZE];
  __shared__ float stx[BSIZE];
  __shared__ float sty[BSIZE];
  __shared__ float stz[BSIZE];

  const unsigned int tid = threadIdx.x;
  const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize with zero
  float fx = 0, fy = 0, fz = 0;
  float tx = 0, ty = 0, tz = 0;

  if(p < particleCount) {
    // Force = mass * acceleration (from fluid on boundary)
    // Note: The acceleration in Aceg for boundary particles represents
    // the force exerted BY the fluid ON the boundary.
    // For the reaction force (boundary on fluid), we'd negate this.
    float3 a = ace[p];
    fx = particleMass * a.x;
    fy = particleMass * a.y;
    fz = particleMass * a.z;

    // Torque = r x F, where r is from CoM to particle
    float3 wpos = worldPos[p];
    float3 r;
    r.x = wpos.x - comPosition.x;
    r.y = wpos.y - comPosition.y;
    r.z = wpos.z - comPosition.z;

    float3 torque = Cross(r, make_float3(fx, fy, fz));
    tx = torque.x;
    ty = torque.y;
    tz = torque.z;
  }

  // Store to shared memory
  sfx[tid] = fx;
  sfy[tid] = fy;
  sfz[tid] = fz;
  stx[tid] = tx;
  sty[tid] = ty;
  stz[tid] = tz;
  __syncthreads();

  // Parallel reduction
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) {
      sfx[tid] += sfx[tid + s];
      sfy[tid] += sfy[tid + s];
      sfz[tid] += sfz[tid + s];
      stx[tid] += stx[tid + s];
      sty[tid] += sty[tid + s];
      stz[tid] += stz[tid + s];
    }
    __syncthreads();
  }

  // First thread of each block writes partial result
  if(tid == 0) {
    atomicAdd(&outForce->x, sfx[0]);
    atomicAdd(&outForce->y, sfy[0]);
    atomicAdd(&outForce->z, sfz[0]);
    atomicAdd(&outTorque->x, stx[0]);
    atomicAdd(&outTorque->y, sty[0]);
    atomicAdd(&outTorque->z, stz[0]);
  }
}

//------------------------------------------------------------------------------
/// Accumulate fluid forces acting on a boundary object.
//------------------------------------------------------------------------------
void AccumulateBoundaryForces(
  unsigned int particleStart,
  unsigned int particleCount,
  const float3* worldPos,
  float3 comPosition,
  const float3* ace,
  float particleMass,
  float3* outForce,
  float3* outTorque,
  cudaStream_t stm)
{
  if(particleCount > 0) {
    dim3 sgrid = GetGridSize(particleCount, BSIZE);
    KerAccumulateBoundaryForces<<<sgrid, BSIZE, 0, stm>>>(
      particleCount,
      worldPos + particleStart,
      comPosition,
      ace + particleStart,
      particleMass,
      outForce,
      outTorque);
  }
}

//------------------------------------------------------------------------------
/// CUDA kernel: Zero a float3 value.
//------------------------------------------------------------------------------
__global__ void KerZeroFloat3(float3* ptr) {
  ptr->x = 0.0f;
  ptr->y = 0.0f;
  ptr->z = 0.0f;
}

//------------------------------------------------------------------------------
/// Zero a float3 value on device.
//------------------------------------------------------------------------------
void ZeroFloat3(float3* ptr, cudaStream_t stm) {
  if(ptr) {
    KerZeroFloat3<<<1, 1, 0, stm>>>(ptr);
  }
}

//------------------------------------------------------------------------------
/// CUDA kernel: Zero an array of float3 values.
//------------------------------------------------------------------------------
__global__ void KerZeroFloat3Array(float3* ptr, unsigned int count) {
  const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
  if(p < count) {
    ptr[p] = make_float3(0.0f, 0.0f, 0.0f);
  }
}

//------------------------------------------------------------------------------
/// Zero an array of float3 values on device.
//------------------------------------------------------------------------------
void ZeroFloat3Array(float3* ptr, unsigned int count, cudaStream_t stm) {
  if(ptr && count > 0) {
    dim3 sgrid = GetGridSize(count, BSIZE);
    KerZeroFloat3Array<<<sgrid, BSIZE, 0, stm>>>(ptr, count);
  }
}

//==============================================================================
// Boundary-Fluid Force Computation Kernel
//==============================================================================

//------------------------------------------------------------------------------
/// Device function: Compute Wendland kernel value.
/// Returns W(r, h) for Wendland C2 kernel.
//------------------------------------------------------------------------------
__device__ float WendlandKernel(float r, float h) {
  const float q = r / h;
  if(q >= 2.0f) return 0.0f;

  // Wendland C2 kernel: (1 - q/2)^4 * (1 + 2q)
  // Normalization: 21 / (16 * pi * h^3) in 3D
  const float oneMinusHalfQ = 1.0f - 0.5f * q;
  const float kernel = oneMinusHalfQ * oneMinusHalfQ * oneMinusHalfQ * oneMinusHalfQ * (1.0f + 2.0f * q);

  // 3D normalization factor
  const float h3 = h * h * h;
  const float alpha = 21.0f / (16.0f * 3.14159265f * h3);

  return alpha * kernel;
}

//------------------------------------------------------------------------------
/// Device function: Compute Wendland kernel gradient magnitude.
/// Returns dW/dr for Wendland C2 kernel.
//------------------------------------------------------------------------------
__device__ float WendlandKernelGrad(float r, float h) {
  const float q = r / h;
  if(q >= 2.0f || q < 1e-8f) return 0.0f;

  // Derivative of Wendland C2: d/dr[(1-q/2)^4*(1+2q)]
  // = -5q(1-q/2)^3 / h
  const float oneMinusHalfQ = 1.0f - 0.5f * q;
  const float gradKernel = -5.0f * q * oneMinusHalfQ * oneMinusHalfQ * oneMinusHalfQ / h;

  // 3D normalization factor
  const float h3 = h * h * h;
  const float alpha = 21.0f / (16.0f * 3.14159265f * h3);

  return alpha * gradKernel;
}

//------------------------------------------------------------------------------
/// Device function: Compute pressure from density using Tait equation.
//------------------------------------------------------------------------------
__device__ float ComputePressure(float rho, float rho0, float cs0, float gamma) {
  const float B = rho0 * cs0 * cs0 / gamma;
  float pressure = B * (powf(rho / rho0, gamma) - 1.0f);
  return fmaxf(pressure, 0.0f);  // Clamp negative pressures
}

//------------------------------------------------------------------------------
/// Device function: Decode cell coordinates from cell code.
//------------------------------------------------------------------------------
__device__ void DecodeCell(unsigned int cellCode, unsigned int cell,
                           unsigned int& cx, unsigned int& cy, unsigned int& cz) {
  // Extract bit widths from cell code
  const unsigned int bx = cellCode & 0x1F;
  const unsigned int by = (cellCode >> 5) & 0x1F;
  // const unsigned int bz = (cellCode >> 10) & 0x1F;  // Not needed

  cx = cell & ((1u << bx) - 1);
  cy = (cell >> bx) & ((1u << by) - 1);
  cz = cell >> (bx + by);
}

//------------------------------------------------------------------------------
/// Device function: Encode cell coordinates to cell index.
//------------------------------------------------------------------------------
__device__ unsigned int EncodeCell(unsigned int cellCode,
                                    unsigned int cx, unsigned int cy, unsigned int cz) {
  const unsigned int bx = cellCode & 0x1F;
  const unsigned int by = (cellCode >> 5) & 0x1F;
  return cx | (cy << bx) | (cz << (bx + by));
}

//------------------------------------------------------------------------------
/// CUDA kernel: Compute fluid pressure forces on boundary particles.
/// Searches for nearby fluid particles and accumulates pressure forces.
//------------------------------------------------------------------------------
__global__ void KerComputeBoundaryFluidForces(
  unsigned int boundaryCount,
  const float3* boundWorldPos,
  const float3* boundWorldNorm,
  float3 comPosition,
  unsigned int np,
  unsigned int npb,
  const double2* fluidPosxy,
  const double* fluidPosz,
  const float4* fluidVelrho,
  const int* cellBegin,
  unsigned int cellCode,
  double3 cellPosMin,
  float cellSize,
  float kernelH,
  float kernelSize,
  float massFluid,
  float massBound,
  float rho0,
  float cs0,
  float gamma,
  float3* outForce,
  float3* outTorque)
{
  // Shared memory for reduction
  __shared__ float sfx[BSIZE];
  __shared__ float sfy[BSIZE];
  __shared__ float sfz[BSIZE];
  __shared__ float stx[BSIZE];
  __shared__ float sty[BSIZE];
  __shared__ float stz[BSIZE];

  const unsigned int tid = threadIdx.x;
  const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;

  float fx = 0, fy = 0, fz = 0;
  float tx = 0, ty = 0, tz = 0;

  if(p < boundaryCount) {
    // Get boundary particle position
    float3 bpos = boundWorldPos[p];

    // Compute boundary particle pressure (assume reference density)
    const float pressB = ComputePressure(rho0, rho0, cs0, gamma);
    const float rho2B = rho0 * rho0;

    // Compute cell for this boundary particle
    double dx = double(bpos.x) - cellPosMin.x;
    double dy = double(bpos.y) - cellPosMin.y;
    double dz = double(bpos.z) - cellPosMin.z;

    int cellX = int(dx / cellSize);
    int cellY = int(dy / cellSize);
    int cellZ = int(dz / cellSize);

    // Extract cell grid dimensions from cell code
    const unsigned int bx = cellCode & 0x1F;
    const unsigned int by = (cellCode >> 5) & 0x1F;
    const unsigned int bz = (cellCode >> 10) & 0x1F;
    const int ncx = 1 << bx;
    const int ncy = 1 << by;
    const int ncz = 1 << bz;

    // Search neighboring cells (3x3x3)
    for(int cz = cellZ - 1; cz <= cellZ + 1; cz++) {
      if(cz < 0 || cz >= ncz) continue;
      for(int cy = cellY - 1; cy <= cellY + 1; cy++) {
        if(cy < 0 || cy >= ncy) continue;
        for(int cx = cellX - 1; cx <= cellX + 1; cx++) {
          if(cx < 0 || cx >= ncx) continue;

          // Get cell index
          unsigned int cellIdx = cx | (cy << bx) | (cz << (bx + by));

          // Get particle range for this cell
          int cellStart = cellBegin[cellIdx];
          int cellEnd = cellBegin[cellIdx + 1];

          // Iterate over fluid particles in cell
          for(int j = cellStart; j < cellEnd; j++) {
            // Skip boundary particles (first npb particles)
            if(j < (int)npb) continue;

            // Get fluid particle position
            double2 fposxy = fluidPosxy[j];
            double fposz = fluidPosz[j];
            float4 fvelrho = fluidVelrho[j];

            // Compute distance vector (boundary - fluid)
            float drx = bpos.x - float(fposxy.x);
            float dry = bpos.y - float(fposxy.y);
            float drz = bpos.z - float(fposz);
            float r2 = drx*drx + dry*dry + drz*drz;
            float r = sqrtf(r2);

            // Check if within kernel support
            if(r < kernelSize && r > 1e-8f) {
              // Compute fluid pressure
              float rhoF = fvelrho.w;
              float pressF = ComputePressure(rhoF, rho0, cs0, gamma);
              float rho2F = rhoF * rhoF;

              // Compute kernel gradient
              float gradW = WendlandKernelGrad(r, kernelH);

              // Direction vector (normalized)
              float invR = 1.0f / r;
              float nx = drx * invR;
              float ny = dry * invR;
              float nz = drz * invR;

              // SPH pressure force: F = -m_b * m_f * (p_f/rho_f^2 + p_b/rho_b^2) * gradW * n
              float forceMag = -massBound * massFluid * (pressF / rho2F + pressB / rho2B) * gradW;

              fx += forceMag * nx;
              fy += forceMag * ny;
              fz += forceMag * nz;
            }
          }
        }
      }
    }

    // Compute torque: tau = r x F
    float3 r;
    r.x = bpos.x - comPosition.x;
    r.y = bpos.y - comPosition.y;
    r.z = bpos.z - comPosition.z;

    tx = r.y * fz - r.z * fy;
    ty = r.z * fx - r.x * fz;
    tz = r.x * fy - r.y * fx;
  }

  // Store to shared memory
  sfx[tid] = fx;
  sfy[tid] = fy;
  sfz[tid] = fz;
  stx[tid] = tx;
  sty[tid] = ty;
  stz[tid] = tz;
  __syncthreads();

  // Parallel reduction
  for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if(tid < s) {
      sfx[tid] += sfx[tid + s];
      sfy[tid] += sfy[tid + s];
      sfz[tid] += sfz[tid + s];
      stx[tid] += stx[tid + s];
      sty[tid] += sty[tid + s];
      stz[tid] += stz[tid + s];
    }
    __syncthreads();
  }

  // First thread of each block writes partial result with atomics
  if(tid == 0) {
    atomicAdd(&outForce->x, sfx[0]);
    atomicAdd(&outForce->y, sfy[0]);
    atomicAdd(&outForce->z, sfz[0]);
    atomicAdd(&outTorque->x, stx[0]);
    atomicAdd(&outTorque->y, sty[0]);
    atomicAdd(&outTorque->z, stz[0]);
  }
}

//------------------------------------------------------------------------------
/// Compute fluid pressure forces on boundary particles.
//------------------------------------------------------------------------------
void ComputeBoundaryFluidForces(
  unsigned int boundaryCount,
  const float3* boundWorldPos,
  const float3* boundWorldNorm,
  float3 comPosition,
  unsigned int np,
  unsigned int npb,
  const double2* fluidPosxy,
  const double* fluidPosz,
  const float4* fluidVelrho,
  const int* cellBegin,
  unsigned int cellCode,
  double3 cellPosMin,
  float cellSize,
  float kernelH,
  float kernelSize,
  float massFluid,
  float massBound,
  float rho0,
  float cs0,
  float gamma,
  float3* outForce,
  float3* outTorque,
  cudaStream_t stm)
{
  if(boundaryCount > 0) {
    dim3 sgrid = GetGridSize(boundaryCount, BSIZE);
    KerComputeBoundaryFluidForces<<<sgrid, BSIZE, 0, stm>>>(
      boundaryCount, boundWorldPos, boundWorldNorm, comPosition,
      np, npb, fluidPosxy, fluidPosz, fluidVelrho,
      cellBegin, cellCode, cellPosMin, cellSize,
      kernelH, kernelSize, massFluid, massBound, rho0, cs0, gamma,
      outForce, outTorque);
  }
}

} // namespace dsphker
