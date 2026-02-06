# DualSPHysics v5.4 - SPH Solver Architecture Analysis

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Core SPH Formulation](#3-core-sph-formulation)
4. [Kernel Functions](#4-kernel-functions)
5. [Equation of State](#5-equation-of-state)
6. [Particle Interaction and Force Computation](#6-particle-interaction-and-force-computation)
7. [Neighbor Search and Spatial Partitioning](#7-neighbor-search-and-spatial-partitioning)
8. [Time Integration Schemes](#8-time-integration-schemes)
9. [Adaptive Time Stepping](#9-adaptive-time-stepping)
10. [Boundary Conditions](#10-boundary-conditions)
11. [Particle Shifting](#11-particle-shifting)
12. [Density Diffusion Terms](#12-density-diffusion-terms)
13. [Viscosity Models](#13-viscosity-models)
14. [Floating Bodies and Rigid Dynamics](#14-floating-bodies-and-rigid-dynamics)
15. [Flexible Structures](#15-flexible-structures)
16. [Inlet/Outlet Conditions](#16-inletoutlet-conditions)
17. [GPU Implementation](#17-gpu-implementation)
18. [CPU vs GPU Architecture Comparison](#18-cpu-vs-gpu-architecture-comparison)
19. [Class Hierarchy and Design Patterns](#19-class-hierarchy-and-design-patterns)
20. [Key Data Structures](#20-key-data-structures)
21. [Performance Characteristics](#21-performance-characteristics)
22. [Build System and Compilation](#22-build-system-and-compilation)

---

## 1. Project Overview

DualSPHysics is a Smoothed Particle Hydrodynamics (SPH) solver for free-surface fluid dynamics, developed by EPHYSLAB (University of Vigo) and the University of Manchester. Version 5.4 implements a weakly-compressible SPH (WCSPH) method with dual CPU/GPU execution backends.

**Primary applications:**
- Coastal and offshore engineering (wave-structure interaction)
- Dam break and flood simulations
- Multi-phase flows (liquid-gas, non-Newtonian)
- Fluid-structure interaction with rigid and flexible bodies

**Codebase statistics:**
- ~220 C++ source files (.cpp)
- ~326 header files (.h)
- ~28 CUDA kernel files (.cu)
- **574 total source files** across the main solver and multi-phase variant

---

## 2. Project Structure

```
DualSPHysics/
├── src/
│   ├── source/                    # Main solver source code (315 files)
│   │   ├── JSph.h/cpp             # Base SPH solver class
│   │   ├── JSphCpu.h/cpp          # CPU implementation
│   │   ├── JSphGpu.h/cpp          # GPU implementation
│   │   ├── JSphCpuSingle.h/cpp    # Single-CPU simulation loop
│   │   ├── JSphGpuSingle.h/cpp    # Single-GPU simulation loop
│   │   ├── JSphGpu_ker.cu         # GPU force computation kernels
│   │   ├── JSphGpuSimple_ker.cu   # GPU particle update kernels
│   │   ├── JCellDiv*.h/cpp        # Neighbor search (CPU & GPU)
│   │   ├── FunSphKernel.h         # SPH kernel functions
│   │   ├── FunSphEos.h            # Equation of state
│   │   ├── DualSphDef.h           # Core definitions and types
│   │   ├── Source_DSphMoorDynPlus/ # MoorDyn mooring library
│   │   ├── CMakeLists.txt          # CMake build
│   │   └── Makefile / Makefile_cpu # GNU Make build
│   ├── lib/                       # Pre-compiled libraries
│   │   ├── linux_gcc/
│   │   └── vs2022/
│   └── VS/                        # Visual Studio project files
├── src_mphase/                    # Multi-phase / Non-Newtonian variant
│   └── DSPH_v5.0_NNewtonian/
├── bin/                           # Pre-compiled binaries
│   ├── linux/                     # GenCase, PartVTK, MeasureTool, etc.
│   └── windows/
├── doc/                           # Documentation
│   ├── help/                      # Tool help files
│   └── xml_format/                # XML case format definitions
├── examples/                      # Example simulation cases
├── README.md
├── LICENSE                        # LGPL
└── CHANGES.txt                    # Version history
```

---

## 3. Core SPH Formulation

DualSPHysics implements the **weakly-compressible SPH (WCSPH)** method. In this approach, the fluid is treated as slightly compressible, with pressure computed from density via an equation of state rather than solving a Poisson equation.

### Governing Equations

The Navier-Stokes equations in SPH form:

**Momentum equation (acceleration of particle a):**
```
dv_a/dt = -sum_b m_b * (P_a/rho_a^2 + P_b/rho_b^2) * grad(W_ab) + viscosity + gravity
```

**Continuity equation (density rate of change):**
```
drho_a/dt = sum_b m_b * (v_a - v_b) . grad(W_ab)
```

Where:
- `W_ab = W(|r_a - r_b|, h)` is the SPH kernel function
- `h` is the smoothing length (`KernelH`)
- `m_b` is the mass of neighbor particle b
- `P` is pressure, `rho` is density, `v` is velocity

### Particle Types

Defined in `DualSphDef.h` via bitfield encoding in `typecode`:

| Type | Code Macro | Description |
|------|-----------|-------------|
| Fixed | `CODE_TYPE_FIXED` | Static boundary particles |
| Moving | `CODE_TYPE_MOVING` | Prescribed-motion boundary particles |
| Floating | `CODE_TYPE_FLOATING` | Free-moving rigid body particles |
| Fluid | `CODE_TYPE_FLUID` | Fluid particles (including inlet/outlet) |

Particles also carry special flags: `CODE_NORMAL`, `CODE_PERIODIC`, `CODE_OUTPOS`, `CODE_OUTRHO`, `CODE_OUTMOV`.

---

## 4. Kernel Functions

**File:** `src/source/FunSphKernel.h`, `src/source/FunSphKernelDef.h`

DualSPHysics supports two SPH kernel functions, selected via the `TpKernel` enum:

### Wendland C2 Kernel (default, `KERNEL_Wendland`)

```
W(q) = awen * (2q + 1) * (1 - q/2)^4     for 0 <= q <= 2
```

Gradient factor:
```
fac = bwenh * (1 - q/2)^3
```

Constants (3D):
- `awen = 0.41778 / h^3`
- `bwenh = -2.08891 / h^5`

Constants (2D):
- `awen = 0.557 / h^2`
- `bwenh = -2.7852 / h^4`

Where `q = r/h` (normalized distance).

### Cubic Spline Kernel (`KERNEL_Cubic`)

```
W(q) = a2 * (1 + 0.75q - 1.5q^2)         for 0 <= q <= 1
W(q) = a24 * (2 - q)^3                     for 1 < q <= 2
```

Gradient factor:
```
fac = (c1*q + d1*q^2) / r                  for 0 <= q <= 1
fac = c2 * (2-q)^2 / r                     for 1 < q <= 2
```

### Kernel Support and Interaction Distance

- **Kernel factor:** 2.0 for both kernels
- **Kernel size (support radius):** `KernelSize = 2h`
- **Maximum interaction distance:** `KernelSize^2` (squared, used in distance checks)

### Template-Based Dispatch

Kernels are dispatched via C++ templates to avoid runtime branching:

```cpp
template<TpKernel tker> inline float GetKernel_Wab(const StCteSph& csp, float rr2);
template<TpKernel tker> inline float GetKernel_Fac(const StCteSph& csp, float rr2);
template<TpKernel tker> inline float GetKernel_WabFac(const StCteSph& csp, float rr2, float& fac);
```

The Wendland kernel also supports a second derivative (`WabFacFacc`) used in advanced formulations.

---

## 5. Equation of State

**File:** `src/source/FunSphEos.h`

Pressure is computed from density using the Tait equation of state (Monaghan, 1994):

```
P = B * ((rho/rho_0)^gamma - 1)
```

Where:
- `B = CteB` - equation of state constant (related to speed of sound)
- `rho_0 = RhopZero` - reference density (typically 1000 kg/m^3 for water)
- `gamma` - polytropic constant (typically 7 for water)

The speed of sound `Cs0` is chosen artificially high (typically 10x maximum fluid velocity) to limit density variations to ~1%, maintaining the weakly-compressible assumption.

---

## 6. Particle Interaction and Force Computation

**Files:** `src/source/JSphCpu.h/cpp`, `src/source/JSphGpu_ker.cu`

### Force Computation Pipeline

The force computation occurs in the `Interaction_Forces()` method chain, which is heavily templated on:

| Template Parameter | Options | Purpose |
|---|---|---|
| `TpKernel tker` | Cubic, Wendland | Kernel function selection |
| `TpFtMode ftmode` | None, Sph, Ext | Floating body interaction mode |
| `TpVisco tvisco` | Artificial, Laminar+SPS | Viscosity formulation |
| `TpDensity tdensity` | None, DDT, DDT2, DDT2Full | Density diffusion |
| `bool shift` | true/false | Particle shifting |
| `TpMdbc2Mode mdbc2` | None, Std, NoPen | Boundary condition variant |

### Interaction Structure (CPU)

The `stinterparmsc` structure bundles all data needed for particle interactions:

```cpp
typedef struct {
  unsigned np, npb, npbok, npf;    // Particle counts
  StDivDataCpu divdata;            // Cell division data
  const unsigned* dcell;           // Cell assignments
  const tdouble3* pos;             // Positions
  const tfloat4* velrho;           // Velocity + density
  const typecode* code;            // Particle type codes
  const float* press;              // Pressure
  float* ar;                       // Output: density rate
  tfloat3* ace;                    // Output: acceleration
  float* delta;                    // Output: density diffusion
  tfloat4* shiftposfs;             // Output: shifting displacement
  tsymatrix3f* spstaurho2;        // SPS stress tensor
  // ... additional fields for mDBC, shifting, divergence cleaning
} stinterparmsc;
```

### Force Loop Algorithm

For each particle `p1`, the algorithm:

1. Retrieves `p1`'s position, velocity, density, and pressure
2. Iterates over the 27 neighboring cells (3x3x3 stencil)
3. For each neighbor `p2` in those cells:
   - Computes squared distance `rr2 = |r1 - r2|^2`
   - If `rr2 <= KernelSize^2`: computes kernel value `wab` and gradient `fac`
   - Accumulates pressure forces: `ace += m_b * (P_a/rho_a^2 + P_b/rho_b^2) * fac * dr`
   - Accumulates density rate: `ar += m_b * (v_a - v_b) . (fac * dr)`
   - Applies viscosity, density diffusion, and shifting as configured

### Two-Phase Processing

Forces are computed in two passes:
1. **Boundary interactions** (`InteractionForcesBound`): Boundary particles accumulate density rate from fluid neighbors
2. **Fluid interactions** (`InteractionForcesFluid`): Fluid particles accumulate acceleration, density rate, and optional quantities from all neighbors

---

## 7. Neighbor Search and Spatial Partitioning

**Files:** `src/source/JCellDivCpu.h/cpp`, `src/source/JCellDivCpuSingle.cpp`, `src/source/JCellDivGpu_ker.cu`

### Cell-Linked List Method

The domain is divided into a uniform grid of cubic cells with size:
```
Scell = KernelSize / ScellDiv
```
Where `ScellDiv` is 1 (full cells) or 2 (half cells).

### Algorithm: `Divide()` Operation

**Step 1 - Domain Sizing:**
```
CalcCellDomain():
  For boundary particles: find min/max occupied cells
  For fluid particles: find min/max occupied cells + kernel halo
  CellDomainMin/Max = merged bounding box
  Ncx, Ncy, Ncz = cell counts per axis
  Nct = Ncx * Ncy * Ncz  (total active cells)
```

**Step 2 - Particle Classification (PreSort):**
```
For each particle p:
  Extract cell (cx, cy, cz) from Dcell[p]
  Transform to local index relative to CellDomainMin
  Classify into box:
    boundary + valid       -> box = cell_index         [0, Nct)
    boundary + ignored     -> box = BoxBoundIgnore
    fluid + normal         -> box = BoxFluid + cell_index
    excluded               -> box = BoxBoundOut/FluidOut
  Count: PartsInCell[box]++
```

**Step 3 - Index Construction (MakeSort):**
```
Compute prefix sums: BeginCell[box] = cumulative particle count
Build SortPart[]: maps sorted position -> original particle index
```

**Step 4 - Particle Reordering:**
All particle arrays (Pos, Velrho, Code, Idp, etc.) are reordered so that particles in the same cell occupy contiguous memory.

### Memory Layout After Divide

```
BeginCell layout:
[BoundOk(nct cells)][BoundIgnore(1)][Fluid(nct cells)][BoundOut(1)][FluidOut(1)]
```

### Neighbor Search During Interaction

```
For particle p1 at cell (cx, cy, cz):
  For each cell (cx±1, cy±1, cz±1):   // 27 neighbor cells
    pini = BeginCell[cell]
    pfin = BeginCell[cell + 1]
    For p2 in [pini, pfin):
      if dist(p1, p2)^2 <= KernelSize^2:
        compute interaction
```

**Complexity:** O(N) for the divide, O(N * k_avg) for interactions where k_avg is the average number of neighbors (~50 in 3D for the Wendland kernel).

---

## 8. Time Integration Schemes

**Files:** `src/source/JSphCpu.cpp` (lines ~1638-2029), `src/source/JSphGpuSimple_ker.cu`

### Verlet Scheme (`STEP_Verlet`)

A multi-step Verlet integrator with periodic resynchronization:

```
Every VerletSteps (typically 40):
  Standard Verlet:
    v_new = v_old + a * dt
    rho_new = rho_old + (drho/dt) * dt
    x_new = x_old + v * dt + 0.5 * a * dt^2

  Resynchronization step (VerletStep == VerletSteps):
    Uses VelrhoM1 (previous step values) for leapfrog correction
    Resets VelrhoM1 <-> Velrho swap
```

### Symplectic (Velocity-Verlet) Scheme (`STEP_Symplectic`)

A two-stage predictor-corrector with superior stability:

**Predictor (half-step):**
```
ComputeSymplecticPre(dt):
  Save state: PosPre = Pos, VelrhoPre = Velrho

  Boundary particles:
    rho(t+dt/2) = rho(t) + 0.5 * dt * (drho/dt)

  Fluid particles:
    rho(t+dt/2) = rho(t) + 0.5 * dt * (drho/dt)
    v(t+dt/2) = v(t) + 0.5 * dt * (a + g)
    x(t+dt/2) = x(t) + v(t) * dt/2
```

**Cell re-division** occurs between predictor and corrector to update neighbor lists.

**Corrector (full-step):**
```
ComputeSymplecticCorr(dt):
  Boundary particles:
    rho(t+dt) = rho(t) * (2 - epsilon) / (2 + epsilon)
    where epsilon = -(drho/dt) / rho * dt   [implicit mass conservation]

  Fluid particles:
    v(t+dt) = v(t) + dt * (a + g)
    x(t+dt) = x(t) + 0.5 * dt * (v(t) + v(t+dt))   [trapezoidal rule]
```

The Symplectic scheme is more stable at higher CFL numbers (0.4-0.5) than Verlet.

---

## 9. Adaptive Time Stepping

**File:** `src/source/JSphCpu.cpp` (`DtVariable()`)

The time step is computed as:

```
dt = CFLnumber * min(dt_force, dt_courant, dt_flexstruc, dt_divclean)
```

Where:

| Constraint | Formula | Physical Meaning |
|---|---|---|
| `dt_force` | `sqrt(h / a_max)` | CFL on acceleration |
| `dt_courant` | `h / (Cs0 + max(10*v_max, viscdt*h))` | Courant + viscous condition |
| `dt_flexstruc` | `h / c_s_struct` | Structural wave speed |
| `dt_divclean` | `h / c_psi_max` | Divergence cleaning wave speed |

- `CFLnumber` is typically 0.2-0.5
- `CoefDtMin` provides a minimum dt floor: `dtmin = CoefDtMin * h / Cs0`
- The selected dt is clamped: `max(DtMin, dt)`

---

## 10. Boundary Conditions

### Dynamic Boundary Condition (DBC)

The original SPH boundary method: boundary particles satisfy the same equations as fluid particles but their positions are prescribed. They repel fluid particles through pressure forces when approached.

### Modified Dynamic Boundary Condition (mDBC, `BC_MDBC`)

**File:** `src/source/JSphCpu_mdbc.cpp`

A significantly improved boundary treatment that extrapolates fluid properties to ghost nodes:

**Algorithm:**

1. **Ghost node placement:** `ghost_pos = boundary_pos + boundary_normal`
2. **Fluid property sampling:** Compute kernel-weighted averages around ghost node
3. **Matrix construction** (4x4 in 3D, 3x3 in 2D):
   ```
   A = [sum(W*V),     sum(drx*W*V),   sum(dry*W*V),   sum(drz*W*V)  ]
       [sum(Gx*V),    sum(drx*Gx*V),  sum(dry*Gx*V),  sum(drz*Gx*V) ]
       [sum(Gy*V),    sum(drx*Gy*V),  sum(dry*Gy*V),  sum(drz*Gy*V) ]
       [sum(Gz*V),    sum(drx*Gz*V),  sum(dry*Gz*V),  sum(drz*Gz*V) ]
   ```
4. **Density extrapolation:**
   - If `|det(A)| >= 1e-3`: First-order Taylor expansion via matrix inverse
   - If `|det(A)| < 1e-3`: Zeroth-order Shepard interpolation (fallback)
5. **Velocity conditions:** Assigned based on slip mode

### mDBC2 (Advanced, `MDBC2_Std` / `MDBC2_NoPen`)

Enhanced variant with:

- **Submersion detection:** Checks if ghost node is actually in fluid using volume integral
- **Pressure cloning:** Accounts for hydrostatic gradient and boundary acceleration
- **Condition number monitoring:** Uses infinity norm to assess matrix quality; switches to Shepard interpolation when `cond > 50`
- **Slip modes:**
  - `SLIP_Vel0`: Original DBC (velocity = 0 at wall)
  - `SLIP_NoSlip`: `v_boundary = 2 * v_motion - v_ghost`
  - `SLIP_FreeSlip`: `v_boundary = v_ghost`
- **No-penetration** (`MDBC2_NoPen`): Additional velocity correction to prevent particle leakage through boundaries

### Periodic Boundary Conditions

Implemented via particle duplication at periodic boundaries. Controlled by `PeriActive` flags on X/Y/Z axes. Periodic "halo" particles are created within `KernelSize` of domain edges.

---

## 11. Particle Shifting

**Files:** `src/source/JSphShifting.h/cpp`, `src/source/JSphCpu_preloop.cpp`

### Standard Shifting

Prevents particle clustering and void formation by displacing particles along concentration gradients:

```
shift_magnitude = dt * ShiftCoef * KernelH * |velocity|
shift_direction = concentration_gradient
max_shift = 0.1 * Dp   (limited to 10% of initial spacing)
```

**Free surface correction:** Near free surfaces, shift magnitude is reduced using a threshold `ShiftTFS` based on kernel summation.

**Modes:**
- `SHIFT_NoBound`: No shifting near boundaries
- `SHIFT_NoFixed`: No shifting near fixed boundaries
- `SHIFT_Full`: All particles shifted

### Advanced Shifting (`SHIFT_FS`)

Implemented in `JSphCpu_preloop.cpp` with enhanced free-surface detection:

1. **Free surface identification:** Particles are classified using a divergence-based criterion
2. **Normal computation:** Surface normals computed from kernel gradient
3. **Umbrella region scan:** Identifies particles in the vicinity of the free surface
4. **Velocity-based shifting:** Displacement computed from a pre-loop interaction pass

---

## 12. Density Diffusion Terms

**Defined in:** `DualSphDef.h`

Stabilizes pressure field by diffusing density oscillations:

| Mode | Reference | Application |
|---|---|---|
| `DDT_None` | - | No diffusion |
| `DDT_DDT` | Molteni & Colagrossi 2009 | Inner fluid only |
| `DDT_DDT2` | Fourtakas et al 2019 | Inner fluid only |
| `DDT_DDT2Full` | Fourtakas et al 2019 | All fluid particles |

Key constants:
- `DDTkh = DDTValue * KernelSize` (DDTValue typically 0.1)
- `DDTgz = RhopZero * gravity.z / CteB` (for hydrostatic correction in DDT2)

DDT2 includes a hydrostatic correction term that improves pressure at boundaries, making it compatible with mDBC.

---

## 13. Viscosity Models

### Artificial Viscosity (`VISCO_Artificial`)

Classic Monaghan artificial viscosity:
```
Pi_ab = -alpha * h * c_s * (v_ab . r_ab) / (r_ab^2 + eta^2)   when v_ab . r_ab < 0
Pi_ab = 0                                                        otherwise
```

Where `eta^2 = (0.1h)^2` prevents singularity.

### Laminar + SPS Model (`VISCO_LaminarSPS`)

Combines laminar viscosity with Sub-Particle Scale (SPS) turbulence model:

- **Laminar:** Standard viscous diffusion
- **SPS:** Smagorinsky-type eddy viscosity
  - `SpsSmag`: Smagorinsky constant
  - `SpsBlin`: Blin constant for SPS
  - Computed via strain rate tensor `Sps2Strain` and stored in `SpsTauRho2` (stress tensor / rho^2)

The `ViscoBoundFactor` parameter allows different viscosity at boundaries (e.g., `Visco * ViscoBoundFactor`).

---

## 14. Floating Bodies and Rigid Dynamics

**Defined in:** `DualSphDef.h` (`StFloatingData`), managed in `JSph.h`

### Floating Body Properties

Each floating object stores:
```cpp
struct StFloatingData {
  word mkbound;           // MK identifier
  unsigned begin, count;  // Particle range
  float mass, massp;      // Total and per-particle mass
  float radius;           // Maximum particle-center distance
  byte constraints;       // Translation/rotation DOF constraints
  tdouble3 center;        // Center of mass
  tfloat3 fvel, fomega;   // Linear and angular velocity
  tfloat3 facelin, faceang; // Accelerations
  tmatrix3f inertiaini;   // Inertia tensor
  bool usechrono;         // Chrono library integration flag
};
```

### Rigid Algorithm Options (`TpRigidMode`)

| Mode | Description |
|---|---|
| `FTRIGID_Free` | Collision-free (no floating-boundary interaction) |
| `FTRIGID_Sph` | SPH-based collision (forces from particle pressure) |
| `FTRIGID_Dem` | Discrete Element Method contacts |
| `FTRIGID_Chrono` | Chrono Engine multi-body dynamics |

### Motion Constraints

6-DOF constraints via bitmask (`TpFtConstrains`):
- `FTCON_MoveX/Y/Z`: Lock translation axes
- `FTCON_RotateX/Y/Z`: Lock rotation axes

### External Forces

Floating bodies can receive:
- Imposed linear/angular velocities (`FtLinearVel`, `FtAngularVel`)
- External forces/torques (`FtLinearForce`, `FtAngularForce`)
- Mooring forces via MoorDyn+ library
- Chrono Engine forces for complex multi-body systems

---

## 15. Flexible Structures

**File:** `src/source/JSphFlexStruc.h`, CPU implementation in `JSphCpu.h`

### Continuum Mechanics Approach

Flexible structures are discretized as SPH particles with continuum solid mechanics:

**Material properties:**
- Young's modulus E
- Poisson's ratio nu
- Material density
- Constitutive model: Plane Strain, Plane Stress, or St. Venant-Kirchhoff

**Computational pipeline:**
1. **Kernel correction** (`CalcFlexStrucKerCorr`): Corrects SPH gradient approximation for boundary consistency
2. **Deformation gradient** (`CalcFlexStrucDefGrad`): F = I + sum(u_j (x) nabla W)
3. **First Piola-Kirchhoff stress** (`CalcFlexStrucPK1Stress`): P = F * S via constitutive matrix
4. **Force computation** (`InteractionForcesFlexStruc`): SPH force from stress divergence
5. **Time integration**: Semi-implicit Euler or Symplectic

**Hourglass correction** (`HgFactor`) prevents zero-energy deformation modes (checkerboard patterns).

**Clamping:** Multiple `MkClamp` zones can fix or prescribe motion at structure supports. Up to `MAX_NUM_MKCLAMP = 8` clamps per structure.

---

## 16. Inlet/Outlet Conditions

**File:** `src/source/JSphInOut.h`

### Open Boundary Implementation

Inlet/outlet zones enable continuous flow through the domain:

**Configuration per zone:**
- Plane definition and width
- Flow direction vector
- Velocity mode: analytical, interpolated from mesh, or special profiles (parabolic, logarithmic)
- Density mode: analytical (hydrostatic) or extrapolated from fluid

**Particle management:**
1. **Identification:** Particles in inlet/outlet zones classified by zone index
2. **Property assignment:** Velocity and density set from analytical or extrapolated values
3. **Surface tracking:** `Zsurf` (surface elevation) can be constant, variable, or computed
4. **Particle creation:** New particles injected at inlet when space becomes available
5. **Particle removal:** Outlet particles beyond boundary plane are removed (code set to dead)

**Refilling:**
- `UseRefillAdvanced`: Optimized injection every N steps (default 10)
- Point-based refilling across inlet cross-section
- Memory managed with over-allocation: `NpResizePlus0/1` for dynamic particle count growth

---

## 17. GPU Implementation

**Files:** `src/source/JSphGpu.h/cpp`, `src/source/JSphGpuSingle.h/cpp`, `src/source/JSphGpu_ker.cu`

### CUDA Architecture

**Thread-to-particle mapping:**
```cuda
const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
if(p < n) { /* process particle p */ }
```

**Block size optimization:**
- Force kernels use `cudaOccupancyMaxPotentialBlockSize()` for dynamic sizing
- Update kernels use fixed `SPHBSIZE = 256`
- Grid: `ceil(N / blockSize)` blocks

### Constant Memory

All SPH parameters stored in GPU constant memory (64 KB limit):

```cuda
__constant__ StCteInteraction CTE;
```

Contents: `kernelh`, `kernelsize2`, `massf`, `massb`, `rhopzero`, `gamma`, `cs0`, kernel constants (`awen`, `bwenh`), domain mapping parameters, periodic increments.

### GPU Memory Layout

```
Device Memory:
├── Particle Arrays (resizable, ~GpuParticlesSize per array):
│   ├── Posxy_g (double2) + Posz_g (double)  -- Positions split for coalescing
│   ├── PosCell_g (float4)                    -- Relative pos + cell encoding
│   ├── Velrho_g (float4)                     -- Velocity + density packed
│   ├── Code_g, Idp_g, Dcell_g               -- Classification
│   ├── Ace_g (float3), Ar_g (float)          -- Force outputs
│   ├── ViscDt_g, Delta_g, ShiftPosfs_g       -- Auxiliary outputs
│   └── [Optional: SPS, BoundNor, MotionVel, ShiftVel, PsiClean]
├── Cell Data:
│   └── BeginEndCell (int2[])                 -- Particle range per cell
├── Floating Body Data:
│   ├── FtoMasspg, FtoDatpg, FtoCenterg
│   └── FtoAceg (with CUDA stream parallelism)
└── DEM/FlexStruc Data
```

### PosCell Encoding

A key GPU optimization: particle positions are split into cell coordinates and fractional parts:

```cuda
float4 PosCell = {frac_x, frac_y, frac_z, __uint_as_float(cell_code)}
```

This avoids double-precision arithmetic in the inner force loop. Distances are computed as:
```cuda
float drx = frac1.x - frac2.x + PosCellSize * (cellX1 - cellX2);
```

Cell codes pack 3D coordinates into 32 bits:
- Default config: 13 bits X (8192 cells), 10 bits Y (1024), 9 bits Z (512)
- 2D config available: 17 bits X, 2 bits Y, 13 bits Z

### GPU Neighbor Search

Uses `cunsearch` namespace with pre-computed `BeginEndCell` arrays:

```cuda
cunsearch::InitCte(dcell, scelldiv, nc, cellzero, ini1, fin1, ini2, fin2, ini3, fin3);
for(c3 = ini3; c3 < fin3; c3 += nc.w)
  for(c2 = ini2; c2 < fin2; c2 += nc.x) {
    cunsearch::ParticleRange(c2, c3, ini1, fin1, beginendcell, pini, pfin);
    for(p2 = pini; p2 < pfin; p2++) {
      // Distance check and force computation
    }
  }
```

### GPU Optimizations

1. **Constant memory:** SPH parameters in fast `__constant__` memory
2. **Coalesced reads:** `float4` packing for positions and velocities
3. **Shared memory reductions:** Template-unrolled warp reductions for max velocity/acceleration
4. **Thrust sorting:** Radix sort for particle reordering by cell
5. **Multi-stream floating bodies:** Up to 18 CUDA streams for parallel floating body processing
6. **Template specialization:** Separate kernel instantiations for each physics configuration (eliminates runtime branches)

---

## 18. CPU vs GPU Architecture Comparison

| Aspect | CPU (`JSphCpu`) | GPU (`JSphGpu`) |
|--------|----------------|----------------|
| **Position storage** | `tdouble3 Pos_c[]` | `double2 Posxy_g[] + double Posz_g[]` + `float4 PosCell_g[]` |
| **Velocity+density** | `tfloat4 Velrho_c[]` | `float4 Velrho_g[]` |
| **Force loop** | Sequential per particle, OpenMP parallel | One CUDA thread per particle |
| **Neighbor search** | `BeginCell[]` array, CPU bucket sort | `BeginEndCell[]` (int2), Thrust radix sort |
| **Parallelism** | OpenMP threads (`OmpThreads`) | CUDA blocks x threads (thousands) |
| **Memory** | System RAM only | Device memory + constant memory |
| **Cell encoding** | `unsigned Dcell[]` with `DomCellCode` | `PosCell float4` with bit-packed cell coords |
| **Reductions** | Sequential or OMP reduction | Shared memory warp reductions |
| **Boundary normals** | `tfloat3 BoundNor_c[]` | `float3 BoundNor_g[]` |
| **Typical speedup** | 1x (baseline) | 10-50x depending on problem size |

### Class Hierarchy

```
JSph (base: constants, configuration, I/O)
├── JSphCpu (CPU arrays, OpenMP, CPU force loops)
│   └── JSphCpuSingle (single-CPU simulation loop)
└── JSphGpu (GPU arrays, CUDA memory, GPU kernels)
    └── JSphGpuSingle (single-GPU simulation loop)
```

---

## 19. Class Hierarchy and Design Patterns

### Inheritance Tree

```
JObject (base exception/naming)
└── JSph (SPH solver base)
    ├── JSphCpu
    │   └── JSphCpuSingle
    └── JSphGpu
        └── JSphGpuSingle
```

### Key Design Patterns

1. **Template Method Pattern:** `JSph::Run()` defines the algorithm skeleton; subclasses implement `ComputeStep()`, `Interaction_Forces()`, etc.

2. **Strategy Pattern (via templates):** Force computation functions are templated on kernel type, viscosity model, density diffusion, etc. Each combination generates a specialized function at compile time.

3. **Composition:** Major features are encapsulated in separate objects:
   - `JCellDivCpu/Gpu` - Neighbor search
   - `JSphShifting` - Particle shifting
   - `JSphInOut` - Inlet/outlet
   - `JChronoObjects` - Chrono integration
   - `JWaveGen` - Wave generation
   - `JDsDamping` - Damping zones
   - `JSphFlexStruc` - Flexible structures
   - `JGaugeSystem` - Measurement gauges

4. **Array Management:** `JArraysCpu`/`JArraysGpu` provide typed, resizable particle array pools with lazy allocation.

---

## 20. Key Data Structures

### StCteSph - Master SPH Constants

```cpp
struct StCteSph {
  bool simulate2d;
  TpKernel tkernel;
  StKCubicCte kcubic;
  StKWendlandCte kwend;
  float kernelh, kernelsize, kernelsize2;
  float cteb, gamma, rhopzero;
  float massfluid, massbound;
  tfloat3 gravity;
  double dp, cs0;
  float eta2;
  float spssmag, spsblin;
  float ddtkh, ddtgz;
};
```

### StDivDataCpu - Cell Division State

```cpp
struct StDivDataCpu {
  unsigned scelldiv;          // Cell subdivision (1 or 2)
  tuint3 nc;                  // Number of cells per axis
  unsigned cellfluid;         // Start of fluid cell range
  const unsigned* begincell;  // Cell particle ranges
  // ... domain mapping info
};
```

### StCteInteraction - GPU Constant Memory

```cpp
struct StCteInteraction {
  // SPH constants (mirror of StCteSph)
  float kernelh, kernelsize2, dp, cs0, eta2;
  float massb, massf, rhopzero, gamma;
  // Kernel constants
  float awen, bwenh;    // Wendland
  float cubic_a1, cubic_a2, cubic_aa; // Cubic
  // Domain
  float3 maprealposmin;
  float poscellsize;
  unsigned cellcode;
  // Periodic
  double3 perixyinc, perixzinc, periyinc, peryzinc, perizinc;
};
```

---

## 21. Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Cell division (sort) | O(N) | Bucket sort (CPU) / Radix sort (GPU) |
| Force computation | O(N * k) | k = average neighbors (~50 in 3D) |
| Time integration | O(N) | Per-particle update |
| Total per step | O(N * k) | Dominated by force computation |

### Memory Requirements

Per particle (approximate):
- CPU: ~200 bytes (positions, velocity, density, code, cell, forces)
- GPU: ~280 bytes (additional PosCell, split position arrays)
- Additional for optional features: SPS (~48B), shifting (~16B), mDBC (~24B)

### Parallelization

- **CPU:** OpenMP with configurable thread count (`OmpThreads`)
- **GPU:** CUDA with dynamic block sizing, multi-stream for floating bodies
- **Particle sorting** ensures cache-friendly memory access for force computation

---

## 22. Build System and Compilation

### CMake

```bash
cd src/source
cmake -DCMAKE_BUILD_TYPE=Release .
make -j$(nproc)
```

### Makefile

```bash
# GPU version (requires CUDA)
cd src/source
make -j$(nproc)

# CPU-only version
make -f Makefile_cpu -j$(nproc)
```

### Compilation Flags

Key preprocessor defines in `DualSphDef.h`:

| Define | Effect |
|--------|--------|
| `_WITHGPU` | Enable GPU/CUDA compilation |
| `_WITHMGPU` | Enable multi-GPU support |
| `_WITHMR` | Enable variable resolution |
| `CODE_SIZE4` | Use 32-bit particle codes (65530 MKs) |
| `DISABLE_CHRONO` | Compile without Chrono library |
| `DISABLE_WAVEGEN` | Compile without wave generation |
| `DISABLE_MOORDYNPLUS` | Compile without MoorDyn+ library |
| `DISABLE_KERNELS_EXTRA` | Wendland-only (ignore Cubic) |

### Dependencies

- CUDA Toolkit (for GPU version, tested with CUDA 12.3)
- OpenMP (for CPU parallelization)
- Pre-compiled libraries in `src/lib/`:
  - Chrono Engine (rigid body dynamics)
  - Wave generation libraries
  - MoorDyn+ (mooring dynamics)

---

*Analysis generated for DualSPHysics v5.4 (March 2025)*
*Source: https://dual.sphysics.org*
