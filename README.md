# cubloaty

**Ever wondered what's making your CUDA binary big?**

Cubloaty is a size profiler for CUDA binaries. It analyzes `.so` files and `.cubin` files to show you the size of each kernel, broken down by architecture (sm_70, sm_80, sm_90, etc.).

Think of it as [bloaty](https://github.com/google/bloaty), but for CUDA kernels.

## Quick Example

```bash
$ cd $(python -c "import torch; print(torch.__path__[0] + '/lib')")
$ cubloaty libtorch_cuda_linalg.so                   

╭─────────────────────────────────────╮
│ 📊 CUDA Kernel Size Analysis Report │
╰─────────────────────────────────────╯
                Architecture Summary                
╭─────────────────┬─────────────────┬──────────────╮
│ Architecture    │      Total Size │   Percentage │
├─────────────────┼─────────────────┼──────────────┤
│ SM_100          │          55.2MB │        18.6% │
│ SM_120          │          78.5MB │        26.4% │
│ SM_80           │          54.3MB │        18.3% │
│ SM_86           │          54.2MB │        18.3% │
│ SM_90           │          54.7MB │        18.4% │
├─────────────────┼─────────────────┼──────────────┤
│ TOTAL           │         296.8MB │       100.0% │
╰─────────────────┴─────────────────┴──────────────╯

                                       Top Kernels (All Architectures)                                       
╭────────┬────────────────────────────────────────────────────────────────────────┬──────────────┬──────────╮
│   Rank │ Kernel Name                                                            │   Total Size │        % │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│      1 │                                                                        │        5.2MB │     1.7% │
│      2 │ void zgetf2_fused_batched_kernel<32>(int, double2**, int, int, int,... │        2.3MB │     0.8% │
│      3 │ void zgetf2_fused_batched_kernel<31>(int, double2**, int, int, int,... │        2.0MB │     0.7% │
│      4 │ void zgetf2_fused_batched_kernel<30>(int, double2**, int, int, int,... │        1.9MB │     0.6% │
│      5 │ void zgetf2_fused_batched_kernel<29>(int, double2**, int, int, int,... │        1.8MB │     0.6% │
│      6 │ void zgetf2_fused_batched_kernel<28>(int, double2**, int, int, int,... │        1.7MB │     0.6% │
│      7 │ void cgetf2_fused_batched_kernel<32>(int, float2**, int, int, int, ... │        1.7MB │     0.6% │
│      8 │ void zgetf2_fused_batched_kernel<27>(int, double2**, int, int, int,... │        1.6MB │     0.5% │
│      9 │ void cgetf2_fused_batched_kernel<31>(int, float2**, int, int, int, ... │        1.6MB │     0.5% │
│     10 │ void zgetf2_fused_batched_kernel<26>(int, double2**, int, int, int,... │        1.5MB │     0.5% │
│     11 │ void zgetf2_fused_batched_kernel<25>(int, double2**, int, int, int,... │        1.5MB │     0.5% │
│     12 │ void cgetf2_fused_batched_kernel<30>(int, float2**, int, int, int, ... │        1.4MB │     0.5% │
│     13 │ void cgetf2_fused_batched_kernel<29>(int, float2**, int, int, int, ... │        1.4MB │     0.5% │
│     14 │ void zgetf2_fused_batched_kernel<24>(int, double2**, int, int, int,... │        1.4MB │     0.5% │
│     15 │ void dgetf2_fused_batched_kernel<31>(int, double**, int, int, int, ... │        1.3MB │     0.4% │
│     16 │ void zgetf2_fused_batched_kernel<23>(int, double2**, int, int, int,... │        1.3MB │     0.4% │
│     17 │ void cgetf2_fused_batched_kernel<28>(int, float2**, int, int, int, ... │        1.3MB │     0.4% │
│     18 │ void dgetf2_fused_batched_kernel<32>(int, double**, int, int, int, ... │        1.3MB │     0.4% │
│     19 │ void cgetf2_fused_batched_kernel<27>(int, float2**, int, int, int, ... │        1.3MB │     0.4% │
│     20 │ void zgetf2_fused_batched_kernel<22>(int, double2**, int, int, int,... │        1.2MB │     0.4% │
│     21 │ void cgetf2_fused_batched_kernel<26>(int, float2**, int, int, int, ... │        1.2MB │     0.4% │
│     22 │ void cgetf2_fused_batched_kernel<25>(int, float2**, int, int, int, ... │        1.2MB │     0.4% │
│     23 │ void zgetrf_batched_smallsq_noshfl_kernel<32, 32>(double2**, int, i... │        1.2MB │     0.4% │
│     24 │ void dgetf2_fused_batched_kernel<30>(int, double**, int, int, int, ... │        1.2MB │     0.4% │
│     25 │ void zgetf2_fused_batched_kernel<21>(int, double2**, int, int, int,... │        1.2MB │     0.4% │
│     26 │ void dgetf2_fused_batched_kernel<29>(int, double**, int, int, int, ... │        1.1MB │     0.4% │
│     27 │ void zgetrf_batched_smallsq_noshfl_kernel<31, 32>(double2**, int, i... │        1.1MB │     0.4% │
│     28 │ void sgetf2_fused_batched_kernel<31>(int, float**, int, int, int, i... │        1.1MB │     0.4% │
│     29 │ void zgetf2_fused_batched_kernel<20>(int, double2**, int, int, int,... │        1.1MB │     0.4% │
│     30 │ void cgetf2_fused_batched_kernel<24>(int, float2**, int, int, int, ... │        1.1MB │     0.4% │
│    ... │ (2038 more kernels)                                                    │              │          │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│        │ TOTAL                                                                  │      296.8MB │   100.0% │
╰────────┴────────────────────────────────────────────────────────────────────────┴──────────────┴──────────╯

                                             Kernels for SM_100                                              
╭────────┬────────────────────────────────────────────────────────────────────────┬──────────────┬──────────╮
│   Rank │ Kernel Name                                                            │         Size │        % │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│      1 │                                                                        │        1.6MB │     2.8% │
│      2 │ void zgetf2_fused_batched_kernel<32>(int, double2**, int, int, int,... │      397.8KB │     0.7% │
│      3 │ void cgetf2_fused_batched_kernel<32>(int, float2**, int, int, int, ... │      380.0KB │     0.7% │
│      4 │ void zgetf2_fused_batched_kernel<31>(int, double2**, int, int, int,... │      326.9KB │     0.6% │
│      5 │ void cgetf2_fused_batched_kernel<31>(int, float2**, int, int, int, ... │      323.5KB │     0.6% │
│      6 │ void zgetf2_fused_batched_kernel<30>(int, double2**, int, int, int,... │      313.6KB │     0.6% │
│      7 │ void zgetf2_fused_batched_kernel<29>(int, double2**, int, int, int,... │      299.8KB │     0.5% │
│      8 │ void cgetf2_fused_batched_kernel<29>(int, float2**, int, int, int, ... │      294.4KB │     0.5% │
│      9 │ void cgetf2_fused_batched_kernel<30>(int, float2**, int, int, int, ... │      289.0KB │     0.5% │
│     10 │ void zgetf2_fused_batched_kernel<28>(int, double2**, int, int, int,... │      286.2KB │     0.5% │
│     11 │ void dgetf2_fused_batched_kernel<31>(int, double**, int, int, int, ... │      283.6KB │     0.5% │
│     12 │ void dgetf2_fused_batched_kernel<32>(int, double**, int, int, int, ... │      282.4KB │     0.5% │
│     13 │ void zgetf2_fused_batched_kernel<27>(int, double2**, int, int, int,... │      272.6KB │     0.5% │
│     14 │ void cgetf2_fused_batched_kernel<27>(int, float2**, int, int, int, ... │      268.0KB │     0.5% │
│     15 │ void cgetf2_fused_batched_kernel<28>(int, float2**, int, int, int, ... │      264.1KB │     0.5% │
│    ... │ (1944 more kernels)                                                    │              │          │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│        │ TOTAL                                                                  │       55.2MB │   100.0% │
╰────────┴────────────────────────────────────────────────────────────────────────┴──────────────┴──────────╯

                                             Kernels for SM_120                                              
╭────────┬────────────────────────────────────────────────────────────────────────┬──────────────┬──────────╮
│   Rank │ Kernel Name                                                            │         Size │        % │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│      1 │                                                                        │        1.4MB │     1.8% │
│      2 │ void zgetf2_fused_batched_kernel<32>(int, double2**, int, int, int,... │      878.5KB │     1.1% │
│      3 │ void zgetf2_fused_batched_kernel<31>(int, double2**, int, int, int,... │      712.0KB │     0.9% │
│      4 │ void zgetf2_fused_batched_kernel<30>(int, double2**, int, int, int,... │      676.2KB │     0.8% │
│      5 │ void zgetf2_fused_batched_kernel<29>(int, double2**, int, int, int,... │      642.5KB │     0.8% │
│      6 │ void zgetf2_fused_batched_kernel<28>(int, double2**, int, int, int,... │      609.2KB │     0.8% │
│      7 │ void zgetf2_fused_batched_kernel<27>(int, double2**, int, int, int,... │      577.1KB │     0.7% │
│      8 │ void zgetf2_fused_batched_kernel<26>(int, double2**, int, int, int,... │      544.0KB │     0.7% │
│      9 │ void zgetf2_fused_batched_kernel<25>(int, double2**, int, int, int,... │      513.6KB │     0.6% │
│     10 │ void zgetrf_batched_smallsq_noshfl_kernel<32, 32>(double2**, int, i... │      485.5KB │     0.6% │
│     11 │ void zgetf2_fused_batched_kernel<24>(int, double2**, int, int, int,... │      484.9KB │     0.6% │
│     12 │ void zgetrf_batched_smallsq_noshfl_kernel<31, 32>(double2**, int, i... │      461.5KB │     0.6% │
│     13 │ void zgetf2_fused_batched_kernel<23>(int, double2**, int, int, int,... │      455.6KB │     0.6% │
│     14 │ void zgetrf_batched_smallsq_noshfl_kernel<30, 32>(double2**, int, i... │      437.2KB │     0.5% │
│     15 │ void zgetf2_fused_batched_kernel<22>(int, double2**, int, int, int,... │      425.1KB │     0.5% │
│    ... │ (1944 more kernels)                                                    │              │          │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│        │ TOTAL                                                                  │       78.5MB │   100.0% │
╰────────┴────────────────────────────────────────────────────────────────────────┴──────────────┴──────────╯

                                              Kernels for SM_80                                              
╭────────┬────────────────────────────────────────────────────────────────────────┬──────────────┬──────────╮
│   Rank │ Kernel Name                                                            │         Size │        % │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│      1 │                                                                        │      709.4KB │     1.3% │
│      2 │ void zgetf2_fused_batched_kernel<32>(int, double2**, int, int, int,... │      343.8KB │     0.6% │
│      3 │ void zgetf2_fused_batched_kernel<31>(int, double2**, int, int, int,... │      328.9KB │     0.6% │
│      4 │ void cgetf2_fused_batched_kernel<32>(int, float2**, int, int, int, ... │      323.5KB │     0.6% │
│      5 │ void cgetf2_fused_batched_kernel<31>(int, float2**, int, int, int, ... │      320.5KB │     0.6% │
│      6 │ void zgetf2_fused_batched_kernel<30>(int, double2**, int, int, int,... │      315.1KB │     0.6% │
│      7 │ void zgetf2_fused_batched_kernel<29>(int, double2**, int, int, int,... │      299.8KB │     0.5% │
│      8 │ void cgetf2_fused_batched_kernel<30>(int, float2**, int, int, int, ... │      295.8KB │     0.5% │
│      9 │ void cgetf2_fused_batched_kernel<29>(int, float2**, int, int, int, ... │      291.2KB │     0.5% │
│     10 │ void zgetf2_fused_batched_kernel<28>(int, double2**, int, int, int,... │      285.6KB │     0.5% │
│     11 │ void sgetf2_native_kernel<512, 47>(int, int, float*, int, int volat... │      283.4KB │     0.5% │
│     12 │ void zgetf2_fused_batched_kernel<27>(int, double2**, int, int, int,... │      271.9KB │     0.5% │
│     13 │ void cgetf2_fused_batched_kernel<28>(int, float2**, int, int, int, ... │      269.2KB │     0.5% │
│     14 │ void cgetf2_fused_batched_kernel<27>(int, float2**, int, int, int, ... │      264.6KB │     0.5% │
│     15 │ void sgetf2_native_kernel<512, 45>(int, int, float*, int, int volat... │      261.5KB │     0.5% │
│    ... │ (2051 more kernels)                                                    │              │          │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│        │ TOTAL                                                                  │       54.3MB │   100.0% │
╰────────┴────────────────────────────────────────────────────────────────────────┴──────────────┴──────────╯

                                              Kernels for SM_86                                              
╭────────┬────────────────────────────────────────────────────────────────────────┬──────────────┬──────────╮
│   Rank │ Kernel Name                                                            │         Size │        % │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│      1 │                                                                        │      710.2KB │     1.3% │
│      2 │ void zgetf2_fused_batched_kernel<32>(int, double2**, int, int, int,... │      343.8KB │     0.6% │
│      3 │ void zgetf2_fused_batched_kernel<31>(int, double2**, int, int, int,... │      328.9KB │     0.6% │
│      4 │ void cgetf2_fused_batched_kernel<32>(int, float2**, int, int, int, ... │      323.5KB │     0.6% │
│      5 │ void cgetf2_fused_batched_kernel<31>(int, float2**, int, int, int, ... │      320.5KB │     0.6% │
│      6 │ void zgetf2_fused_batched_kernel<30>(int, double2**, int, int, int,... │      315.1KB │     0.6% │
│      7 │ void zgetf2_fused_batched_kernel<29>(int, double2**, int, int, int,... │      299.8KB │     0.5% │
│      8 │ void cgetf2_fused_batched_kernel<30>(int, float2**, int, int, int, ... │      295.8KB │     0.5% │
│      9 │ void cgetf2_fused_batched_kernel<29>(int, float2**, int, int, int, ... │      291.9KB │     0.5% │
│     10 │ void zgetf2_fused_batched_kernel<28>(int, double2**, int, int, int,... │      285.6KB │     0.5% │
│     11 │ void zgetf2_fused_batched_kernel<27>(int, double2**, int, int, int,... │      271.9KB │     0.5% │
│     12 │ void cgetf2_fused_batched_kernel<28>(int, float2**, int, int, int, ... │      269.2KB │     0.5% │
│     13 │ void cgetf2_fused_batched_kernel<27>(int, float2**, int, int, int, ... │      264.8KB │     0.5% │
│     14 │ void zgetf2_fused_batched_kernel<26>(int, double2**, int, int, int,... │      258.4KB │     0.5% │
│     15 │ void zgetf2_fused_batched_kernel<25>(int, double2**, int, int, int,... │      246.0KB │     0.4% │
│    ... │ (2051 more kernels)                                                    │              │          │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│        │ TOTAL                                                                  │       54.2MB │   100.0% │
╰────────┴────────────────────────────────────────────────────────────────────────┴──────────────┴──────────╯

                                              Kernels for SM_90                                              
╭────────┬────────────────────────────────────────────────────────────────────────┬──────────────┬──────────╮
│   Rank │ Kernel Name                                                            │         Size │        % │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│      1 │                                                                        │      827.4KB │     1.5% │
│      2 │ void zgetf2_fused_batched_kernel<32>(int, double2**, int, int, int,... │      343.6KB │     0.6% │
│      3 │ void zgetf2_fused_batched_kernel<31>(int, double2**, int, int, int,... │      328.5KB │     0.6% │
│      4 │ void cgetf2_fused_batched_kernel<32>(int, float2**, int, int, int, ... │      323.8KB │     0.6% │
│      5 │ void cgetf2_fused_batched_kernel<31>(int, float2**, int, int, int, ... │      320.8KB │     0.6% │
│      6 │ void zgetf2_fused_batched_kernel<30>(int, double2**, int, int, int,... │      315.4KB │     0.6% │
│      7 │ void zgetf2_fused_batched_kernel<29>(int, double2**, int, int, int,... │      300.6KB │     0.5% │
│      8 │ void cgetf2_fused_batched_kernel<30>(int, float2**, int, int, int, ... │      296.0KB │     0.5% │
│      9 │ void cgetf2_fused_batched_kernel<29>(int, float2**, int, int, int, ... │      292.5KB │     0.5% │
│     10 │ void zgetf2_fused_batched_kernel<28>(int, double2**, int, int, int,... │      285.9KB │     0.5% │
│     11 │ void zgetf2_fused_batched_kernel<27>(int, double2**, int, int, int,... │      273.2KB │     0.5% │
│     12 │ void cgetf2_fused_batched_kernel<28>(int, float2**, int, int, int, ... │      269.5KB │     0.5% │
│     13 │ void cgetf2_fused_batched_kernel<27>(int, float2**, int, int, int, ... │      265.0KB │     0.5% │
│     14 │ void zgetf2_fused_batched_kernel<26>(int, double2**, int, int, int,... │      259.4KB │     0.5% │
│     15 │ void zgetf2_fused_batched_kernel<25>(int, double2**, int, int, int,... │      246.0KB │     0.4% │
│    ... │ (1945 more kernels)                                                    │              │          │
├────────┼────────────────────────────────────────────────────────────────────────┼──────────────┼──────────┤
│        │ TOTAL                                                                  │       54.7MB │   100.0% │
╰────────┴────────────────────────────────────────────────────────────────────────┴──────────────┴──────────╯

✓ Analysis complete!
```

## Features

- 📊 **Multi-architecture analysis** - See kernel sizes across sm_70, sm_80, sm_90, etc.
- 🔍 **Regex filtering** - Filter kernels by name pattern
- 📦 **Multiple formats** - `.so` libraries and standalone `.cubin` files
- 🎨 **Rich output** - Beautiful tables or JSON for scripting
- ⚡ **Fast** - Analyzes binaries in seconds

## Dependencies

Cubloaty requires the following tools to be installed and available in your `PATH`:

- **CUDA Toolkit** - for `cuobjdump` (part of the CUDA installation)
- **binutils** - for `objdump`, `objcopy`, and `readelf`
- **gcc/g++** - for `c++filt` (symbol demangling)

On Ubuntu/Debian:
```bash
sudo apt-get install binutils gcc
```

CUDA Toolkit can be downloaded from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

## Installation

```bash
pip install -e .
```

Or install directly from git:
```bash
pip install git+https://github.com/yzh119/cubloaty.git
```

## Usage

### Analyze a shared library

```bash
cubloaty libmykernel.so
```

### Analyze a cubin file

```bash
cubloaty kernel.sm_90.cubin
```

### Show top 50 kernels

```bash
cubloaty libmykernel.so --top 50
```

### Filter by architecture

```bash
cubloaty libmykernel.so --arch sm_90
```

### Filter kernels by name (regex)

```bash
# Find all GEMM kernels
cubloaty libmykernel.so --filter "gemm"

# Find attention-related kernels
cubloaty libmykernel.so --filter "attention|flash"
```

### Output as JSON

```bash
cubloaty libmykernel.so --format json > analysis.json
```

### Show full kernel names without truncation

```bash
cubloaty libmykernel.so --full-names
```

### Combine filters

```bash
# Show top 20 GEMM kernels for sm_90 in JSON format
cubloaty lib.so --arch sm_90 --filter "gemm" --top 20 --format json
```

## Advanced Examples

### Compare kernel sizes across architectures

```bash
# Show per-architecture breakdown
cubloaty libmykernel.so --verbose
```

### Find the largest kernels

```bash
# Show just the top 10
cubloaty libmykernel.so --top 10
```

### Export for further analysis

```bash
# Get JSON output and process with jq
cubloaty lib.so --format json | jq '.kernels[] | select(.size > 100000)'
```

## Options

```
  file                    Path to .so or .cubin file to analyze
  --top N, -n N          Show top N kernels (default: 30)
  --arch ARCH, -a ARCH   Filter by architecture (e.g., sm_90, sm_80)
  --filter REGEX, -r     Filter kernel names by regex (case-insensitive)
  --format {table,json}  Output format (default: table)
  --full-names           Show full kernel names without truncation
  --no-color             Disable colored output
  --verbose, -v          Show detailed processing information
  --version              Show version number
```

## How It Works

Cubloaty extracts CUDA fatbinary sections from shared libraries using `objdump` and `objcopy`, then uses `cuobjdump` to extract individual cubins for each architecture. It analyzes each cubin with `readelf` to extract kernel symbols and their sizes, and uses `c++filt` to demangle C++ symbol names.

## Contributing

Issues and pull requests are welcome!
