import os

# Environment setup
env = Environment(ENV=os.environ)
env.Append(CXX='nvcc')

# Custom builder for CUDA files
cuda_build = Builder(action='nvcc -o $TARGET $SOURCES -O3 -lcublas')
env.Append(BUILDERS={'CudaBuild': cuda_build})

# Find all .cu files
cuda_files = Glob('*.cu')

# Compile each CUDA file separately
for cuda_file in cuda_files:
    program = cuda_file.name.split('.')[0]
    env.CudaBuild(target=program, source=cuda_file.name)