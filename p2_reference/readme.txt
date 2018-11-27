Login Pace
command line:
    qsub -I -q coc-ice -l nodes=1:ppn=2:gpus=1,walltime=2:00:00,pmem=2gb
After above:
    module load gcc
    module load cuda
Then:
    nvcc test.cu -o test
    ./test