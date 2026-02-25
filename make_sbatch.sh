#!/usr/bin/env zsh

# Example usage:
# GPU=4 TIME=0-04:00:00 PARTITION=batch TYPE=train ./make_sbatch.sh

DATETIME=$(date +"%Y%m%d_%H%M%S")

TIME=${TIME:-2-00:00:00}
PARTITION=${PARTITION:-batch}
TYPE=${TYPE:-train} # jupyter, eval, test
CONDA_ENV=${CONDA_ENV:-nav}
NODES=${NODES:-1}

GPU=${GPU:-1}
CPUS=${CPUS:-32}
MEM=${MEM:-64G}
PY_ARGS="${@}"
BRANCH=${BRANCH:-main}

if [ "${PARTITION}" = "short" ]; then
    TIME="0-04:00:00"
    CPUS=16
fi

HOME_DIR=${HOME_DIR:-"/users/ejlaird/Projects/inertial-nav"}
ENV_DIR=${ENV_DIR:-"/lustre/smuexa01/client/users/ejlaird/envs"}
WORK_DIR=${WORK_DIR:-"/lustre/smuexa01/client/users/ejlaird/inertial_nav/workdirs"}

if [ "${BRANCH}" = "local" ]; then
    WORK_DIR=${HOME_DIR}
fi


if [ "${TYPE}" = "jupyter" ]; then
    WORK_DIR=${HOME_DIR}
    COMMAND="jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
elif [ "${TYPE}" = "train" ]; then
    COMMAND="HYDRA_FULL_ERROR=1 python src/train.py ${PY_ARGS}"
elif [ "${TYPE}" = "test" ]; then
    COMMAND="HYDRA_FULL_ERROR=1 python src/test.py ${PY_ARGS}"
elif [ "${TYPE}" = "posterior_diagnostics" ]; then
    COMMAND="HYDRA_FULL_ERROR=1 python scripts/eval_posterior_diagnostics.py ${PY_ARGS}"
elif [ "${TYPE}" = "eval_maneuver_stratified" ]; then
    COMMAND="HYDRA_FULL_ERROR=1 python scripts/eval_maneuver_stratified.py ${PY_ARGS}"
elif [ "${TYPE}" = "eval_anomalous_sequences" ]; then
    COMMAND="HYDRA_FULL_ERROR=1 python scripts/eval_anomalous_sequences.py ${PY_ARGS}"
elif [ "${TYPE}" = "ablation_table" ]; then
    COMMAND="python scripts/generate_ablation_table.py ${PY_ARGS}"
elif [ "${TYPE}" = "pytest" ]; then
    COMMAND="python -m pytest ${PY_ARGS} -v"
fi

LOG_FILE="output/${TYPE}/${TYPE}_%j.out"

echo "COMMAND: GPU=${GPU} CPUS=${CPUS} MEM=${MEM} PARTITION=${PARTITION} TIME=${TIME} TYPE=${TYPE} CONDA_ENV=${CONDA_ENV} ./make_sbatch.sh ${COMMAND}"

# write sbatch script
echo "#!/usr/bin/env zsh
#SBATCH -J ${TYPE}
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -o ${HOME_DIR}/output/${TYPE}/${TYPE}_%j.out
#SBATCH --cpus-per-task=${CPUS} 
#SBATCH --mem=${MEM}     
#SBATCH --nodes=${NODES}
#SBATCH --gres=gpu:${GPU}
#SBATCH --time=${TIME} 
#SBATCH --partition=${PARTITION}
#SBATCH --tasks-per-node=1

module purge
module load conda
conda activate ${ENV_DIR}/${CONDA_ENV}

which python
echo $CONDA_PREFIX

# Clone repo for this job
if [ \"${BRANCH}\" = \"local\" ]; then
    echo Skipping clone for local testing
else
    cd ${WORK_DIR}
    mkdir -p inertial-nav_\${SLURM_JOB_ID}
    cd inertial-nav_\${SLURM_JOB_ID}
    echo "Current working directory: inertial-nav_\${SLURM_JOB_ID}"
    git clone git@github.com:elilaird/inertial-nav.git .
    git checkout ${BRANCH}

    # Print git state information
    echo \"=== GIT STATE ===\"
    echo \"Branch: \$(git branch --show-current)\"
    echo \"Commit hash: \$(git rev-parse HEAD)\"
    echo \"Commit short: \$(git rev-parse --short HEAD)\"
    echo \"Commit message: \$(git log -1 --pretty=format:'%s')\"
    echo \"Commit author: \$(git log -1 --pretty=format:'%an <%ae>')\"
    echo \"Commit date: \$(git log -1 --pretty=format:'%ad' --date=iso)\"
    echo \"===============\"
fi

echo \"WORK_DIR: \$(pwd)\"
echo "COMMAND: GPU=${GPU} CPUS=${CPUS} MEM=${MEM} PARTITION=${PARTITION} TYPE=${TYPE} TIME=${TIME} CONDA_ENV=${CONDA_ENV} ./make_sbatch.sh ${COMMAND}"

srun --ntasks=${NODES} --distribution=block  bash -c \"${COMMAND}\"
" > ${TYPE}_${DATETIME}.sbatch

# submit sbatch script
sbatch ${TYPE}_${DATETIME}.sbatch

sleep 0.1
rm -f ${TYPE}_${DATETIME}.sbatch