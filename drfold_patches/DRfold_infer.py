
import os, sys
import torch
import numpy as np
from subprocess import Popen, PIPE, STDOUT

# Get the directory where the script is located
exp_dir = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"
# dlexps = ['cfg_95','cfg_96','cfg_97','cfg_99']
dlexps = ['cfg_97']

print(f"[DRfold2] Starting prediction pipeline on {device} device")

# Get input FASTA file and output directory from command line arguments
fastafile = os.path.realpath(sys.argv[1])
outdir = os.path.realpath(sys.argv[2])

print(f"[DRfold2] Input: {fastafile}")
print(f"[DRfold2] Output: {outdir}")

# Initialize clustering flag and AF3 file path
pclu = False
af3file = None

# Parse command line arguments
# Acceptable formats:
# python DRfold_infer.py input.fasta output_dir
# python DRfold_infer.py input.fasta output_dir 1
# python DRfold_infer.py input.fasta output_dir --af3 af3_model.pdb
# python DRfold_infer.py input.fasta output_dir 1 --af3 af3_model.pdb

for i in range(3, len(sys.argv)):
    if sys.argv[i] == "1" and i == 3:
        pclu = True
        print('[DRfold2] Clustering enabled - will generate multiple models')
    elif sys.argv[i] == "--af3" and i+1 < len(sys.argv):
        af3file = os.path.realpath(sys.argv[i+1])
        print(f'[DRfold2] Using AlphaFold3 structure: {af3file}')

if not pclu:
    print('[DRfold2] Clustering disabled - will generate single model')

# Create output directory if it doesn't exist
if not os.path.isdir(outdir):
    os.makedirs(outdir)
    print(f"[DRfold2] Created output directory: {outdir}")  

# Create subdirectories for different outputs
ret_dir = os.path.join(outdir,'rets_dir')  # For return files
if not os.path.isdir(ret_dir):
    os.makedirs(ret_dir)
    print(f"[DRfold2] Created returns directory: {ret_dir}")

folddir = os.path.join(outdir,'folds')     # For folded structures
if not os.path.isdir(folddir):
    os.makedirs(folddir)
    print(f"[DRfold2] Created folds directory: {folddir}")

refdir = os.path.join(outdir,'relax')      # For relaxed structures
if not os.path.isdir(refdir):
    os.makedirs(refdir)
    print(f"[DRfold2] Created relaxation directory: {refdir}")


# Helper function to run commands and capture output
def run_cmd(cmd, description):
    print(f"[DRfold2] {description}")
    print(f"[DRfold2] Command: {cmd}")
    
    # Execute the command and capture output in real-time
    process = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, universal_newlines=True, bufsize=1)
    
    # Print output line by line as it becomes available
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            print(f"[DRfold2 subprocess] {line}")
    
    # Get return code
    return_code = process.wait()
    if return_code == 0:
        print(f"[DRfold2] {description} completed successfully")
    else:
        print(f"[DRfold2] {description} failed with return code {return_code}")
    return return_code


# Create paths for model directories and test scripts
dlmains = [os.path.join(exp_dir, one_exp, 'test_modeldir.py') for one_exp in dlexps]
dirs = [os.path.join(exp_dir, 'model_hub', one_exp) for one_exp in dlexps]


# Check if processing has been done before
if not os.path.isfile(ret_dir + '/done'): 
    print("[DRfold2] Step 1/4: GENERATING INITIAL PREDICTIONS")
    print(f"[DRfold2] No previous predictions found, will generate e2e and geo files")
    
    # Run each model configuration
    for idx, (dlmain, one_exp, mdir) in enumerate(zip(dlmains, dlexps, dirs)):
        # Construct command to run the model
        cmd = f'python {dlmain} {device} {fastafile} {ret_dir}/{one_exp}_ {mdir}'
        description = f"Running model {idx+1}/{len(dlexps)}: {one_exp}"
        run_cmd(cmd, description)

    # Mark processing as complete
    wfile = open(ret_dir+'/done','w')
    wfile.write('1')
    wfile.close()
    print("[DRfold2] Initial predictions generation completed")
else:
    print("[DRfold2] Step 1/4: USING EXISTING PREDICTIONS")
    print(f"[DRfold2] Found previous predictions in {ret_dir}, using existing e2e and geo files")

# Helper function to get model PDB file
def get_model_pdb(tdir,opt):
    files = os.listdir(tdir)
    files = [afile for afile in files if afile.startswith(opt)][0]
    return files


# Set up directory paths and configuration files
cso_dir = folddir                                                    # Directory for coarse-grained structures
clufile = os.path.join(folddir,'clu.txt')                            # Clustering results file
config_sel = os.path.join(exp_dir,'cfg_for_selection.json')          # Selection configuration
foldconfig = os.path.join(exp_dir,'cfg_for_folding.json')            # Folding configuration
selpython = os.path.join(exp_dir,'PotentialFold','Selection.py')     # Selection script
optpython = os.path.join(exp_dir,'PotentialFold','Optimization.py')  # Optimization script
clupy = os.path.join(exp_dir,'PotentialFold','Clust.py')             # Clustering script
arena = os.path.join(exp_dir,'Arena','Arena')                        # Arena executable for structure refinement


# Set up initial save prefixes for optimization and selection
optsaveprefix = os.path.join(cso_dir, f'opt_0')
save_prefix = os.path.join(cso_dir, f'sel_0')

# Get all .ret files from the return directory
rets = os.listdir(ret_dir)
rets = [afile for afile in rets if afile.endswith('.ret')]
rets = [os.path.join(ret_dir,aret) for aret in rets ]
ret_str = ' '.join(rets)

print("[DRfold2] Step 2/4: SELECTION PROCESS")
print(f"[DRfold2] Found {len(rets)} return files for selection")
print(f"[DRfold2] Using selection config: {config_sel}")
print(f"[DRfold2] Output prefix: {save_prefix}")


# Run selection process
cmd = f'python {selpython} {fastafile} {config_sel} {save_prefix} {ret_str}'
run_cmd(cmd, "Running selection process")

print("[DRfold2] Step 3/4: OPTIMIZATION PROCESS")
print(f"[DRfold2] Using fold config: {foldconfig}")
print(f"[DRfold2] Optimization output prefix: {optsaveprefix}")

# Run optimization process with optional AF3 file
cmd = f'python {optpython} {fastafile} {optsaveprefix} {ret_dir} {save_prefix} {foldconfig}'
if af3file and os.path.exists(af3file):
    cmd += f' {af3file}'
run_cmd(cmd, "Running optimization process")

# Get the coarse-grained PDB and save refined structure
cgpdb = os.path.join(folddir,get_model_pdb(folddir,'opt_0'))
savepdb = os.path.join(refdir,'model_1.pdb')

print("[DRfold2] Step 4/4: STRUCTURE REFINEMENT")
print(f"[DRfold2] Found optimized structure: {cgpdb}")
print(f"[DRfold2] Final output will be saved to: {savepdb}")

cmd = f'{arena} {cgpdb} {savepdb} 7'
run_cmd(cmd, "Running structure refinement")

# If clustering is enabled (pclu=True)
if pclu:
    print("[DRfold2] ADDITIONAL STEP: CLUSTERING")
    print(f"[DRfold2] Running clustering process, output: {clufile}")
    
    # Run clustering process
    cmd = f'python {clupy} {ret_dir} {clufile}'
    run_cmd(cmd, "Running clustering")

    # Read clustering results
    lines = open(clufile).readlines()
    lines = [aline.strip() for aline in lines]
    lines = [aline for aline in lines if aline]
    
    cluster_count = len(lines) - 1
    print(f"[DRfold2] Found {cluster_count} additional clusters to process")

    # Process each cluster
    for i in range(1,len(lines)):
        print(f"[DRfold2] PROCESSING CLUSTER {i}/{cluster_count}")
        
        # Get return files for this cluster
        rets = lines[i].split()
        rets = [os.path.join(ret_dir,aret.replace('.pdb','.ret')) for aret in rets ]
        ret_str = ' '.join(rets)

        # Set up save prefixes for this cluster
        optsaveprefix =  os.path.join(cso_dir,f'opt_{str(i+1)}')
        save_prefix = os.path.join(cso_dir,f'sel_{str(i+1)}')
        
        print(f"[DRfold2] Cluster {i} Selection Process")
        print(f"[DRfold2] Found {len(rets)} return files for selection")
        print(f"[DRfold2] Selection output prefix: {save_prefix}")

        # Run selection process for this cluster
        cmd = f'python {selpython} {fastafile} {config_sel} {save_prefix} {ret_str}'
        run_cmd(cmd, f"Running selection for cluster {i}")
        
        print(f"[DRfold2] Cluster {i} Optimization Process")
        print(f"[DRfold2] Optimization output prefix: {optsaveprefix}")

        # Run optimization process for this cluster with optional AF3 file
        cmd = f'python {optpython} {fastafile} {optsaveprefix} {ret_dir} {save_prefix} {foldconfig}'
        if af3file and os.path.exists(af3file):
            cmd += f' {af3file}'
        run_cmd(cmd, f"Running optimization for cluster {i}")

        # Get the coarse-grained PDB and save refined structure for this cluster
        cgpdb = os.path.join(folddir,get_model_pdb(folddir,f'opt_{str(i+1)}'))
        savepdb = os.path.join(refdir,f'model_{str(i+1)}.pdb')
        
        print(f"[DRfold2] Cluster {i} Refinement Process")
        print(f"[DRfold2] Found optimized structure: {cgpdb}")
        print(f"[DRfold2] Final output will be saved to: {savepdb}")

        cmd = f'{arena} {cgpdb} {savepdb} 7'
        run_cmd(cmd, f"Running refinement for cluster {i}")

print("[DRfold2] PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
