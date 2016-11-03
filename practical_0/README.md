# Surf Sara short tutorial

## How to login

To login to Cartesius run the following command:

```bash
ssh myusername@cartesius.surfsara.nl
```

You have to replace *myusername* with your actual user name that the SURF SARA people gave you.

Once you have logged in, it's like any other terminal. You can make directories, search for files and run commands. However, before running some commands you need to load the respective modules, e.g. the python modules or the tensorflow modules.

## How to load a module?

To load a new module is rather easy. You type:

```bash
module load mymodule
```

Modules that you will mostly use in this course are:

```bash
module load python/2.7.11
module load cuda/7.5.18
module load cudnn/7.5-v5
```

For simplicity, you can do either of the three things:
- just call the commands in the terminal;
- have a script somewhere in your home folder, /home/myusername/loadmodules.sh, where you insert all the modules you want to load every time you connect to Cartesius;
- or add the commands to the job script file that you will use for running an experiment.

If you are unsure about your options, you can always query the module system for available modules with:

```bash
module av cuda
```

if you would want to see what cuda modules are available for instance. This command uses wildcards, so if you're unsure also about the name, you can try "module av *ud*"

```bash
module av *ud*
```

To get the list of already loaded modules you type:

```bash
module list
```

To get the list of available modules you type:

```bash
module avail
```

To get the help you type:

```bash
module help
```

## What is a job file and how do I make one?

In Cartesius you are given a terminal. **You should never run experiments directly in the terminal**, e.g.:

```bash
python myexperiment.py
```

You shouldn't do that, because in that case you do not actually use the HPC framework (queuing, parallelism), only the hardware. **If you run directly an experiment, you will definitely get a warning or might get banned.**

Instead, you should use job files, which are scripts that define how to run an experiment. A job file is a simple bash script, which contains two types of information.

First, it includes some header statements regarding the cluster you are going to use, including:

- number of nodes;
- for how long are you going to run the experiment. Make a safe estimate, do not be overly optimistic about the speed of your code. On the other hand do not just state that your tasks are going to run forever, because you have a limited (but not little) time budget for running jobs in SURF SARA.

For more information regarding the header statements it is highly recommended to check - https://userinfo.surfsara.nl/systems/cartesius/usage/batch-usage

Second, it includes the commands that you are going to run for your experiment, for instance:

```bash
python myexperiment.py
```

We give you three example job files: example_job_interactive.sh and example_job.sh and example_job_gpu.sh. In the headers you see that:

1. We ask for 1 process ("#SBATCH -n 1").
2. We say that the experiment is going to take about 10 seconds ("#SBATCH -t 00:00:10").
3. We say where the output should be stored ("#SBATCH -o fibonacci_%A.output"), as in the output you would normally get into a terminal if you would run the command directly.
4. We say where the error file should be stored (""#SBATCH -e fibonacci_%A.error"). The error file contains the errors that the system threw in case, of course, there is an error and a job is terminated.
5. We say which modules we want to load ("module load python/2.7.11"). We just need python for this example.
6. Finally, we say which command/experiment we want to run.
  - If you want an interactive mode, use the command srun (*srun -u python fibonacci.py*). The *srun* is running the command through the batching system. **DO NOT call directly the python command**. That would work, but it would allocate directly resources that normally would go to other users not through the batching system. **By running commands directly might get you banned**.
  - The interactive mode is good for short or fast experiments. If you are going to run an experiment that takes longer, you should use the command *sbatch*. For using this you can check the second file, example_job.sh. The last command is calling the python script directly. However, the difference now is that we do not call the job file directly, instead we use the *sbatch* command:

```bash
sbatch example_job.job
```

You can run the third job file as

```bash
sbatch example_job_gpu.job
```

This file additionally contains the header "#SBATCH gpu", which states that we require a gpu. Also, it loads modules that we need to run processes in the GPU, such as cuda and cudnn. Note that if you ask for nodes that have GPUs only your experiment might take a while to start as the nodes with the GPUs might already be occupied for another process.

  **If you do not use the command sbatch, you will call directly python without using the batching system, which in turn might get you banned**.

## How to run my job file?

Once you have your job file, you can run your experiment:

```bash
sbatch example_job.job
```

Then you have your experiment submitted through the batching system. If you want to see what jobs you have submitted, you can run:

```bash
squeue -u myusername
```

If you want to get information regarding your account, e.g., how much of your time budget is left, you can run:

```bash
accinfo
```

## Some other very useful commands

### screen

A very useful command is *screen*. With *screen* you have two great benefits.

First, you can open as many virtual terminals as you want. This means you do not need to *ssh* multiple times. You only *ssh* once, then you can open new terminals by simply calling the command:

```bash
screen
```

Run the following to get help:

```bash
screen -h
```

Second, generally, when you logout from your machine or from your ssh session your active processes might get killed, very much like when you shut down your computer. However, quite often you are running something which you want to keep alive. With *screen* you can do that very easily. Simple, open a new terminal, run whatever you want to run there, then detach from the screen (practically getting out of the screen), and you should be fine. When you wanna enter the screen, you just re-attach it.

### nvidia-smi

Another useful command is nvidia-smi, which you run as

```bash
nvidia-smi
```

With cuda loaded you can access nvidia-smi which gives you information about the usage of the GPU(s). Note that this should be one of your nodes you are doing your computation on. 
