# Deeplearning HW

This repository contains scripts and experimental code for CPE 587 Deep Learning Home Work.
The code is intended for homework.

## Environment Setup

We recommend using a virtual environment.
```
git clone git@github.com:wonderingpanda510gh/CPE487587_DLHW.git
cd CPE487587_DLHW
uv venv
source .venv/bin/activate
uv sync
```

## How to Run
Run the main script from the project root:

For the HW01:
```
bash scripts/binaryclassification_impl.py
```
The script will automatically:

- create a output directories

- save trained weights and save generated PDF

Update the HW01, found the right way to initialize the uv environment.

For the HW02:

We have the pdf for the theory part. Here we just discuss about the HW02-Q7 and HW02-Q8

HW02-Q7:

First, you need to log in to a server have linux system, then
```
git clone git@github.com:wonderingpanda510gh/CPE487587_DLHW.git

cd CPE487587_DLHW

code . # this is to use vs code to visualize the code
```
Then, we have the .sh file you can directly use them by applying
```
bash scripts/binaryclassification_animate_impl.sh
```
After all, the results are recorded at the "large_weights_evolution" and "wt_animation" folder

HW02-Q8:

First, clone the repository
```
git clone git@github.com:wonderingpanda510gh/CPE487587_DLHW.git

cd CPE487587_DLHW

code . # this is to use vs code to visualize the code
```
After that, we need to download the dataset
```
bash scripts/malwaredatadownload.sh

# please pay attention to this dataset; 
# it's not a clean dataset, and we need to manually clean it. 
# I suggested using a CSV file loader to check that everything is right.
```
Then, we have the .sh file that can used to directly run the code
```
bash scripts/multiclass_impl.sh
```
this script will automaticlly run the code five times, and all the output will be recorded in "results/" folder.
## Author
Zhehao Yi
