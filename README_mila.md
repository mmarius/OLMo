## Setup

### Install virtual environment

- `module load python/3.10`

- `python3 -m venv $HOME/venvs/olmo`

- `source $HOME/venvs/olmo/bin/activate`

- `pip install --upgrade pip`

- `pip install -e .[all]`

## Download training data

Run the following to download the training data for OLMo:

(Note: This will run wget on a lot of files, so this will take quite some time and will require a lot of storage space.)

```bash 
FILE_PATHS=./training_data/data_paths.csv
OUTPUT_DIR=/change/me/to/your/output/dir
PARALLEL_DOWNLOADS=4
bash ./bash_scripts/download_data.sh $FILE_PATHS $OUTPUT_DIR $PARALLEL_DOWNLOADS
```

## Download pre-trained checkpoints

```bash 
OUTPUT_DIR=/change/me/to/your/output/dir
bash ./bash_scripts/download_checkpoints.sh $OUTPUT_DIR 
```

## Re-start pre-training

TODO