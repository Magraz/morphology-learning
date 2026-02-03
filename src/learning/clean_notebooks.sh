SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
find $SCRIPT_DIR -type f -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;