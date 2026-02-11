SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PARENT_DIR=$( dirname "$SCRIPT_DIR" )
find "$PARENT_DIR" -type f -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;