#!/bin/bash

# Check the number of arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <operating_system> <environment_name>"
    exit 1
fi

# Assign the arguments to variables
os="$1"
environment_name="$2"

# Check the argument for the operating system
if [ "$os" == "windows" ]; then
    # Extract installed pip packages
    pip_packages=$(conda list -n "$environment_name" | grep -v "^#" | grep "pypi" | awk '{ print $1 }')

    # Export conda environment without builds, and append pip packages
    conda env export -n "$environment_name" --no-builds | grep -v "^prefix: " > environment-windows.yml
    echo "$pip_packages" >> environment-windows.yml

    echo "Environment exported to environment-windows.yml"

elif [ "$os" == "mac" ]; then
    # Extract installed pip packages
    pip_packages=$(conda list -n "$environment_name" | grep -v "^#" | grep "pypi" | awk '{ print $1 }')

    # Export conda environment without builds, and append pip packages
    conda env export -n "$environment_name" --no-builds | grep -v "^prefix: " > environment-mac.yml
    echo "$pip_packages" >> environment-mac.yml

    echo "Environment exported to environment-mac.yml"

elif [ "$os" == "linux" ]; then
    # Extract installed pip packages
    pip_packages=$(conda list -n "$environment_name" | grep -v "^#" | grep "pypi" | awk '{ print $1 }')

    # Export conda environment without builds, and append pip packages
    conda env export -n "$environment_name" --no-builds | grep -v "^prefix: " > environment-linux.yml
    echo "$pip_packages" >> environment-linux.yml

    echo "Environment exported to environment-linux.yml"

else
    echo "Invalid argument. Please specify 'windows', 'mac', or 'linux' as the first argument."
    exit 1
fi
