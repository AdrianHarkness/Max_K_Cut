#!/bin/bash
# Create directories if they don't exist
mkdir -p src notebooks tests plots docs

# Move Python modules to src/ and rename (if needed)
mv Helper_Functions.py src/helper_functions.py
mv Models.py src/models.py
mv Penalties.py src/penalties.py
mv Plot_qaoa_histogram.py src/plot_qaoa_histogram.py
mv Run_qaoa_extract_samples.py src/run_qaoa_extract_samples.py

# Move Jupyter notebooks to notebooks/
mv box_plots.ipynb notebooks/box_plots.ipynb
mv model_testing.ipynb notebooks/model_testing.ipynb
mv penalty_interpolation.ipynb notebooks/penalty_interpolation.ipynb

# Rename Plots folder to lowercase if desired
if [ -d "Plots" ]; then
    mv Plots plots
fi

# Create an empty __init__.py in src/ if it doesn't exist
if [ ! -f "src/__init__.py" ]; then
    touch src/__init__.py
fi

echo "Repository restructured successfully."