papermill PreProcessing.ipynb waveNotebooks/output_notebook_wave_1.ipynb -p wave_param 1
papermill PreProcessing.ipynb waveNotebooks/output_notebook_wave_2.ipynb -p wave_param 2
papermill PreProcessing.ipynb waveNotebooks/output_notebook_wave_3.ipynb -p wave_param 3
papermill PreProcessing.ipynb waveNotebooks/output_notebook_wave_4.ipynb -p wave_param 4
papermill PreProcessing.ipynb waveNotebooks/output_notebook_wave_5.ipynb -p wave_param 5

python3 labeller.py 1 2 3 4 5