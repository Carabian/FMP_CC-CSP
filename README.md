# CV-CSP
Reposetori for the final master project of Pau Carabí. 

Here are presnted the 3 main scripts programmed by the student. The pipe line would be:
- Run VEP over the ClinVar data.
- Run hail_parsing.py on the ClinVar annotated chromosomes.
- Run cleaning_data_pandas.py on the chromosomes outputed by the anterior script.
- Run nn_pytorch.py which runs the neural netork and creates graphs and figures. nn_pytoch_loop.py was used to run the grid search.

Graphs created during the grid search found in ROC_curve, loss_graph and confusion_matrix folders.

As well a file with the system specifications used is provided:  system_specs.txt, and a .yml file to reproduce the virtual environment used to create the essay (torch_env.yml)
