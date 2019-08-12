The folder structure is as follows:
- learning: contains the respective learning algorithms
- visualizations: contains the algorithms for visualizing code
- datasets: contains the scripts for extracting the different datasets (bcb database has to be running!) and contains the extraction results; further is also used to store the images (configurable)
- bcb_stats: contains a jupyter notebook, used to create the statistics on the BCB dataset (bcb database has to be running!)


First steps:
- Install the BigCloneBench as described in https://github.com/jeffsvajlenko/BigCloneEval.
- Run the database:
  - Get the h2 client jar file from http://www.h2database.com/html/download.html
  - Run the command from DB_run_command.txt, from a folder containing the "h2-1.x.xxx.jar" and by replacing the path to the bceval from the example command.
  - Access the database via browser and update the password of the "sa" user to "sa" (unset by default)
- Run scripts:
  - Set various paths throughout the python files. (grep for "Path must be set")
  - You can run all the experiments by calling the pipeline.sh script from this folder.
  - The respective juypter notebooks must be run from a jupyter installation 

Note: You might encounter various dead or out-commented code
