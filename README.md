# Adaptability

### Three steps are required to run the adaptability
- Update the current_path in creat_dir.sh, which creates folders for holding intermediate and final results. Run this script under linux enviroment by "./creat_dir.sh" 
2. Update the current_path in main.py
3. Run the main.py file. Since we built 20 adaptability models, we need to run the main.py for 20 times with different parameters. The file "adaptability_jobs.sh" includes these 20 commands for all the adaptability models. For example, "python main.py 1997_2004_1999_2001 1999 2001 1997 1998 2002 2004". Parameter 1, "1997_2004_1999_2001", is a label for the model. Parameter 2 & 3, "1999 2001", indicate the start and end year of the drugs in test set. Parameter 4 & 5, "1997 1998", indicate the start and end year of additional drugs added into the training set. Parameter 6 & 7, "2002 2004", indicate the start and end year of another additional drugs added into the training set. 

### Pakages information
python=3.7.3 
numpy=1.16.4 
pandas=0.24.2 
keras=2.2.4 
scikit-learn=0.21.2 
xgboost=0.90
