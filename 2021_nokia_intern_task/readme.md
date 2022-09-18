# AI/ML Nokia assignment
### STRUCTURE: 
* Source in Solution_Kamil_Bartocha.ipynb
* models directory contains 17 CA models saved in pickle 
* dataset directiry contains 17 CA sets and 17 CA labels in .csv 
* Description.docx - task defined and data description

### DATA DESCRIPTION

For each CA 2 csv files are created: x_train and y_train. x_train file has a vector of rational numbers per each row. These numbers depict assessment of an object via given assessment model. Values in single column (i.e. column 1) of one file (i.e. x_train_CA1) were produced with same model. Corresponding y_train has an expected binary label (with a noise) related to a row in x_train file. In that sense y=f_CA (x).

### GOAL

Using provided data and performing its analysis build prediction model that based on x vector will produce y value without knowing function f_CA (frankly, we do not know it either!).

### EXPECTED OUTCOME

A Jupyter (not to be mixed with Jupiter, we are not having any premises on that planet) notebook containing description of data, its analysis and information about reason and purpose of each step, its input, output and result. Notebook should produce a final model or models that based on x vector predicts y value for each CA with analysis of results.
