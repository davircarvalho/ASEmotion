# ASEmotion
Audio speech emotion recognition using Temporal Convolutional Networks (TCNs)

## Folder structure
- src: Stores source code (python) which serves multiple scenarios. During data exploration and model training, we have to transform data for particular purpose. We have to use same code to transfer data during online prediction as well. So it better separates code from notebook such that it serves different purpose.
- test: In R&D, data science focus on building model but not make sure everything work well in unexpected scenario. However, it will be a trouble if deploying model to API. Also, test cases guarantee backward compatible issue but it takes time to implement it.
- model: Folder for storing binary (json or other format) file for local use.
- data: Folder for storing subset data for experiments. It includes both raw data and processed data for temporary use.
- notebook: Storing all notebooks includeing EDA and modeling stage.


## Training 
The pipeline begins with getting the datasets, have a look at the <src/preparation'> folder, after that preprocessing the datasets with Preprocess_datasets.py at <src/procesing>, finally the trainning function (smile_Train.py) can be found at <src/modeling>.

*All functions were developed with vscode's interactive python, which is basically a jupyter notebook, but in a .py file. If you want to run them in the terminal, you'll need to add* if __name__ == "__main__": *manually.*