# Weighted-Overlook-Graph
This repository holds the source code of Overlook Graph method.
The code consists of three parts: Data, Complex_network, and 2D-CNN.

Data

Due to the size limitation of upload file, we provide source code for batch processing of data. The Bonn dataset can be obtained from：http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3.

Complex_network

This part contains four code files, LVG.m, LPHVG.m, overlook.m, and Weighted_Overlook.m. In LPVG and LPHVG, we set the variable L of limited number of penetrations. When you set L to 0, you can build VG and HVG.

2D-CNN

This part contains three code files, input_data.py, model.py and training.py. The main file is training.py. To use this classifier, please keep all three files in a sinlge folder, and make sure that you have Tensorflow, Python 3 and all the packages we have used installed.

Next, please take the following two steps.

  Step 1. Change the path in line 20 and 21 of training.py to the train and test folder's path.
  
  Step 2. Run the command in your command council.
