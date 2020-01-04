# NLP-tasks-with-Nao-Robot
Final Year Project for Master of Engineering at ECU

This project was developed to improve the human-robot interaction with humanoid Nao Robot. The project involves using AllenNLP for NLP tasks and pythia for visual questioning and answering.

The project also involved testing both local and server implementation. The code found on this repo is specifically of local + machine implementation.
The local machine runs a main program which makes use of softmax activated 2 layer Deep Neural Network that is trained on intents from intents.json file.
The intents descirbe what user is speaking and a corresponding response. After that for the demo, it can go into 3 different states:
1. Closed Domain
2. Visual Questioning and Answering
3. Captioning

The closed domain talk is basically the user telling the robot things and then the robot is able to answer back the user's question from the information the robot was provided. This is done to show that robot can be given capabilities to understand the human context and answer back in the same context.
The closed domain part makes use of AllenNLP.

The VQA and image captioning are parts where Pythia is used. The robot takes and image and sends that to the sever. Then it asks the user for the queries and based on what the image contains, the robot answers back.
The captioning is making a caption of the picture the robot took.

During the demo, the robot was able to tell if the robot sees certain objects, colors of various things, detect the number of people in the frame, etc.

The resutls can be improved by using training data for a specific application especaially of the environment in which the robot is to be used.

Note: I am not incuding the Pythia code as it depends on you, how and what all you want to make the robot answer questions on. For more look into the Pythia repo here: https://github.com/facebookresearch/pythia

Demo Video: https://youtu.be/fhbk0k7_Cfo
