import nltk
from nltk.stem.lancaster import LancasterStemmer
import random
import json
import pickle
import requests
import base64
import os
import socket
from mictest import Voice
import time
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    # warnings.filterwarnings("ignore",category=DeprecationWarning)
    import tflearn, tensorflow, numpy

stemmer = LancasterStemmer()
os.system("/usr/bin/python rest.py")
s = requests.session()
SERVER_IP = "SERVER_IP_HERE"
SERVICE_ENDPOINT = "http://"+ SERVER_IP+"/voice.php?"

with open("intents.json") as file:
    data= json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds=nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag=[]

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")

except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w ==se:
                bag[i]= 1
    return numpy.array(bag)

def chat():
    print("Start talking with the bot!")
    current_context = "custom"
    while True:
        time_start=time.time()
        tags = ["closed_domain", "vqa", "caption"]
        voiceobject = Voice()
        text=voiceobject.Voice_return()
        print(text)
        time_text=time.time()
        time_input=time_text-time_start
        print(time_input)
        if ("please stop" in text):
            break
        results = model.predict([bag_of_words(text, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            #print(random.choice(responses))
            os.system("/usr/bin/python try.py \""+random.choice(responses)+"\"")
            time_answer=time.time()
            answering_time=time_answer-time_text
            print(answering_time)
            if (tag in tags) and (tag != current_context):
                current_context = tag
                if tag == "closed_domain":
                    while True: #it stays in closed domain until please stop is spoken
                        closeddomaintalk = Voice()
                        closed_text = closeddomaintalk.Voice_return()
                        print("You said: "+closed_text)
                        time_closed_start=time.time()
                        if("please stop" in closed_text):
                            print("ended closed domain")
                            closed_text=""
                            current_context = "custom"
                            break
                        resp = s.get(SERVICE_ENDPOINT+"txt="+base64.b64encode(closed_text.encode()).decode()+"&context=closed")
                        response = json.loads(resp.text)
                        time_closed_response=time.time()
                        print("Request time: ",time_closed_start-time_closed_response)
                        print(resp.text)
                        os.system("/usr/bin/python try.py \""+response["result"]+"\"")
                elif tag == "vqa":
                    os.system("/usr/bin/python Image.py")
                    file = {'file': open('camImage.png', 'rb')}
                    resp = s.post(SERVICE_ENDPOINT+"context=vqa", files=file)
                    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    c.connect((SERVER_IP, 8888))
                    os.system("/usr/bin/python try.py \""+"picture taken! ready for a question"+"\"")
                    while True:
                        print("Ready for a question")
                        vqatalk = Voice()
                        vqa_text = vqatalk.Voice_return()
                        time_vqa_start=time.time()
                        if("please stop" in vqa_text):
                            print("ended vqa session")
                            c.close()
                            vqa_text=""
                            current_context = "custom"
                            break
                        print("question asked: "+vqa_text)
                        c.send(vqa_text.encode("utf-8"))
                        response = c.recv(1024).decode("utf-8")
                        time_vqa_response=time.time()
                        print("VQA Request delay: ",time_vqa_start-time_vqa_response)
                        os.system("/usr/bin/python try.py \""+response+"\"")
                elif tag == "caption":
                    os.system("/usr/bin/python Image.py")
                    file = {'file': open('camImage.png', 'rb')}
                    time_caption_start=time.time()
                    resp = s.post(SERVICE_ENDPOINT+"context=caption", files=file)
                    os.system("/usr/bin/python try.py \""+"picture taken! Analyzing"+"\"")
                    cap = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    cap.connect((SERVER_IP, 9999))
                    cap_response = cap.recv(1024).decode("utf-8")
                    os.system("/usr/bin/python try.py \"I see "+cap_response+"\"")
                    cap.close()
                    time_caption_response=time.time()
                    print("Caption request delay: ",time_caption_start-time_caption_response)
                    current_context = "custom"
        else:
            os.system("/usr/bin/python try.py \"I didn't get the question, please try again!\"")


chat()
