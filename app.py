from flask import Flask, request, jsonify # web provider
from flask_modus import Modus  # methods provider
from flaskext.mysql import MySQL  # datebase
from datetime import datetime
import os
import numpy as np
import re
import random
import json
from random import shuffle
import logging
#===================================================================
#                         import nlp class
#===================================================================
import tensorflow as tf
from NLP_model.model_sentence2token import token
from NLP_model.conversation_tag import tag
import keras.models
from keras.models import model_from_yaml,load_model
from keras.utils import to_categorical

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',level=logging.DEBUG)
global intence_model, question_model, graph, nameEntity_model
dir_path = os.getcwd()
word_vector = {}
with open(dir_path + '/NLP_model/dict/cc_vector.txt', 'r', encoding='utf-8-sig') as f:
    for i in f:
        i.replace("\n",'')
        temp = i.split()
        word_vector[temp[0]] = [float(temp[j]) for j in range(1,len(temp))]
    f.close()
question_stack = []
conver_stack = []
ans_pool = []

# web config
app = Flask(__name__)
app.config['MYSQL_DATABASE_HOST'] = '35.247.146.113'
app.config['MYSQL_DATABASE_PORT'] = 3306
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = '580610629'
app.config['MYSQL_DATABASE_DB'] = 'mysql_db'
modus = Modus(app)
mysql = MySQL()
mysql.init_app(app)

def create_logs(message_in, message_out, user_id, sys_question):
    con = mysql.connect()
    cur = con.cursor()
    timestamp = datetime.now()
    str_id = str(timestamp)
    str_id = re.sub(r'[^0-9]', r'', str_id)
    str_id = str_id[4:]
    cur.execute("INSERT INTO chat_log(id, timestamp, user_id, user_question, sys_answer, sys_question) VALUES(%s,%s, %s, %s, %s, %s)", 
                (str_id, timestamp, user_id, message_in, message_out, sys_question))
    con.commit()
    return str(str_id)

# API chat message
@app.route('/message/requestMessage', methods=["POST"])
def api_message():
    data = json.loads(request.form["json_string"])
    # ตัดคำและแยกคลาส
    message_token = token(data["message"])
    if len(message_token) > 35:
        message_token = message_token[0:36]
    # หา word vector
    message_vec = np.zeros((1,35,300))
    word_count = 0
    message_inModel = [None] * 35
    for i in message_token:
        try:
            message_vec[0,34-word_count,:] = word_vector[i]
            message_inModel[34-word_count] = i
            word_count += 1
        except:
            pass
    # ทำ NER
    name_tag = None
    RES_NAME = []
    MENU = []
    TIME = []
    CONTACT = []
    PRICE = []
    BRANCH = []
    LOCATION = re.findall(r'(https?://[^\s]+)', data["message"])
    reliability = 0
    with graph.as_default():
        name_tag = nameEntity_model.predict(message_vec)
    name_tag = name_tag.tolist()
    bound = 35-len(message_inModel)
    point = 34
    while point >= bound:
        max_index = name_tag[0][point].index(max(name_tag[0][point]))
        if max_index == 0:
            RES_NAME.append(message_inModel[point])
        elif max_index == 1:
            MENU.append(message_inModel[point])
        elif max_index == 2:
            TIME.append(message_inModel[point])
        elif max_index == 3:
            CONTACT.append(message_inModel[point])
        elif max_index == 4:
            PRICE.append(message_inModel[point])
        elif max_index == 5:
            BRANCH.append(message_inModel[point])
        else:
            pass
        point -= 1

    try:
        joint_arr = message_token[message_token.index(RES_NAME[0]):message_token.index(RES_NAME[len(RES_NAME)-1])+1]
        RES_NAME = "".join(joint_arr)
        reliability += 1
    except:
        RES_NAME = ""
        
    try:
        joint_arr = message_token[message_token.index(MENU[0]):message_token.index(MENU[len(MENU)-1])+1]
        MENU = "".join(joint_arr)
        reliability += 1
    except:
        MENU = ""

    try:
        joint_arr = message_token[message_token.index(TIME[0]):message_token.index(TIME[len(TIME)-1])+1]
        TIME = "".join(joint_arr)
        reliability += 1
    except:
        TIME = ""

    try:
        joint_arr = message_token[message_token.index(CONTACT[0]):message_token.index(CONTACT[len(CONTACT)-1])+1]
        CONTACT = "".join(joint_arr)
        reliability += 1
    except:
        CONTACT = ""

    try:
        joint_arr = message_token[message_token.index(PRICE[0]):message_token.index(PRICE[len(PRICE)-1])+1]
        PRICE = "".join(joint_arr)
        reliability += 1
    except:
        PRICE = ""

    try:
        joint_arr = message_token[message_token.index(BRANCH[0]):message_token.index(BRANCH[len(BRANCH)-1])+1]
        BRANCH = "".join(joint_arr)
        reliability += 1
    except:
        BRANCH = ""
    
    logging.info("RES NAME: {}".format(RES_NAME))
    logging.info("MENU: {}".format(MENU))
    logging.info("TIME: {}".format(TIME))
    logging.info("CONTACT: {}".format(CONTACT))
    logging.info("PRICE: {}".format(PRICE))
    logging.info("BRANCH: {}".format(BRANCH))
    logging.info("LOCATION: {}".format(LOCATION))
    
    # หาว่าเป็นคลาสไหน
    predict_result = None
    with graph.as_default():
        predict_result = intence_model.predict(message_vec)
    predict_result = predict_result.tolist()
    logging.info("intense predict: {}".format(predict_result))
    predict_result = predict_result[0].index(max(predict_result[0]))
    if predict_result == 0:
        logging.debug("RESTAURANT QUESTION CASE")
        question_type = None
        with graph.as_default():
            question_type = question_model.predict(message_vec)
        question_type = question_type.tolist()
        logging.info("question type predict: {}".format(question_type))
        question_type = question_type[0].index(max(question_type[0]))

        sending_message = None
        con = mysql.connect()
        cur = con.cursor()
        if question_type == 0:
            logging.debug("MENU CASE")
            cur.execute("SELECT tag,id FROM restaurant_tag")
            temp = cur.fetchall()
            temp = [list(t) for t in temp]
            logging.info("Search result: {}".format(len(temp)))
            shuffle(temp)
            MENU_NAME = temp[0][0]
            MENU = temp[0][1]
            cur.execute("SELECT answer FROM template_answer WHERE answer LIKE %s",('%MENU%'))
            temp = cur.fetchall()
            temp = [list(t) for t in temp]
            shuffle(temp)
            sending_message = temp[0][0]
            sending_message = re.sub(r'MENU',MENU_NAME,sending_message)
        elif question_type == 1:
            logging.debug("RESTAURANT CASE")
            logging.info("menu id: {}".format(data["menu_id"]))
            if BRANCH != "":
                logging.info("Search form branch")
                cur.execute("SELECT res_id FROM restaurant_branch WHERE branch LIKE %s",('%'+BRANCH+'%'))
            elif MENU != "":
                logging.info("Search form menu")
                cur.execute("SELECT res_id FROM restaurant_tag WHERE tag LIKE %s",('%'+MENU+'%'))
            elif data["menu_id"] != -1:
                logging.info("Search form menu id: {}".format(data["menu_id"]))
                cur.execute("SELECT res_id FROM restaurant_tag WHERE id=%",(data["menu_id"]))
            else:
                logging.info("Search random")
                cur.execute("SELECT id FROM restaurant_info")
            temp = cur.fetchall()
            RN = None
            temp = [list(t) for t in temp]
            logging.info("Search result: {}".format(len(temp)))
            shuffle(temp)
            RES_NAME = temp[0][0]
            cur.execute("SELECT name FROM restaurant_info WHERE id=%s",(RES_NAME))
            temp = cur.fetchone()
            RN = temp[0]
            cur.execute("SELECT answer FROM template_answer WHERE answer LIKE %s",('%RN%'))
            temp = cur.fetchall()
            temp = [list(t) for t in temp]
            shuffle(temp)
            sending_message = temp[0][0]
            sending_message = re.sub(r'RN',RN,sending_message)
        elif question_type == 2:
            logging.debug("RESTAURANT TYPE CASE")
            RT = None
            if RES_NAME == "" and data["res_topic"] != -1:
                logging.info("Search form res_id: {}".format(data["res_topic"]))
                cur.execute("SELECT tag FROM restaurant_tag WHERE 	res_id=%s",(data["res_topic"]))
                temp = cur.fetchall()
                logging.info("Search tag: {}".format(len(temp)))
                if len(temp) > 0:
                    RT = " ".join([temp[i][0] for i in range(len(temp))])
                RES_NAME = data["res_topic"]    
            else:
                try:
                    logging.info("Search form name: {}".format(RES_NAME))
                    cur.execute("SELECT id FROM restaurant_info WHERE name LIKE %s",('%'+RES_NAME+'%'))
                    temp = cur.fetchone()
                    RES_NAME = temp[0]
                    cur.execute("SELECT tag FROM restaurant_tag WHERE res_id=%s",(RES_NAME))
                    temp = cur.fetchall()
                    logging.info("Search tag: {}".format(len(temp)))
                    if len(temp) > 0:
                        RT = " ".join([temp[i][0] for i in range(len(temp))])            
                except Exception as e:
                    RES_NAME = -1
                    MENU = -1
                    question_stack.append([data["userID"],data["message"],0])
                    logging.debug("QUESTION ADDED")
                    logging.info("restaurant stack: {}".format(question_stack))
                    sending_message = "ไม่รู้จักร้านนี้อ่ะ เดี่ยวถามเพื่อนแปป"
                    cur.close()
                    log_id = create_logs(data["message"], sending_message, data["userID"], "")
                    logging.info("previous: {}".format(data["message"]))
                    logging.info("message reply: {}".format(sending_message))
                    logging.info("question: {}".format(sys_question))
                    logging.info("log id: {}".format(log_id))
                    return jsonify(userID=data["userID"],previous_message=data["message"],message=sending_message,
                                    sys_question="",res_topic=RES_NAME,menu_id=MENU,log_id=log_id,request_count=data["request_count"] + 1)
                    
            if RT != None:
                cur.execute("SELECT answer FROM template_answer WHERE answer LIKE %s",('%RT%'))
                temp = cur.fetchall()
                temp = [list(t) for t in temp]
                shuffle(temp)
                sending_message = temp[0][0]
                sending_message = re.sub(r'RT',RT,sending_message)
            else:
                sending_message = "ไม่รู้เหมือนกันอ่ะว่าเป็นร้านอะไร"
        elif question_type == 3:
            logging.debug("PRICE CASE")
            RP = None
            if RES_NAME == "" and data["res_topic"] != -1:
                logging.info("Search form res_id: {}".format(data["res_topic"]))
                cur.execute("SELECT price FROM restaurant_info WHERE id=%s",(data["res_topic"]))
                temp = cur.fetchone()
                RP = temp[0]
                RES_NAME = data["res_topic"]
            else:
                try:
                    logging.info("Search form name: {}".format(RES_NAME))
                    cur.execute("SELECT price,id FROM restaurant_info WHERE name LIKE %s",('%'+RES_NAME+'%'))
                    temp = cur.fetchone()
                    RES_NAME = temp[1]
                    RP = temp[0]            
                except Exception as e:
                    RES_NAME = -1
                    MENU = -1
                    question_stack.append([data["userID"],data["message"],0])
                    logging.debug("QUESTION ADDED")
                    logging.info("restaurant stack: {}".format(question_stack))
                    sending_message = "ไม่รู้จักร้านนี้อ่ะ เดี่ยวถามเพื่อนแปป"
                    cur.close()
                    log_id = create_logs(data["message"], sending_message, data["userID"], "")
                    logging.info("previous: {}".format(data["message"]))
                    logging.info("message reply: {}".format(sending_message))
                    logging.info("question: {}".format(sys_question))
                    logging.info("log id: {}".format(log_id))
                    return jsonify(userID=data["userID"],previous_message=data["message"],message=sending_message,
                                    sys_question="",res_topic=RES_NAME,menu_id=MENU,log_id=log_id,request_count=data["request_count"] + 1)
            if RP != None:
                cur.execute("SELECT answer FROM template_answer WHERE answer LIKE %s",('%RP%'))
                temp = cur.fetchall()
                temp = [list(t) for t in temp]
                shuffle(temp)
                sending_message = temp[0][0]
                sending_message = re.sub(r'RP',RP,sending_message)
            else:
                sending_message = "ไม่รู้ราคาอาหารร้านนี้อ่ะ"
        elif question_type == 4:
            logging.debug("TIME CASE")
            RO = None
            if RES_NAME == "" and data["res_topic"] != -1:
                logging.info("Search form res_id: {}".format(data["res_topic"]))
                cur.execute("SELECT time FROM restaurant_info WHERE id=%s",(data["res_topic"]))
                temp = cur.fetchone()
                RO = temp[0]
                RES_NAME = data["res_topic"]
            else:
                try:
                    logging.info("Search form name: {}".format(RES_NAME))
                    cur.execute("SELECT time,id FROM restaurant_info WHERE name LIKE %s",('%'+RES_NAME+'%'))
                    temp = cur.fetchone()
                    RES_NAME = temp[1]
                    RO = temp[0]            
                except Exception as e:
                    RES_NAME = -1
                    MENU = -1
                    question_stack.append([data["userID"],data["message"],0])
                    logging.debug("QUESTION ADDED")
                    logging.info("restaurant stack: {}".format(question_stack))
                    sending_message = "ไม่รู้จักร้านนี้อ่ะ เดี่ยวถามเพื่อนแปป"
                    cur.close()
                    log_id = create_logs(data["message"], sending_message, data["userID"], "")
                    logging.info("previous: {}".format(data["message"]))
                    logging.info("message reply: {}".format(sending_message))
                    logging.info("question: {}".format(sys_question))
                    logging.info("log id: {}".format(log_id))
                    return jsonify(userID=data["userID"],previous_message=data["message"],message=sending_message,
                                    sys_question="",res_topic=RES_NAME,menu_id=MENU,log_id=log_id,request_count=data["request_count"] + 1)

            if RO != None:
                cur.execute("SELECT answer FROM template_answer WHERE answer LIKE %s",('%RO%'))
                temp = cur.fetchall()
                temp = [list(t) for t in temp]
                shuffle(temp)
                sending_message = temp[0][0]
                sending_message = re.sub(r'RO',RO,sending_message)
            else:
                sending_message = "ไม่รู้เวลาร้านนี้อ่ะ"
        elif question_type == 5:
            logging.debug("LOCATION CASE")
            RL = None
            if RES_NAME == "" and data["res_topic"] != -1:
                logging.info("Search form res_id: {}".format(data["res_topic"]))
                cur.execute("SELECT address FROM restaurant_info WHERE id=%s",(data["res_topic"]))
                temp = cur.fetchone()
                RL = temp[0]
                RES_NAME = data["res_topic"]
            else:
                try:
                    logging.info("Search form name: {}".format(RES_NAME))
                    cur.execute("SELECT address,id FROM restaurant_info WHERE name LIKE %s",('%'+RES_NAME+'%'))
                    temp = cur.fetchone()
                    RES_NAME = temp[1]
                    RL = temp[0]   
                except Exception as e:
                    RES_NAME = -1
                    MENU = -1
                    question_stack.append([data["userID"],data["message"],0])
                    logging.debug("QUESTION ADDED")
                    logging.info("restaurant stack: {}".format(question_stack))
                    sending_message = "ไม่รู้จักร้านนี้อ่ะ เดี่ยวถามเพื่อนแปป"
                    cur.close()
                    log_id = create_logs(data["message"], sending_message, data["userID"], "")
                    logging.info("previous: {}".format(data["message"]))
                    logging.info("message reply: {}".format(sending_message))
                    logging.info("question: {}".format(sys_question))
                    logging.info("log id: {}".format(log_id))
                    return jsonify(userID=data["userID"],previous_message=data["message"],message=sending_message,
                                    sys_question="",res_topic=RES_NAME,menu_id=MENU,log_id=log_id,request_count=data["request_count"] + 1)
            if RL != None:
                sending_message = RL
            else:
                sending_message = "ไม่มีลิ้งค์ร้านอ่ะ"
        else:
            logging.debug("CONTACT CASE")
            RC = None
            if RES_NAME == "" and data["res_topic"] != -1:
                logging.info("Search form res_id: {}".format(data["res_topic"]))
                cur.execute("SELECT contact FROM restaurant_info WHERE id=%s",(data["res_topic"]))
                temp = cur.fetchone()
                RC = temp[0]
                RES_NAME = data["res_topic"]
            else:
                try:
                    logging.info("Search form name: {}".format(RES_NAME))
                    cur.execute("SELECT contact,id FROM restaurant_info WHERE name LIKE %s",('%'+RES_NAME+'%'))
                    temp = cur.fetchone()
                    RES_NAME = temp[1]
                    RC = temp[0]            
                except Exception as e:
                    RES_NAME = -1
                    MENU = -1
                    question_stack.append([data["userID"],data["message"],0])
                    logging.debug("QUESTION ADDED")
                    logging.info("restaurant stack: {}".format(question_stack))
                    sending_message = "ไม่รู้จักร้านนี้อ่ะ เดี่ยวถามเพื่อนแปป"
                    cur.close()
                    log_id = create_logs(data["message"], sending_message, data["userID"], "")
                    logging.info("previous: {}".format(data["message"]))
                    logging.info("message reply: {}".format(sending_message))
                    logging.info("question: {}".format(sys_question))
                    logging.info("log id: {}".format(log_id))
                    return jsonify(userID=data["userID"],previous_message=data["message"],message=sending_message,
                                    sys_question="",res_topic=RES_NAME,menu_id=MENU,log_id=log_id,request_count=data["request_count"] + 1)
            if RC != None:
                cur.execute("SELECT answer FROM template_answer WHERE answer LIKE %s",('%RC%'))
                temp = cur.fetchall()
                temp = [list(t) for t in temp]
                shuffle(temp)
                sending_message = temp[0][0]
                sending_message = re.sub(r'RC',RC,sending_message)
            else:
                sending_message = "ไม่มีเบอร์ร้านนี้อ่ะ"
        cur.close()
        log_id = create_logs(data["message"], sending_message, data["userID"], "")

        if RES_NAME is not int:
            RES_NAME = -1
        if MENU is not int:
            MENU = -1
        logging.debug("LOG CREATED")
        logging.info("res_id: {}".format(RES_NAME))
        logging.info("previous: {}".format(data["message"]))
        logging.info("message reply: {}".format(sending_message))
        logging.info("log id: {}".format(log_id))
        logging.info("menu id: {}".format(MENU))
        return jsonify(userID=data["userID"],previous_message=data["message"],message=sending_message,
                                    sys_question="",res_topic=RES_NAME,menu_id=MENU,log_id=log_id,request_count=data["request_count"] + 1)
    elif predict_result == 1:
        # con = mysql.connect()
        # cur = con.cursor()
        # try:
        #     if reliability >= ans_pool[7]:
        #         if RES_NAME != "":
        #             ans_pool[0] = RES_NAME
        #         if PRICE != "":
        #             ans_pool[1] = PRICE
        #         if TIME != "":
        #             ans_pool[2] = TIME
        #         if CONTACT != "":
        #             ans_pool[3] = CONTACT
        #         if len(LOCATION) > 0:
        #             ans_pool[4] = LOCATION[0]
        #         if BRANCH != "":
        #             ans_pool[5] = BRANCH
        #         if MENU != "":
        #             ans_pool[6] = MENU
        #         ans_pool[7] = reliability
        #         ans_pool[8] += 1
        
        #     if ans_pool[8] >= 3:
        #         try:
        #             cur.execute("INSERT INTO restaurant_info(name, price, time, contact, address) VALUES(%s, %s, %s, %s, %s)", 
        #                         (ans_pool[0], ans_pool[1], ans_pool[2], ans_pool[3], ans_pool[4]))
        #             con.commit()
        #             cur.execute("SELECT id FROM restaurant_info WHERE name=%s",(ans_pool[0]))
        #             temp = cur.fetchone()
        #             if ans_pool[5] != None:
        #                 cur.execute("INSERT INTO restaurant_branch(res_id, branch) VALUES(%s, %s)", 
        #                             (temp[0], ans_pool[5]))
        #                 con.commit()
        #             if ans_pool[6] != None:
        #                 cur.execute("INSERT INTO restaurant_tag(res_id, tag) VALUES(%s, %s)", 
        #                             (temp[0], ans_pool[6]))
        #                 con.commit()
        #             question_stack.pop(0)
        #             ans_pool = None
        #         except Exception as e:
        #             print(e)
        #             pass
        # except:
        #     pass
        # print(ans_pool)
        # log_id = create_logs(data["message"], "message_out", data["userID"], "question_out")
        # logging.info("log id: {}".format(log_id))
        return jsonify(userID=data["userID"],previous_message="",message="receive information: " + str(reliability),
                   sys_question="",res_topic="",menu_id="",log_id=log_id,request_count=data["request_count"] + 1)
    else:
        logging.debug("CONVERSATION CASE")
        logging.info("user message: {}".format(data["message"]))
        logging.info("previous message: {}".format(data["previous_message"]))
        message_out = None
        previous_message = None
        sys_question = ""
        con = mysql.connect()
        cur = con.cursor()

        try:
            if conver_stack[0][0] == data["previous_message"]:
                general_message = tag(conver_stack[0][0])
                cur.execute("INSERT INTO template_conversation(sentence_in, sentence, use_count) VALUES(%s, %s, %s)", 
                        (general_message, data["message"], 0))
                con.commit()
                logging.debug("ANSWER CONVER ADDED")
                message_out = "โอเคคราวหน้าเราจะได้ตอบถูก"
                conver_stack[0][1] += 1
                if conver_stack[0][1] > 5:
                    conver_stack.pop(0)
                return jsonify(userID=data["userID"],previous_message="",message=message_out,
                                sys_question="",res_topic=-1,menu_id=-1,log_id="",request_count=0)
        except Exception as e:
            if e is IndexError:
                pass
                
        if data["previous_message"] != "":
            general_message = tag(data["previous_message"])
            cur.execute("SELECT * FROM template_conversation WHERE sentence_in=%s AND sentence=%s",
                        (general_message, data["message"]))
            temp = cur.fetchone()
            if not temp:
                cur.execute("INSERT INTO template_conversation(sentence_in, sentence, use_count) VALUES(%s, %s, %s)", 
                    (general_message, data["message"], 0))
                con.commit()
                logging.debug("PAIR SENTENCE CREATED")

        general_message = tag(data["message"])
        logging.info("Normalization: {}".format(general_message))
        cur.execute("SELECT * FROM template_conversation WHERE sentence_in=%s",(general_message))
        temp = cur.fetchall()
        temp = [list(t) for t in temp]
        shuffle(temp)
        if len(temp) > 0:
            cur.execute ("UPDATE template_conversation SET use_count=%s WHERE id=%s",
                        (temp[0][3] + 1, temp[0][0]))
            con.commit()
            previous_message = temp[0][2]
            message_out = temp[0][2]
        else:
            conver_stack.append([data["message"],0])
            logging.debug("QUESTION STACKED")
            logging.info("question stack: {}".format(conver_stack))
            previous_message = ""
            message_out = "เอิ่มหมายถึงอะไรเหรอ"
        
        req = data["request_count"]
        treshold = 1 - (req/10)
        alpha = random.random() - 0.1
        if alpha > treshold:
            req = 0
            try:
                sys_question = "นี่ถามหน่อย ถ้ามีคนถามว่า \"" + conver_stack[0][0] + "\" จะตอบยังไงดี"
                previous_message = conver_stack[0][0]
                logging.debug("QUESTION ADDED")
            except Exception as e:
                pass

        cur.close()
        log_id = create_logs(data["message"], message_out, data["userID"], sys_question)
        logging.debug("LOG CREATED")
        logging.info("previous: {}".format(previous_message))
        logging.info("message reply: {}".format(message_out))
        logging.info("question: {}".format(sys_question))
        logging.info("log id: {}".format(log_id))
        return jsonify(userID=data["userID"],previous_message=previous_message,message=message_out,
                       sys_question=sys_question,res_topic=-1,menu_id=-1,log_id=log_id,request_count=req + 1)

# API user provice
@app.route('/user/', methods=["POST"])
def api_user():
    try:
        con = mysql.connect()
        cur = con.cursor()
        cur.execute("INSERT INTO user(user_id,last_login) VALUES(%s, %s)", 
                    (request.form["userID"], request.form["lastLogin"]))
        con.commit()
        cur.close()
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'}
    except Exception as e:
        return json.dumps({'error':e, 'success':False}), 500, {'ContentType':'application/json'}

@app.route('/user/Login', methods=["POST"])
def api_userLogin():
    try:
        con = mysql.connect()
        cur = con.cursor()
        cur.execute ("UPDATE user SET last_login=%s WHERE user_id=%s",
                    (request.form["lastLogin"], request.form["userID"]))
        con.commit()
        cur.close()
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'}
    except Exception as e:
        return json.dumps({'error':e, 'success':False}), 500, {'ContentType':'application/json'}

#API check signal
@app.route('/signal', methods=["POST"])
def api_replySignal():
    # con = mysql.connect()
    # cur = con.cursor()
    # data = json.loads(request.form["json_string"])

    # print("question stack: {}".format(question_stack))
    # print("conversation stack: {}".format(conver_stack))
    # # เช็คใน tabel คิวคำตอบ
    # cur.execute("SELECT * FROM restaurant_answer WHERE user_id=%s",(data["userID"]))
    # reply_answer = cur.fetchone()
    # if not reply_answer:
    #     pass
    # else:
    #     cur.execute("DELETE FROM restaurant_answer WHERE id=%s",(reply_answer[0]))
    #     con.commit()
    #     cur.execute("SELECT * FROM restaurant_info WHERE id=%s",(reply_answer[1]))
    #     reply_answer = cur.fetchone()
    #     cur.close()
    #     sending_message = "เออร้านที่ถามครั้งก่อนอ่ะร้าน" + reply_answer[1] + "รู้ข้อมูลแล้วนะถามมาได้เลย"
    #     return jsonify(userID=data["userID"],reply_data=sending_message,res_id=reply_answer[0],
    #                    stage_data="answer")
    
    # # เช็คใน tabel คิวคำถามร้านอาหาร
    # if len(question_stack) == 0:
    #     pass
    # else:
    #     if question_stack[0][0] == data["userID"]:
    #         sending_message = "ผมมีคำถามครับแนะนำข้อมูลร้าน " + question_stack[0][1] + " หน่อยสิ"
    #         question_stack[0][2] += 1
    #         if question_stack[0][2] > 5:
    #             question_stack.pop(0)
    #             ans_pool = [None,None,None,None,None,None,None,0,0]
    #         return jsonify(userID=data["userID"],reply_data=sending_message,res_id="",stage_data="question")

    # return jsonify(userID=data["userID"],reply_data="",res_id="",stage_data="")
    return jsonify(userID="",reply_data="",res_id="",stage_data="")

#API report log
@app.route('/log/report', methods=["POST"])
def api_logReport():
    if request.form["log_id"] == "":
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'}
    try:
        con = mysql.connect()
        cur = con.cursor()
        cur.execute("UPDATE chat_log SET report=%s WHERE id=%s",
                    (request.form["report_type"], request.form["log_id"]))
        con.commit()
        cur.close()
        logging.debug("LOG {} HAS REPORT".format(request.form["log_id"]))
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'}
    except Exception as e:
        return json.dumps({'error':e, 'success':False}), 500, {'ContentType':'application/json'}

if __name__ == '__main__':
    yaml_file = open(dir_path + '/NLP_model/intence.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    intence_model = model_from_yaml(loaded_model_yaml)
    intence_model.load_weights(dir_path + "/NLP_model/intence.h5")

    yaml_file = open(dir_path + '/NLP_model/question.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    question_model = model_from_yaml(loaded_model_yaml)
    question_model.load_weights(dir_path + "/NLP_model/question.h5")

    yaml_file = open(dir_path + '/NLP_model/nameEntity.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    nameEntity_model = model_from_yaml(loaded_model_yaml)
    nameEntity_model.load_weights(dir_path + "/NLP_model/nameEntity.h5")

    graph = tf.get_default_graph()
        
    app.run(host = '0.0.0.0', port = 80)