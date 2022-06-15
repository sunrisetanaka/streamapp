import requests
from bs4 import BeautifulSoup
import time
import re
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import Model
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,LSTM,Dropout,concatenate
from keras.optimizers import Adam
import tensorflow as tf
from keras import Input
import warnings
warnings.simplefilter("ignore",FutureWarning)
import streamlit as st

#日付取得
dt_now=datetime.datetime.now()
year_now=dt_now.year
month_now=dt_now.month
date_now=dt_now.day

window_size=30

toshidict={
    "大阪":["osaka",62,47772],
    "京都":["kyoto",61,47759],
    "広島":["hiroshima",67,47765],
    "高知":["kouti",74,47893],
    "熊本":["kumamoto",86,47819],
    "那覇":["naha",91,47936],
    "名古屋":["nagoya",51,47636],
    "金沢":["kanazawa",56,47605],
    "東京":["tokyo",44,47662],
    "新潟":["nigata",54,47604],
    "仙台":["sendai",34,47590],
    "札幌":["sapporo",14,47412]
}

def main():
    st.title("Kion-Stream(全国版)")
    st.write("今日の気温を予測します")
    selected_item=st.selectbox("地点を選択",["地点を選択","京都","東京","大阪","名古屋","広島","札幌","仙台","金沢","高知","熊本","新潟","那覇"])
    if selected_item!="地点を選択":
        app(selected_item)
    
def app(selected_item):
    if st.button("{}の予測を始める(絶対に連打しない)".format(selected_item)):
        stream(toshidict[selected_item][0],toshidict[selected_item][1],toshidict[selected_item][2])
    
def stream(toshi,prec,block):
    st.write("データ収集中...")
    nowdf=now_df_scrap(prec,block)
    df=pd.read_csv("df_{}.csv".format(toshi))
    st.write("データ確保")
    st.write("計算中...")
    mid,top,bot=pred(toshi,nowdf,df)
    st.write("計算終了")
    st.success("最低気温:{}".format(bot))
    st.success("平均気温:{}".format(mid))
    st.success("最高気温:{}".format(top))


def now_df_scrap(prec,block):  

    #データを分ける
    monthdict={
        "年":[],
        "月":[],
        "日":[],
        "平均気温":[],
        "最高気温":[],
        "最低気温":[],
        "日照時間":[]
    }

    param=["平均気温","最高気温","最低気温","日照時間"]
  
    if date_now<30:
        #もし1月なら
        if month_now==1:
            load_url="https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no={}&block_no={}&year={}&month={}&day=&view=".format(prec,block,str(year_now-1),str(12))

            html=requests.get(load_url)
            soup=BeautifulSoup(html.content,"html.parser")
            ifdata=[]
            for td in soup.find_all("td"):
                ifdata.append(td.text)

            #中身がない要素を削除
            ifdata=[i for i in ifdata if i!='']


            #検索
            year=year_now-1
            month=12

            #データを詰め込んでいく
            for i in ifdata[::21]:
                monthdict["年"].append(year)
                monthdict["月"].append(month)
                monthdict["日"].append(i)

            #データ加工(最後だけdelete)
            while monthdict["日"][-1]!=str(31):
                monthdict["年"].pop()
                monthdict["月"].pop()
                monthdict["日"].pop()

            #他も詰め込む　
            for i in ifdata[6::21]:
                monthdict["平均気温"].append(i)

            for i in ifdata[7::21]:
                monthdict["最高気温"].append(i)

            for i in ifdata[8::21]:
                monthdict["最低気温"].append(i)

            for i in ifdata[16::21]:
                monthdict["日照時間"].append(i)

            #確認    
            while len(monthdict["平均気温"])!=len(monthdict["日"]) or len(monthdict["最高気温"])!=len(monthdict["日"]) or len(monthdict["最低気温"])!=len(monthdict["日"]) or len(monthdict["日照時間"])!=len(monthdict["日"]):
                for i in param:
                    monthdict[i].pop()

        #1月以外         
        else:
            load_url="https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no={}&block_no={}&year={}&month={}&day=&view=".format(prec,block,str(year_now),str(month_now-1))

            html=requests.get(load_url)
            soup=BeautifulSoup(html.content,"html.parser")
            ifdata=[]
            for td in soup.find_all("td"):
                ifdata.append(td.text)

            #中身がない要素を削除
            ifdata=[i for i in ifdata if i!='']

            year=year_now
            month=month_now-1

            #データを詰め込んでいく
            for i in ifdata[::21]:
                monthdict["年"].append(year)
                monthdict["月"].append(month)
                monthdict["日"].append(i)

            #データ加工(最後だけdelete)
            while monthdict["日"][-1]!=(str(31) or str(30)):
                monthdict["年"].pop()
                monthdict["月"].pop()
                monthdict["日"].pop()

            #他も詰め込む　
            for i in ifdata[6::21]:
                monthdict["平均気温"].append(i)

            for i in ifdata[7::21]:
                monthdict["最高気温"].append(i)

            for i in ifdata[8::21]:
                monthdict["最低気温"].append(i)

            for i in ifdata[16::21]:
                monthdict["日照時間"].append(i)

            #確認
            while len(monthdict["平均気温"])!=len(monthdict["日"]) or len(monthdict["最高気温"])!=len(monthdict["日"]) or len(monthdict["最低気温"])!=len(monthdict["日"]) or len(monthdict["日照時間"])!=len(monthdict["日"]):
                for i in param:
                    monthdict[i].pop()
            #サーバー負荷軽減
            time.sleep(3)

    #その月のデータ
    load_url="https://www.data.jma.go.jp/obd/stats/etrn/view/daily_s1.php?prec_no={}&block_no={}&year={}&month={}&day=&view=".format(prec,block,str(year_now),str(month_now))
    #データロード
    html=requests.get(load_url)
    soup=BeautifulSoup(html.content,"html.parser")

    #要素をリスト化
    monthdata=[]
    for td in soup.find_all("td"):
        monthdata.append(td.text)

    #中身がない要素を削除
    monthdata=[i for i in monthdata if i!='']

    year=year_now
    month=month_now

    #データを詰め込んでいく
    for i in monthdata[::21]:
        monthdict["年"].append(year)
        monthdict["月"].append(month)
        monthdict["日"].append(i)

    #データ加工(最後だけdelete)
    while monthdict["日"][-1]!=str(date_now-1):
        monthdict["年"].pop()
        monthdict["月"].pop()
        monthdict["日"].pop()

    #他も詰め込む　
    for i in ifdata[6::21]:
        monthdict["平均気温"].append(i)

    for i in ifdata[7::21]:
        monthdict["最高気温"].append(i)

    for i in ifdata[8::21]:
        monthdict["最低気温"].append(i)

    for i in ifdata[16::21]:
        monthdict["日照時間"].append(i)

    #確認
    while len(monthdict["平均気温"])!=len(monthdict["日"]) or len(monthdict["最高気温"])!=len(monthdict["日"]) or len(monthdict["最低気温"])!=len(monthdict["日"]) or len(monthdict["日照時間"])!=len(monthdict["日"]):
        for i in param:
            monthdict[i].pop()
    #サーバー負荷軽減
    time.sleep(3)

    #データフレーム作成
    nowdf=pd.DataFrame(monthdict)
    #変な文字を消す
    for i in param:
        nowdf[i]=nowdf[i].str.replace(")","")
        nowdf[i]=nowdf[i].str.replace("]","")
        nowdf[i]=nowdf[i].str.replace(" ","")

    #数値化
    for i in param:
        nowdf[i]=pd.to_numeric(nowdf[i],errors="coerce")

    #30日のデータにする
    nowdf=nowdf.drop(nowdf.index[:-30])

    nowdf=nowdf.fillna(nowdf.median())
    return nowdf


def pred(toshi,nowdf,df):
    model=load_model("{}_model.h5".format(toshi))
    #入力を分ける
    input_data1=nowdf["平均気温"].values.astype(float)
    input_data2=nowdf["最高気温"].values.astype(float)
    input_data3=nowdf["最低気温"].values.astype(float)
    input_data4=nowdf["日照時間"].values.astype(float)
    
    #スケールの正規化
    df=pd.read_csv("df_{}.csv".format(toshi))
    norm_scale=df[["平均気温","最高気温","最低気温","日照時間"]].max().max()
    input_data1/=norm_scale
    input_data2/=norm_scale
    input_data3/=norm_scale
    input_data4/=norm_scale

    nowX1=np.array(input_data1)
    nowX2=np.array(input_data2)
    nowX3=np.array(input_data3)
    nowX4=np.array(input_data4)

    #データ成型
    nowX1=nowX1.reshape(1,window_size,1)
    nowX2=nowX2.reshape(1,window_size,1)
    nowX3=nowX3.reshape(1,window_size,1)
    nowX4=nowX4.reshape(1,window_size,1)
    
    #予測
    pred=model.predict({"input_a":nowX1,"input_b":nowX2,"input_c":nowX3,"input_d":nowX4})
    mid=pred[0]
    mid*=norm_scale
    top=pred[1]
    top*=norm_scale
    bot=pred[2]
    bot*=norm_scale
    return mid,top,bot
    


if __name__ =="__main__":
    main()