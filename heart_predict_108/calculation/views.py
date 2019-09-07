from django.shortcuts import render
from django.views import View
from django.http import JsonResponse, HttpResponse

import numpy as np # linear algebra
import pandas as pd
# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import json
import io
import csv
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

class all_views(View):
    def get(self, request):
        headers = {'content-type': ''}
        return JsonResponse(data={'hi':'happy'})


from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn.linear_model import LogisticRegression

class train_data():
    def __init__(self):
        pass

    def traindata(self, train_file):
        df = pd.read_csv(train_file)
        df['Age_Category'] = pd.cut(df['age'], bins=list(np.arange(25, 85, 5)))
        df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'Age_Category'])
        df.drop(['age'], axis=1, inplace=True)
        y = df['target']
        x = df.drop(['target'], axis=1)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
        self.lr = LogisticRegression()
        self.lr.fit(X_train, y_train)
        return self.lr

data = train_data()
train_set = data.traindata('predict/heart.csv')

@method_decorator(csrf_exempt, name='dispatch')
class picture_view(View):

    def get(self, request):

        f = matplotlib.figure.Figure()

        FigureCanvasAgg(f)
        A = io.BytesIO()
        df = pd.read_csv("D:\heart-disease-uci\heart.csv")
        response = HttpResponse('data unfetched')
        try:
            image_type = request.GET['image_name']
            print(image_type)
            if image_type.lower() == "heartdiagpostive":
                df['Age_Category'] = pd.cut(df['age'], bins=list(np.arange(25, 85, 5)))
                df[df['target'] == 1].groupby('Age_Category')['age'].count().plot(kind='bar')
                plt.title('Age Distribution of Patients with +ve Heart Diagonsis')
                A = io.BytesIO()
                plt.savefig(A, format='png')
                plt.close(f)
                response = HttpResponse(A.getvalue(), content_type='image/png')
            elif image_type.lower() == "heartdiagnegative":
                df['Age_Category'] = pd.cut(df['age'], bins=list(np.arange(25, 85, 5)))
                df[df['target'] == 0].groupby('Age_Category')['age'].count().plot(kind='bar')
                plt.title('Age Distribution of Patients with -ve Heart Diagonsis')
                B = io.BytesIO()
                plt.savefig(B, format='png')
                plt.close(f)
                response = HttpResponse(B.getvalue(), content_type='image/png')
            elif image_type.lower() == "blood":
                sns.countplot(x='fbs', data=df, hue='target')
                plt.xlabel('< 120mm/Hg                   >120 mm/Hg')
                plt.ylabel('Fasting blood sugar')
                plt.legend(['No disease', ' disease'])
                c = io.BytesIO()
                plt.savefig(c, format='png')
                plt.close(f)
                response = HttpResponse(c.getvalue(), content_type='image/png')
            elif image_type.lower() == "excercise":
                sns.countplot(x='exang', data=df, hue='target')
                plt.xlabel('No ex                                     Exercise')
                plt.title(' Excercise effect on Heart diease')
                plt.legend(['No disease', ' disease'])
                d = io.BytesIO()
                plt.savefig(d, format='png')
                plt.close(f)
                response = HttpResponse(d.getvalue(), content_type='image/png')
            elif image_type.lower() == "default":
                sns.countplot(x='target', data=df, hue='sex')
                plt.legend(['Female ', 'Male'])
                plt.xlabel('No Heart disease                 Heart Disease')
                E = io.BytesIO()
                plt.savefig(E, format='png')
                plt.close(f)
                response = HttpResponse(E.getvalue(), content_type='image/png')
        except KeyError:
            response = HttpResponse('unable fetch')

        return response

    def post(self, request):
        t = open('predict/download.csv', 'w')
        for filename, file in request.FILES.items():
            if filename == "test":
                k = open('predict/download.csv', 'w')
                file = file.readlines()
                for each_line in file:
                    k.write(each_line.decode())
                k.close()
            elif filename == "train":
                k = open('train/download.csv', 'w')
                file = file.readlines()
                for each_line in file:
                    k.write(each_line.decode())
                k.close()
        return HttpResponse("got the file thank you")


@method_decorator(csrf_exempt, name='dispatch')
class predict(View):
    def get(self, request):
        df = pd.read_csv("predict/heart.csv")
        df_clone = deepcopy(df)
        df['Age_Category'] = pd.cut(df['age'], bins=list(np.arange(25, 85, 5)))
        df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'Age_Category'])
        df.drop(['age'], axis=1, inplace=True)
        df.drop(['target'], axis=1, inplace=True)
        predicted = train_set.predict(df)
        df_clone['target'] = predicted
        df_clone.to_csv('predict/output.csv')
        labeled_data_file = df_clone.to_csv(index=False, header=True)

        response = HttpResponse(labeled_data_file, content_type='text/csv' )
        response['Content-Disposition'] = 'attachment;filename=export.csv'
        return response