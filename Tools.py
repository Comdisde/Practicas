#encoding: utf8
from __future__ import division
import pandas as pd
import numpy as np
from re import escape, sub
from string import punctuation
from math import log


from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


import seaborn as sns
import matplotlib.pyplot as plt

from bokeh.io import output_notebook, show
from bokeh.plotting import figure



def frecuencia(df, columns=None, every=False, head=5): #Calcula la frecuencia de un dataframe
    if type(columns) == str and every == False:
        df[columns] = df[columns].replace(np.nan, "NA")
        b = pd.DataFrame(df[columns])
        b["Conteo"] = 1
        b = b.groupby(columns).sum()
        b.reset_index(inplace=True)
        b[columns] = b[columns].map(str)
        b["Frecuencia"] = b["Conteo"] / (b["Conteo"].sum())
        b.sort_values("Frecuencia", ascending=False, inplace=True)
        b.reset_index(inplace=True, drop=True)
        return b
    elif columns == None and every == True:
        for i in df.columns:
            print frecuencia(df, columns=i).head(head), "\n", "\n"
    
    
    elif type(columns) == list and every == False:
        for i in columns:
            print frecuencia(df, columns=i).head(head), "\n", "\n"
    
    
    else:
        return None


def metricas(model,Xt,Xv,yt,yv): #Mide efectividad de un Modelo Predictivo
    print "Roc Train: %.3f" %roc_auc_score(y_score=model.predict_proba(Xt)[:,1],y_true=yt),
    print "|| Roc Validate: %.3f" %roc_auc_score(y_score=model.predict_proba(Xv)[:,1],y_true=yv)
    print "Acc Train: %.3f" %accuracy_score(y_pred=model.predict(Xt),y_true=yt),
    print "|| Acc Validate: %.3f" %accuracy_score(y_pred=model.predict(Xv),y_true=yv)
    print "Matrix Conf Train: ","\n",confusion_matrix(y_pred=model.predict(Xt),y_true=yt)
    print "|| Matrix Conf Validate: ", "\n",confusion_matrix(y_pred=model.predict(Xv),y_true=yv)
    

def woe(df, var_dis, obj): #Despliega Dataframe's con el calculo del woe para la respectiva variable
    if type(var_dis)==str:
        aux=df[[var_dis, obj]]
        aux=aux.replace(np.nan, "Na")
        aux.reset_index(inplace=True)
        aux = aux.pivot_table(aggfunc='count',columns=obj,fill_value=0,
                          index=var_dis,values='index')
        aux["Total"]=aux.iloc[:,0]+aux.iloc[:,1]
        aux["Pm"]=aux.iloc[:,0]/aux.iloc[:,0].sum()
        aux["Pb"]=aux.iloc[:,1]/aux.iloc[:,1].sum()
        aux["Woe"]=(aux["Pb"]/aux["Pm"])
        aux["Woe"]=aux["Woe"].map(lambda x: 100*np.log(x))
        
        
        return aux
    
    elif type(var_dis)==list:
        for i in var_dis:
            print i, "\n", woe(df,i, obj),"\n","\n"
    elif var_dis==True:
        for i in {df.columns}-{obj}:
            print i, "\n", woe(df,i,obj),"\n","\n"

def del_outl(data, m=2): #Elimina outliers
    return pd.DataFrame(data[abs(data - np.mean(data)) < m * np.std(data)], columns=["Numero"])

def resta_fecha(d1,d2): #Resta  Fechas
    d1=datetime.timedelta(hours=d1.hour,minutes=d1.minute, seconds=d1.second)
    d2=datetime.timedelta(hours=d2.hour,minutes=d2.minute, seconds=d2.second)
    if d2>=d1:return int((d2-d1).seconds/60)
    elif d1>d2: return int((d1-d2).seconds/60)
            
            
def woe_c(df, obj, var, bins=5): 
    #Calcula el woe a Variables, a diferencia del anterior hace particiones a variables continuas
    if type(var)==list:
        for i in var:
            print i, "\n", woe_c(df, obj, i, bins),"\n","\n"
    else:
        aux=df[[var,obj]].copy()
        try:
           
            aux[var]=pd.cut(aux[var], bins=bins).astype(str)
        finally:
            return woe(aux, var, obj )
        
def iv(df, obj, var=None, bins=5, sm=False, every=False): #Calcula el iv de una variable
    if every==True: var=list(set(df.columns)-{obj})
    
    if type(var) == str and sm==False: #Despliega un data frame con el iv a un lado 
        a=woe_c(df,obj,var, bins=bins)
        a["IV"]=((a["Pb"]-a["Pm"])*a["Woe"]/100)
        return a
    elif type(var)==list and sm==False:
        for i in var:
            print iv(df, obj, i, bins), "\n", "\n"
    
    elif type(var)==list and sm==True:
        z=[]
        for k in var: #Da una lista donde viene el IV y el nombre de la variable
            z.append((k,iv(df, obj, k, bins)["IV"].sum() ))
        return z

def woe_l(df, obj,var ,bins=5):
    if type(var)==str: #Da un diccionario con el woe de todas las variables
        l={}
        u=woe_c(df, obj,var, bins)
        for j in range(len(u.index)):
            l[u.index[j]]=u["Woe"][j]
        return l
    elif type(var)==list:
        for i in var:
            v=woe_l(df, obj,i ,bins)
            print i.center(50, "=") 
            print pd.DataFrame(v, index=["Woe"]).T, "\n", "\n"

            
    
    

class WOE:
    def __init__(self,  data, target):
        #data must be a data frame 
        #assuming that target is a dataframe's column
        self.k=0
        self.data=data.copy()
        self.columns=list(data.columns)
        self.columns.remove(target)
        self.target=target
    
    def table_woe(self, column=None, bins=5):
        if column is None:
            column = self.columns
        
        return woe_c(bins=bins,df=self.data,obj=self.target,var=column)
    
    def iv(self, column=None, bins=5):
        if column is None:
            column = list(self.columns)
        
        return iv(df=self.data,obj=self.target,sm=True, bins=5, var=column)
    
    def iv_table(self, column=None, bins=5):
        if column is None:
            column = self.columns
        
        return iv(df=self.data,obj=self.target,sm=False, bins=5, var=column)
    
    def dict(self, column):
        return woe_l(df=self.data, var=column, obj=self.target)
    
    
    def fit(self, column=None):
        if column is None:
            column = self.columns
        
        for x in column:
            exec("self.dict_%s=self.dict(column='%s')" %(x.replace(" ","_"),x))
            exec("self.dict_inv_%s= {v:k for k,v in self.dict_%s.items()}" %(x.replace(" ","_"),x.replace(" ","_")))
          
        return None
    def fit_cont(self, column, bins=5):
        if self.k==0: self.resp_cont=self.data[column].copy()
        else: 
            
            self.data=pd.concat([self.data.drop(column, axis=1),self.resp_cont], axis=1)
        
        for x in column:
            self.data[x]=pd.cut(self.data[x], bins=bins).replace(
                np.nan,"nan").astype(str).replace(np.nan,"nan")
        self.k+=1
        print self.k
        
    def transform(self, df=None,column=None):
        if df is None: df = self.data.copy()
        if column is None: column = df.columns
        aux=df.copy()
        
        for x in column:
            exec("aux['%s']=aux['%s'].map(self.dict_%s)" %(x,x,x.replace(" ","_")) )
        return aux
    
    def inverse_transform(self, df=None, column=None):
        if df is None: df = self.data.copy()
        if column is None: column = aux.columns
        aux=df.copy()
        
        for x in column:
            exec("aux['%s']=aux['%s'].map(self.dict_inv_%s)" %(x,x,x.replace(" ","_")) )
        return aux
    
        
    
    
    
    

def sus_woe(df, obj, var_dist):#Sustituye los valores discretos por el del woe
    a=woe_l(df, obj, var_dist)
    for i in range(len(df[var_dist])):
        for j in a.keys():
            if df[var_dist][i]==j: df[var_dist][i]=a[j]

def sust_woe_2(df1, df2, obj_df1, var): 
    #Sustituye los valores discretos por el del woe, pero tabién a futuras observaciones
    b=woe(df=df1, obj=obj_df1, var_dis=var).reset_index()[[var,"Woe"]]
    c=df2.merge(b, how="inner", on=var).copy()
    
    del c[var]
    
    c.rename(columns={"Woe": var}, inplace=True)
    
   
    return c



class MDS_amv:
    def __init__(self, X0, n_comp, sample=False):
        if type(sample)==float: self.X0=X0.sample(frac=sample)
        else: self.X0=X0
        
        self.n_comp=n_comp
        
        mm = MinMaxScaler()
        mds= MDS(n_components=n_comp)
        
        mm.fit(X0)
        
        
        self.Xmm = pd.DataFrame(mm.transform(self.X0),columns=self.X0.columns)
        
        
        
        
        self.Xmds = pd.DataFrame(mds.fit_transform(self.Xmm),columns=["d%s" %i 
                                                                      for i in range(1, n_comp+1)])


        
    def plot_mds(self, dx,dy, sample=False, frac=0.1,cl=False):
        
        if sample ==True:self.Xplot = self.Xmds.sample(frac=frac)
        
        else:self.Xplot=self.Xmds
        
        
        if cl==False: return sns.lmplot(data=self.Xplot,x=dx,y=dy, fit_reg=False, )
        
        else:return sns.lmplot(data=self.Xplot,x=dx,y=dy, fit_reg=False, hue=cl )


class PCA_amv:
    def __init__(self, X0, n_comp):
        self.X0=X0
        self.n_comp=n_comp
        
        sc = StandardScaler()
        pca = PCA(n_components=n_comp)
        
        sc.fit(X0)
        self.Xs = pd.DataFrame(sc.transform(self.X0),columns=self.X0.columns)
        pca.fit(self.Xs)
        
        print pca.explained_variance_ratio_.cumsum()
        
        self.Xp = pd.DataFrame(pca.transform(self.Xs),columns=['p%s' %i for i in range(1,n_comp+1)])

        
    def plot_pca(self, px,py, sample=False, frac=0.1,cl=False):
        
        if sample ==True:
            self.Xplot = self.Xp.sample(frac=frac)
        
        if cl==False:
            return sns.lmplot(data=self.Xp,x=px,y=py, fit_reg=False, )
        else:
            return sns.lmplot(data=self.Xp,x=px,y=py, fit_reg=False, hue=cl )
        

        
class TrateNan:
    def __init__(self, df):
        self.df=df
        
    def count_nan(self, col=True):
        df_aux=pd.DataFrame(np.zeros(shape=(1, 3)), columns=["Columna", "Num_NaN","Cardinalidad"])
        
        if col==True: col=self.df.columns
        
        
        for j in col:
            ser=self.df[j]

            df_aux=pd.concat([df_aux, pd.DataFrame({ "Columna": [j],"Num_NaN" : 
                        [abs(int(ser.value_counts().sum()-len(ser)))], 
                        "Cardinalidad":[len(ser.value_counts().index)]})])
        df_aux=df_aux.reset_index(drop=True).iloc[1:][["Columna","Cardinalidad","Num_NaN"]].copy()
        return df_aux
    
    
    def select_var(self, tol_car=1, tol_nan=.20):
        data=self.count_nan()
        a=data.iloc[np.where(data.Cardinalidad>tol_car)]
        return list(a.Columna.iloc[np.where(a.Num_NaN<self.df.shape[0]*tol_nan)])
    def df_to_var(self, tol_car=1, tol_nan=.20):
        return self.df[self.select_var(tol_car, tol_nan)]


def plot_predict(y_real,y_pred,title="", color_real=(100,191,163), color_pred=(249, 139, 96), circle=True):
    p = figure(plot_width=980, plot_height=400, title=title)

    p.title.text_font_size = '20pt'
    p.title.align="center"

    #Real
    p.line(y=y_real,x=range(len(y_real)),  line_width=3, color=color_real, legend="Real")
    if circle==True:
        p.circle(y=y_real,x=range(len(y_real)), alpha=.6, size=9, color=color_real)

    #predict
    p.line(y=y_pred,x=range(len(y_pred)),  line_width=2, color=color_pred, legend=u"Predicción")
    if circle==True:
        p.circle(y=y_pred,x=range(len(y_pred)), alpha=.4, size=9, color=color_pred)




    show(p)

def replace_punctuation ( text ):
    return sub('[%s]' % escape(punctuation), '', text)
        
def mean_st(dfr,target, n):
    aux=dfr.copy()
    for k in range(len(n)):
        mean_ob=pd.Series([np.nan for x in range(n[k])])
        desv_ob=pd.Series([np.nan for x in range(n[k])])
        for j in range(aux.shape[0]-n[k]):
            mean_ob=mean_ob.append(pd.Series(aux[target].iloc[j:n[k]+j].mean()))
            desv_ob=desv_ob.append(pd.Series(aux[target].iloc[j:n[k]+j].std()))


        mean_ob.reset_index(drop=True, inplace=True)
        desv_ob.reset_index(drop=True, inplace=True)

        aux["desv_ob "+str(n[k])]=desv_ob
        aux["mean_ob "+str(n[k])]=mean_ob
    return aux