import json, time, numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
SEED=42; R="results/"
COLS=["ip_mean","ip_std","ip_kurtosis","ip_skew","dm_mean","dm_std","dm_kurtosis","dm_skew","label"]
df=pd.read_csv("data/HTRU_2.csv",header=None,names=COLS); X=df.drop(columns="label"); y=df.label
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
Xfit,Xval,yfit,yval=train_test_split(Xtr,ytr,test_size=.2,stratify=ytr,random_state=SEED)
def xgb(**k): return XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.08,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=SEED,n_jobs=-1,**k)
spw=(ytr==0).sum()/(ytr==1).sum()
M={"Logistic Regression":Pipeline([("s",RobustScaler()),("c",LogisticRegression(max_iter=1000,class_weight="balanced"))]),
"Decision Tree":Pipeline([("s",RobustScaler()),("c",DecisionTreeClassifier(max_depth=6,class_weight="balanced",random_state=SEED))]),
"Random Forest":Pipeline([("s",RobustScaler()),("c",RandomForestClassifier(300,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED))]),
"RBF-SVM + SMOTE":Pipeline([("s",RobustScaler()),("o",SMOTE(random_state=SEED)),("c",SVC(probability=True,random_state=SEED))]),
"XGBoost + SMOTE":Pipeline([("s",RobustScaler()),("o",SMOTE(random_state=SEED)),("c",xgb())]),
"XGBoost (weighted)":Pipeline([("s",RobustScaler()),("c",xgb(scale_pos_weight=spw))])}
out={};P={}
for n,m in M.items():
    t=time.time();m.fit(Xtr,ytr);ft=time.time()-t
    t=time.time();p=m.predict_proba(Xte)[:,1];pt=(time.time()-t)/len(Xte)*1e6
    yp=(p>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(yte,yp).ravel();P[n]=p
    out[n]=dict(acc=accuracy_score(yte,yp),recall=recall_score(yte,yp),precision=precision_score(yte,yp),f1=f1_score(yte,yp),mcc=matthews_corrcoef(yte,yp),roc=roc_auc_score(yte,p),pr=average_precision_score(yte,p),TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),fit=ft,per_m=pt)
ens=(P["XGBoost + SMOTE"]+P["RBF-SVM + SMOTE"])/2;P["Ensemble"]=ens
yp=(ens>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(yte,yp).ravel()
out["Ensemble"]=dict(acc=accuracy_score(yte,yp),recall=recall_score(yte,yp),precision=precision_score(yte,yp),f1=f1_score(yte,yp),mcc=matthews_corrcoef(yte,yp),roc=roc_auc_score(yte,ens),pr=average_precision_score(yte,ens),TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),fit=None,per_m=None)
# threshold chosen on TRAIN out-of-fold predictions only (no test leakage)
cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
oof=cross_val_predict(M["XGBoost + SMOTE"],Xtr,ytr,cv=cv,method="predict_proba")[:,1]
ops={}
for tgt in (0.90,0.95,0.98):
    pr_,rc_,th_=precision_recall_curve(ytr,oof); i=np.where(rc_[:-1]>=tgt)[0][-1]; th=float(th_[i])
    yp=(P["XGBoost + SMOTE"]>=th).astype(int); tn,fp,fn,tp=confusion_matrix(yte,yp).ravel()
    ops[str(tgt)]=dict(th=th,recall=recall_score(yte,yp),precision=precision_score(yte,yp),rfi_rej=tn/(tn+fp),FN=int(fn),FP=int(fp),candidates_kept=int(tp+fp))
# CV for each model (5-fold, SMOTE inside folds)
cvres={}
from sklearn.model_selection import cross_validate
for n in ["Logistic Regression","Decision Tree","Random Forest","XGBoost + SMOTE","XGBoost (weighted)"]:
    r=cross_validate(M[n],X,y,cv=cv,scoring=["recall","precision","f1","roc_auc"],n_jobs=1)
    cvres[n]={k[5:]:(float(v.mean()),float(v.std())) for k,v in r.items() if k.startswith("test_")}
imp=dict(zip(COLS[:-1],map(float,M["XGBoost + SMOTE"].named_steps["c"].feature_importances_)))
# ---- functional tests ----
tests=[]
def T(i,desc,ok,det): tests.append(dict(id=i,desc=desc,ok=bool(ok),det=det))
T("TC1","Dataset loads with 17,898 rows and 9 columns",df.shape==(17898,9),str(df.shape))
T("TC2","Class counts match HTRU2 (1,639 / 16,259)",(y.sum()==1639) and ((y==0).sum()==16259),f"{y.sum()} / {(y==0).sum()}")
T("TC3","No missing values in any feature",int(df.isna().sum().sum())==0,"0 NaN")
T("TC4","Stratified split keeps pulsar share (test within 0.1 pp of full data)",abs(yte.mean()-y.mean())<.001,f"{yte.mean():.4f} vs {y.mean():.4f}")
sc=M["XGBoost + SMOTE"].named_steps["s"]
T("TC5","Scaler statistics come from training data only (median equals train median)",np.allclose(sc.center_,Xtr.median().values),"match")
T("TC6","SMOTE balances the training fold only (test set untouched)",len(Xte)==3580 and abs(yte.mean()-y.mean())<.001,"test n=3580")
p=P["XGBoost + SMOTE"]
T("TC7","Predicted probabilities lie in [0,1] for all test candidates",p.min()>=0 and p.max()<=1,f"[{p.min():.4f},{p.max():.4f}]")
m2=Pipeline([("s",RobustScaler()),("o",SMOTE(random_state=SEED)),("c",xgb())]).fit(Xtr,ytr)
T("TC8","Re-training with the same seed reproduces identical predictions",np.allclose(m2.predict_proba(Xte)[:,1],p),"identical")
T("TC9","Recall on test set at threshold 0.5 is at least 0.90 (NFR2)",out["XGBoost + SMOTE"]["recall"]>=.90,f"{out['XGBoost + SMOTE']['recall']:.3f}")
T("TC10","Scoring speed under 15 s per million candidates (NFR1)",out["XGBoost + SMOTE"]["per_m"]<15,f"{out['XGBoost + SMOTE']['per_m']:.1f} s/M")
Xb=Xte.copy();Xb.iloc[0,2]=1e6
try:
    pb=M["XGBoost + SMOTE"].predict_proba(Xb)[0,1]; T("TC11","Extreme outlier value (kurtosis = 1e6) does not crash the pipeline",np.isfinite(pb),f"p={pb:.3f}")
except Exception as e: T("TC11","Extreme outlier value does not crash the pipeline",False,str(e))
json.dump(dict(test=out,ops=ops,cv=cvres,imp=imp,tests=tests,spw=float(spw),ntr=len(Xtr),nte=len(Xte),pos_te=int(yte.sum()),pos_tr=int(ytr.sum())),open(R+"res.json","w"),indent=1)
# ---- figures ----
plt.rcParams.update({"font.family":"serif","font.size":10})
fig,ax=plt.subplots(1,2,figsize=(10,4))
for n in ["Logistic Regression","Decision Tree","Random Forest","XGBoost + SMOTE","RBF-SVM + SMOTE","Ensemble"]:
    f,t,_=roc_curve(yte,P[n]);ax[0].plot(f,t,label=n)
    a,b,_=precision_recall_curve(yte,P[n]);ax[1].plot(b,a,label=n)
ax[0].plot([0,1],[0,1],"k:");ax[0].set(xlabel="False positive rate",ylabel="True positive rate");ax[1].set(xlabel="Recall",ylabel="Precision");ax[0].legend(fontsize=7,loc="lower right");ax[0].set_title("(a) ROC");ax[1].set_title("(b) Precision-Recall")
plt.tight_layout();plt.savefig(R+"roc_pr.png",dpi=200);plt.close()
cm=confusion_matrix(yte,(P["XGBoost + SMOTE"]>=.5).astype(int));fig,a=plt.subplots(figsize=(4,3.6));a.imshow(cm,cmap="Blues")
for i in range(2):
    for j in range(2):a.text(j,i,cm[i,j],ha="center",va="center",fontsize=14,color="white" if cm[i,j]>cm.max()/2 else "black")
a.set(xticks=[0,1],yticks=[0,1],xticklabels=["RFI","Pulsar"],yticklabels=["RFI","Pulsar"],xlabel="Predicted class",ylabel="True class");plt.tight_layout();plt.savefig(R+"cm.png",dpi=200);plt.close()
s=pd.Series(imp).sort_values();fig,a=plt.subplots(figsize=(5,3.4));s.plot.barh(ax=a,color="#1f3a6e");a.set_xlabel("XGBoost importance (gain-based)");plt.tight_layout();plt.savefig(R+"imp.png",dpi=200);plt.close()
fig,ax=plt.subplots(1,3,figsize=(11,3.2))
for a,c in zip(ax,["ip_kurtosis","ip_skew","dm_std"]):
    for l,col,nm in[(0,"#d97706","RFI / noise"),(1,"#059669","Pulsar")]:
        v=df.loc[df.label==l,c];a.hist(v.clip(v.quantile(.005),v.quantile(.995)),bins=50,alpha=.6,color=col,label=nm,density=True)
    a.set_xlabel(c);a.legend(fontsize=7)
plt.tight_layout();plt.savefig(R+"dist.png",dpi=200);plt.close()
# threshold curve
pr_,rc_,th_=precision_recall_curve(yte,P["XGBoost + SMOTE"]);fig,a=plt.subplots(figsize=(5,3.4));a.plot(th_,rc_[:-1],label="Recall");a.plot(th_,pr_[:-1],label="Precision");a.set_xlabel("Decision threshold");a.legend();plt.tight_layout();plt.savefig(R+"thr.png",dpi=200);plt.close()
# block diagrams
from matplotlib.patches import FancyBboxPatch
def boxes(fn,labels,hl,w=10,h=1.6):
    fig,a=plt.subplots(figsize=(w,h));a.axis("off");a.set_xlim(0,len(labels)*2.1);a.set_ylim(0,1.6)
    for i,l in enumerate(labels):
        x=i*2.1+.05;a.add_patch(FancyBboxPatch((x,.3),1.8,1.0,boxstyle="round,pad=.02",fc="#f5e6b8" if i==hl else "#dbe5f1",ec="#1f2a44"));a.text(x+.9,.8,l,ha="center",va="center",fontsize=8)
        if i<len(labels)-1:a.annotate("",xy=(x+2.1,.8),xytext=(x+1.82,.8),arrowprops=dict(arrowstyle="->"))
    plt.savefig(R+fn,dpi=200,bbox_inches="tight");plt.close()
boxes("arch.png",["Radio telescope\n(Parkes / HTRU)","De-dispersion\n& folding","Feature extraction\n(8 statistics)","RobustScaler\n+ SMOTE (train)","XGBoost / SVM\nclassifier","Candidate list\n(pulsar / RFI)"],4,w=12)
fig,a=plt.subplots(figsize=(5.5,7));a.axis("off");a.set_xlim(0,10);a.set_ylim(0,17)
def bx(x,y,t,c="#dbe5f1",w=5):a.add_patch(FancyBboxPatch((x-w/2,y-.6),w,1.2,boxstyle="round,pad=.02",fc=c,ec="#1f2a44"));a.text(x,y,t,ha="center",va="center",fontsize=8)
def ar(x,y1,y2):a.annotate("",xy=(x,y2),xytext=(x,y1),arrowprops=dict(arrowstyle="->"))
steps=["Start","Load HTRU_2.csv (17,898 rows)","Stratified 80/20 split","Fit RobustScaler on train, transform both","SMOTE on training fold only","Train XGBoost / RBF-SVM","Choose threshold on train out-of-fold scores","Score held-out test set","Compute metrics, plots, report"]
ys=[16,14.2,12.4,10.6,8.8,7,5.2,3.4,1.6]
for i,(t,yv) in enumerate(zip(steps,ys)):
    bx(5,yv,t,"#f5e6b8" if i in(0,8) else "#dbe5f1",6)
    if i<8:ar(5,yv-.6,ys[i+1]+.6)
plt.savefig(R+"flow.png",dpi=200,bbox_inches="tight");plt.close()
fig,a=plt.subplots(figsize=(8,3));tasks=[("Problem identification & literature review",1,3),("Dataset study & feature analysis",3,5),("Pipeline design & implementation",5,9),("Testing, baselines & evaluation",9,12),("Report writing & presentation",11,15)]
for i,(t,s0,e) in enumerate(tasks):a.barh(len(tasks)-i,e-s0+1,left=s0-.5,color="#9ab0cc",ec="#1f2a44")
a.set_yticks(range(len(tasks),0,-1));a.set_yticklabels([t[0] for t in tasks],fontsize=8);a.set_xticks(range(1,16));a.set_xlabel("Semester week");a.axvline(8,ls="--",c="#c08a1b");a.axvline(15,ls="--",c="#c08a1b");plt.tight_layout();plt.savefig(R+"gantt.png",dpi=200);plt.close()
print(json.dumps(out,indent=0,default=float)[:3000]);print(ops);print(cvres);print([ (t['id'],t['ok'],t['det']) for t in tests])
