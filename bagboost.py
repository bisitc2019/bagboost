import pandas as pd
fg=pd.read_csv('/Users/biswaranjantripathy/Desktop/Datascience/proj/credit.csv')

class bagboost():
    pass
    def labelencode(self):
        from sklearn.preprocessing import LabelEncoder
        test = LabelEncoder()
        fg['checking_balance'] = test.fit_transform(fg['checking_balance'])
        fg['savings_balance'] = test.fit_transform(fg['savings_balance'])
        fg['employment_duration'] = test.fit_transform(fg['employment_duration'])
        fg['credit_history'] = test.fit_transform(fg['credit_history'])
        fg['job'] = test.fit_transform(fg['job'])
        fg['phone'] = test.fit_transform(fg['phone'])
        fg['default'] = test.fit_transform(fg['default'])
        fg['purpose'] = test.fit_transform(fg['purpose'])
        fg['housing'] = test.fit_transform(fg['housing'])
        fg['other_credit']=test.fit_transform(fg['other_credit'])
        return fg
    def rel(self,sd):
        sd=self.labelencode()
        #df=self.sd.corr()
        #print(df['default'].sort_values(ascending=False))
        df=sd
        df.drop(['employment_duration', 'years_at_residence', 'phone', 'existing_loans_count', 'purpose','age', 'savings_balance', 'checking_balance', 'percent_of_income', 'housing'], axis=1, inplace=True)
        return df
    def classreport(self,y_test,pred):
        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
        print('classification report {}'.format(classification_report(pred,y_test)))
        print('confusion matrix {}'.format(confusion_matrix(pred,y_test)))
        print('accuracy score {}'.format(accuracy_score(pred,y_test)))
    def bagging(self,x_train,y_train,x_test,y_test):
        from sklearn.ensemble import BaggingClassifier
        test=BaggingClassifier()
        test.fit(x_train,y_train)
        pred=test.predict(x_test)
        print('--------BAGGING MECHANISM---------')
        self.classreport(y_test, pred)
    def adaboost(self,x_train,y_train,x_test,y_test):
        from sklearn.ensemble import AdaBoostClassifier
        test=AdaBoostClassifier()
        test.fit(x_train,y_train)
        pred=test.predict(x_test)
        print('--------AddaBoost-----------')
        self.classreport(y_test,pred)
    def gradientboost(self,x_train,y_train,x_test,y_test):
        from sklearn.ensemble import GradientBoostingClassifier
        test=GradientBoostingClassifier()
        test.fit(x_train, y_train)
        pred = test.predict(x_test)
        print('--------GradientBoost-----------')
        self.classreport(y_test, pred)
    def split(self):
        from sklearn.model_selection import train_test_split
        ou=self.rel('sd')
        x=ou.drop(['default'],axis=1)
        y=ou['default']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        from sklearn.ensemble import RandomForestClassifier
        test = RandomForestClassifier()
        test.fit(x_train, y_train)
        pred = test.predict(x_test)
        print('--------RANDOM FOREST------------')
        self.classreport(y_test, pred)
        self.bagging(x_train,y_train,x_test,y_test)
        self.adaboost(x_train,y_train,x_test,y_test)
        self.gradientboost(x_train,y_train,x_test,y_test)
if __name__=='__main__':
    oe=bagboost()
    oe.labelencode()
    oe.split()
