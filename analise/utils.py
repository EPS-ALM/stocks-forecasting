import re
import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt

def corrige_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Padroniza os nomes das colunas
    df.columns = [re.sub('[^A-Za-z0-9]+', '', unidecode(col.lower())) for col in df.columns]
    
    # Faz o casting da coluna para o dtype de datetime do pandas
    df['data'] = pd.to_datetime(df['data'],format="%d.%m.%Y")
    
    # Corrige os valores de ponto flutuante que estavam definidos com virgula
    df['ultimo'] = df['ultimo'].apply(lambda x: str(x).replace('.','').replace(',','.')).astype(float)
    df['abertura'] = df['abertura'].apply(lambda x: str(x).replace('.','').replace(',','.')).astype(float)
    df['maxima'] = df['maxima'].apply(lambda x: str(x).replace('.','').replace(',','.')).astype(float)
    df['minima'] = df['minima'].apply(lambda x: str(x).replace('.','').replace(',','.')).astype(float)
    
    # Corrige a coluna de porcentagens
    df['var'] = df['var'].apply(lambda x: str(x).replace('.','').replace(',','.').replace('%','')).astype(float)
    
    df = df.rename(columns={'ultimo':'fechamento'})
    
    df.sort_values('data', inplace=True)
    
    return df.loc[:,['data', 'fechamento', 'abertura', 'maxima', 'minima', 'var']]

def agrega_mes(df:pd.DataFrame) -> pd.DataFrame:
    # Cria uma coluna identificado o ano e o mes para facilitar a agregação ao invés de especificar a data
    df['ano_mes'] = df['data'].apply(lambda x: x.strftime('%Y-%m'))
    df['media'] = df.loc[:, 'fechamento']
    
    # Agrega pelo ano e o mês, faz a média das cotações e faz a somatória das variações no mês
    df_agg = df.groupby(['ano_mes'], as_index=False).\
                agg({'fechamento':'first',
                    'abertura':'last',
                    'maxima':'max',
                    'minima':'min',
                    'media':'mean',
                    'var':'sum',})
                
    return df_agg

def agrega_ano(df:pd.DataFrame) -> pd.DataFrame:
    # Cria uma coluna identificado o ano para facilitar a agregação ao invés de especificar a data
    df['ano'] = df['data'].apply(lambda x: x.strftime('%Y'))
    df['media'] = df.loc[:, 'fechamento']
    
    # Agrega pelo ano, faz a média das cotações e faz a somatória das variações no ano
    df_agg = df.groupby(['ano'], as_index=False).\
                agg({'fechamento':'first',
                    'abertura':'last',
                    'maxima':'max',
                    'minima':'min',
                    'media':'mean',
                    'var':'sum',})
                
    return df_agg

def normaliza_dataframe(df):
    df_normalizado = df.copy()
    df_normalizado['valor'] = (df['fechamento'] - df['fechamento'].min()) / (df['fechamento'].max() - df['fechamento'].min())
    return df_normalizado