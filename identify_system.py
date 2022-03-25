# -*- coding: utf-8 -*-
"""
Created on Mon Jul 05 23:04:10 2021

@author: Marcelo Feliciano Filho

Técnicas de Programação e Introdução a Identificação de Sistemas

Esse código tem por objetivo realizar a identificação de um sistema utilizando
o método Sindy com a biblioteca Pysindy:
    Para instalar as bibliotecas necessárias rode no terminal do python com a virtualenv ativada:
    pip install pysindy, matplotlib, sklearn, pandas, numpy

https://towardsdatascience.com/sysidentpy-a-python-package-for-modeling-nonlinear-dynamical-data-f21fa3445a3c
"""
from pandas import DataFrame, read_csv
from os import getcwd, listdir, path as ospath, mkdir
from pysindy import SINDy, STLSQ, PolynomialLibrary
from numpy import array, arange, power, sum as soma, delete
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt


class IdentifyF16BenchMark:
    """
    Classe que identifica o benchmark F16 pelo método Sindy
    """

    def __init__(self, path=ospath.join(getcwd(), 'BenchmarkData')):
        """
        Inicia a classe para identificação do benchmark não linear
        Sparcefy Dynamics - (redução de termos para visualizar o modelo)
        Parametros
        ----------
        path: str - Opicional
            Diretório da pasta com os datasets em CSV, se não for informado, pega
            do diretório raiz
        """
        self.root_path = path
        self.header = ["Force", "Voltage", "Acceleration1", "Acceleration2", "Acceleration3"]
        self.dict_in = {'FullMSine': [4.8, 19.2, 28.8, 57.6, 67.0, 86.0, 95.6],
                        'SineSw': [12.4, 24.6, 36.8, 61.4, 73.6, 85.7, 97.8],
                        'SpecialOddMSine': [12.2, 49.0, 97.1]}
        self.tipos = self.dict_in.keys()
        self.file_dict = {tipo: [ospath.join(path, dt) for dt in listdir(path) if
                                 '.csv' in dt and tipo in dt] for tipo in self.tipos}
        self.df_dict = {tipo: self.csvs_to_df(arqs) for tipo, arqs in self.file_dict.items()}
        self.graph_dir = ospath.join(self.root_path, datetime.strftime(datetime.now(), '%d_%m_%Y__%Hh%M'))
        mkdir(self.graph_dir)

    def csvs_to_df(self, files):
        """
        Essa função converte um arquivo para dataframe, 
    
        Parameters
        ----------
        files : list
            lista contendo todos os diretórios dos arquivos.
    
        Returns
        -------
        Dataframe completo contendo toda a série de dados similares.
    
        """
        df = DataFrame()
        for file in files:
            data = read_csv(file, sep=',')[self.header]
            df = df.append(DataFrame(data), ignore_index=True)
    
        return df

    def sindy_identify(self, sys_df, tipo_sis):
        """
        Esse método realiza a identificação do sistema para cada um dos
        cenários especificados e chama a função para plotar e salvar os
        gráficos gerados
        
        Parameters
        ----------
        sys_df : pd.DataFrame
            Dataframe com os valores para identificação

        Returns
        -------
        Modelo do sistema identificado.
        
        """
        dt = 0.0025
        t_train = arange(0, len(sys_df)*dt, dt)
        x_train = array(sys_df)
        poly_order = 3
        
        model = SINDy(
            optimizer=STLSQ(threshold=dt),
            feature_library=PolynomialLibrary(degree=poly_order),
        )
        model.fit(x_train, t=dt)
        model.print()
        
        dict_plot = {2: f'{tipo_sis} - Acceleration Eixo X', 3: f'{tipo_sis} - Acceleration Eixo Y', 
                     4: f'{tipo_sis} - Acceleration Eixo Z'}
        predito = model.predict(x_train)/100
            
        for index in dict_plot.keys():
            self.plot(x_train, t_train, dict_plot[index], index, predito)

        return {'model': model, 'metrics': self.math_err(x_train, predito)}

    def plot(self, x_train, t_train, title, index, simulate=array([])):
        """
        Função que imprime os plots 

        Parameters
        ----------
        x_train : numpy.array
            Array de treinamento com todos os dados.
        t_train : numpy.array
            Array com o tempo em segundos.
        title : str
            Título do gráfico a ser plotado.
        index : int
            Índice do loop (1, 2 ou 3).
        simulate : numpy.array, optional
            Variável que indica a simulação e se essa será plotada com o gráfico real. 
            Por podrão o valor é um array vazio.
        Returns
        -------
        None. Mas plota o gráfico necessário.
        """
        fig = plt.figure(figsize=(15, 4))
        ax = fig.add_subplot(131, projection="3d")
        # Plot dos dados obtidos para treino
        ax.plot(x_train[: t_train.size, 0], x_train[: t_train.size, 1],
                x_train[: t_train.size, index], label='Real')
        if simulate.size:  # Se não for falso, 
            ax.plot(x_train[: t_train.size, 0], x_train[: t_train.size, 1],
                    simulate[: t_train.size, index], label='Predicted')
        plt.title(title)
        plt.legend(labelcolor='linecolor')
        ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
        
        # Salva o gráfico
        #plt.savefig(ospath.join(self.graph_dir, f'{title}.png'))

    def math_err(self, real_val, predito):
        """
        Função que calcula o erro pelo método L2

        Parameters
        ----------
        real_val : list
            Lista contendo todos os valores reais medidos no benchmark.
        predito : list
            Lista com os valores preditos pelo sindy.

        Returns: dict
        -------
        Dicionário com os valores das SCORES das amostras apresentadas e preditas pelo modelo.

        """
        actual_value = delete(delete(real_val, 0, 1), 0, 1)
        predicted_value = delete(delete(predito, 0, 1), 0, 1)  # Normaliza

        MAPE = metrics.mean_absolute_percentage_error(actual_value, predicted_value)
        MAE = metrics.mean_absolute_error(actual_value, predicted_value)
        MSE = metrics.mean_squared_error(actual_value, predicted_value)
        RMSE = MSE**.5
        R2 = metrics.r2_score(actual_value, predicted_value)
        L2 = soma(power((actual_value-predicted_value), 2))
        
        return {'MAPE': MAPE, 'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE,'R2': R2, 'L2': L2}

    def simulate_model(self, type_sys):
        """
        Essa função chama o sistema que deseja ser identificado, dentre as
        opções estão: 
            'FullMSine',
            'SineSw',
            'SpecialOddMSine'
        Parameters
        ----------
        type_sys : STR
            Nome do tipo de sistema.

        Returns
        -------
        Dicionário com o tipo de 

        """
        sys = self.df_dict[type_sys][1:].astype(float).values
        model = self.sindy_identify(sys, type_sys)
        
        return {type_sys: model}


if __name__ == '__main__':  # Executa a classe e retorna os dados automaticamente
    tic = datetime.now()
    benchmark = IdentifyF16BenchMark()
    dict_solution = {key: [] for key in benchmark.tipos}
    for tipo in benchmark.tipos:  # Emula a classe para todos os tipos
        dict_solution.update(benchmark.simulate_model(tipo))
    
    # Imprime no console os resultados
    bar = '-------------------------------'
    print(f'\n{bar}\n\nTotal Processing Time: {datetime.now() - tic} \n\n--Result--\n\n{dict_solution}')
