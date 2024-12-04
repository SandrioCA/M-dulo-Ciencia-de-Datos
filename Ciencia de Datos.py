

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:49:06 2024

@author: Simon
"""
import numpy as np
import math

#Promedio aritmético
def promedio(array):
    """
    Calcua el promedio de una lista de números
    -----------
    arrray : array
        lista con números
    
    retorna
    ------------
    promedio : float
        proedio aritmético de los numeros
    """
    lenArray = len(array)
    totalArray = sum(array)
    mean = totalArray/lenArray
    
    return mean

def mediana(lista):
    """

    Parameters
    ----------
    lista : list
        Toma una lista de valores

    Devuelve la mediana de los valores
    -------
    float/int
    """
    listaNan = []
    for i in listaNan:
        if math.isfinite(i):
            listaNan.append(i)
    
    listaNan = listaNan.sort()
    largoLista = len(listaNan)
    valorMedio = len(listaNan)//2
    
    if largoLista%2 != 0:           
        return lista[valorMedio]
    
    else:
        mediana = (listaNan[valorMedio]+listaNan[valorMedio-1])/2  
        return mediana


def moda(lista):
    """
    Parameters
    ----------
    lista : list
        Una lista donde se repiten los datos

    Returns
    -------
        el valor que más se repite
    """
    
    listaDatos = []
    listaFrecuencia = []
    
    for i in lista:
        if i not in listaDatos:
            listaDatos.append(i)
            
    for j in listaDatos:
        cont = 0
        for k in lista:
            if j == k:
                cont += 1
        listaFrecuencia.append(cont)
    
    iMax = 0
    valMax = listaFrecuencia[0]
    
    for i in range(1, len(listaFrecuencia)):
        if listaFrecuencia[i] > valMax:
            iMax = i
            valMax = listaFrecuencia[i]
    
    return listaDatos[iMax]  


def rango(datos):
    listaNan = []
    for i in datos:
        if math.isfinite(i):
            listaNan.append(i)
    
    listaNan = np.array(listaNan)
    
    maximo = max(listaNan)
    minimo = min(listaNan)
    
    rango = maximo - minimo
    return rango
    


def varianza(datos):
    listaNan = []
    for i in datos:
        if math.isfinite(i):
            listaNan.append(i)
    
    listaNan = np.array(listaNan)
    
    porm = promedio(listaNan)
    largo = len(listaNan)
    
    varianza = (sum((listaNan - porm)**2))/largo
    
    return varianza


def desvEstandar(datos):
    listaNan = []
    for i in datos:
        if math.isfinite(i):
            listaNan.append(i)
            
    listaNan = np.array(listaNan)
    var = varianza(listaNan)
    
    std = (var)**0.5
    
    return std

def MAD(datos):
    listaNan = []
    for i in datos:
        if math.isfinite(i):
            listaNan.append(i)
            
    listaNan = np.array(listaNan)
    
    med = mediana(listaNan)
    dsvAbs = mediana(listaNan-med)
    
    return dsvAbs


def covarianza(x,y):
    listaX = []
    listaY = []
    for i in x:
        if math.isfinite(i):
            listaX.append(i)
    
    for i in y:
        if math.isfinite(i):
            listaY.append(i)
    
    X = np.array(listaX)
    Y = np.array(listaY)
    
    promX= promedio(X)
    promY = promedio(Y)
    
    
    denominador = (sum((x-promX)**2))*(sum((y-promY**2)))
    
    covar = denominador/len(x)

    return covar


def coeficiente_correlacion(x, y):
    """
    Calcula el coeficiente de correlación de Pearson entre dos variables x e y.
    
    Parámetros:
    x (list or np.array): La primera variable de datos.
    y (list or np.array): La segunda variable de datos.
    
    Retorna:
    float: El coeficiente de correlación de Pearson entre x e y.
    """

    listaX = []
    listaY = []
    for i in x:
        if math.isfinite(i):
            listaX.append(i)
    
    for i in y:
        if math.isfinite(i):
            listaY.append(i)
    
    x = np.array(listaX)
    y = np.array(listaY)
    
    
    # Calcular la covarianza entre x y y
    covar = covarianza(x,y)
    
    # Calcular las desviaciones estándar de x y y
    desviacionX = desvEstandar(x)
    desviacionY = desvEstandar(y)
    
    # Calcular el coeficiente de correlación de Pearson
    r = covar / (desviacionX * desviacionY)
    
    return r
    

def calcular_cuartiles(datos):
    """
    Calcula los cuartiles de un conjunto de datos.
    
    Parámetros:
    datos (list or np.array): Un arreglo o lista de datos numéricos.
    
    Retorna:
    tuple: Un tuple con los cuartiles (Q0, Q1, Q2, Q3, Q4)
    """
    datos = np.array(datos)
    
    # Calcular los cuartiles
    Q0 = np.min(datos)  # Valor mínimo (0% cuartil)
    Q1 = np.percentile(datos, 25)  # Primer cuartil (25% cuartil)
    Q2 = np.percentile(datos, 50)  # Mediana (50% cuartil)
    Q3 = np.percentile(datos, 75)  # Tercer cuartil (75% cuartil)
    Q4 = np.max(datos)  # Valor máximo (100% cuartil)
    
    return Q0, Q1, Q2, Q3, Q4



# Método de gradient decent

def MSE(x,y,theta):
    m,b = theta
    
    residuos = [(y_i - (m*x_i+b) )**2 for x_i, y_i in zip(x,y)]
    mse = sum(residuos)/len(residuos)
    
    return mse


    
def limiteDeCuociente(x,y,f,v,i,h=0.0001):
    #agregar el valor h solo al valor i-esimo elemento de v
    w = [v_j + (h if j==i else 0) for j,v_j in enumerate(v)]
    
    return (f(x,y,w)-f(x,y,v)) / h

def estimarGradiente(x,y,f,v,i,h=0.0001):
    return [limiteDeCuociente(x, y, f, v, i,h) for i in range(len(v))]



def pasoEnGradiente(v,gradient,step_size):
    step = [step_size*g_i for g_i in gradient]
    
    return [a + b for a,b in zip(v, step)]
    
    

def gradienteMse(x,y,theta):
    pendiente, intercepto = theta
    
    y_pred = [pendiente *xv + intercepto for xv in x]
    
    g1 = 2/len(x) * sum([ (y_p - y_d) for x_d,y_d ])
    
    g2 = 2/len(x) *sum([(y_p - y_d) for x_d, y_d, y_p in zip(x,y,y_pred)])
    
 
    return [g1,g2]
    


