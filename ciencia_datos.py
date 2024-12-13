

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
    listaNan = []
    
    for i in array:
        if math.isfinite(i):
            listaNan.append(i)
    
    lenArray = len(listaNan)
    totalArray = sum(listaNan)
    mean = totalArray/lenArray
    
    return mean

def mediana(lista):
    """
    Calcula el valor medio de un set de datos
    
    Parameters
    ----------
    lista : list
        Toma una lista de valores.

    Devuelve la mediana de los valores
    -------
    float/int
    """
    listaNan = []
    
    for i in lista:
        if math.isfinite(i):
            listaNan.append(i)
    
    listaNan.sort()
    
    largoLista = len(listaNan)
    valorMedio = largoLista // 2
    
    if largoLista % 2 != 0:
        return listaNan[valorMedio]
    
    else:
        mediana = (listaNan[valorMedio] + listaNan[valorMedio - 1]) / 2
        return mediana


def moda(lista):
    
    """
    Cañcula en valor que más se repite
    
    
    Parameters Array/lista de numeros
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
    """
    Calcula el rango de un conjunto de datos, que es la diferencia 
    entre el valor máximo y el valor mínimo
    
    Parámetros:
    datos (iterable): Un conjunto de datos numéricos, puede 
    ser una lista o un arreglo de numpy.
    
    Retorna:
    float: El rango de los datos, calculado como la diferencia 
    entre el valor máximo y el mínimo.
    """
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
    """
    Calcula la varianza de un conjunto de datos, que es la medida 
    de la dispersión de los datos respecto a la media.
    
    Parámetros:
    datos (iterable): Un conjunto de datos numéricos, 
    puede ser una lista o un arreglo de numpy.
    
    Retorna:
    float: La varianza de los datos.
    """
    listaNan = []
    for i in datos:
        if math.isfinite(i):
            listaNan.append(i)
    
    listaNan = np.array(listaNan)
    
    porm = promedio(listaNan)  # Se asume que la función promedio ya está definida
    largo = len(listaNan)
    
    varianza = (sum((listaNan - porm)**2))/largo
    
    return varianza


def desvEstandar(datos):
    """
    Calcula la desviación estándar de un conjunto de datos, 
    que es la raíz cuadrada de la varianza.
    
    Parámetros:
    datos (iterable): Un conjunto de datos numéricos, 
    puede ser una lista o un arreglo de numpy.
    
    Retorna:
    float: La desviación estándar de los datos.
    """
    listaNan = []
    for i in datos:
        if math.isfinite(i):
            listaNan.append(i)
            
    listaNan = np.array(listaNan)
    var = varianza(listaNan)
    
    std = (var)**0.5
    
    return std

def MAD(datos):
    """
    Calcula el MAD (desviación absoluta mediana) de un 
    conjunto de datos, que es una medida robusta de la dispersión.
    
    Parámetros:
    datos (iterable): Un conjunto de datos numéricos, 
    puede ser una lista o un arreglo de numpy.
    
    Retorna:
    float: El valor de la desviación absoluta mediana de los datos.
    """
    listaNan = []
    for i in datos:
        if math.isfinite(i):
            listaNan.append(i)
            
    listaNan = np.array(listaNan)
    
    med = mediana(listaNan)  # Se asume que la función mediana ya está definida
    dsvAbs = mediana(np.abs(listaNan - med))
    
    return dsvAbs


def covarianza(x, y):
    """
    Calcula la covarianza entre dos listas x y y.

    Parámetros:
    x La primera variable de datos
    y La segunda variable de datos

    Retorna:
    float: La covarianza entre x e y.
    """
    
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
    
   
    promX = promedio(X)
    promY = promedio(Y)
    
   
    covar = 0
    
    # calcular la suma de (Xi - promedio(X)) * (Yi - promedio(Y))
    for i in range(len(X)):
        covar += (X[i] - promX) * (Y[i] - promY)
    
    
    covar /= len(X)  #  ambos deberían tener el mismo tamaño
    
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
    

def percentil(valores):
    """
    Calcula los percentiles de un conjunto de datos. 
    La función devuelve los percentiles del 1% al 99%
    de un conjunto de datos ordenado.
    
    Parámetros:
    valores (iterable): Un conjunto de datos numéricos, 
    puede ser una lista o un arreglo de numpy.
    
    Retorna:
    list: Una lista de los percentiles del 1% al 99% de los datos, 
    calculados sobre un conjunto de datos ordenado.
    
    Nota:
        para acceder al percentil deseado este tiene que se (N_percentil - 1)
        ya que la lista parte de el cero el percentil 5 está en la posicion 4
        
        y se llama así percentil(datos)[4]
    """
    
    lista = []  
    for k in valores:
        if math.isfinite(k):  # Filtra valores no finitos (NaN, inf)
            lista.append(k)
            
    lista.sort()  # Ordena los datos
    listaPercentiles = []
    n = len(lista)
    
    for i in range(1, 100):  # Itera sobre los percentiles del 1% al 99%
        percentil = n * i // 100  # Calcula el índice para el percentil
        listaPercentiles.append(lista[percentil])  # Añade el percentil a la lista
    
    return listaPercentiles  # Devuelve la lista de percentiles

def IQR(valores):
    """
    Calcula el Rango Intercuartílico (IQR) de un conjunto de datos, que es la diferencia entre el tercer 
    cuartil (percentil 75) y el primer cuartil (percentil 25).
    
    Parámetros:
    valores: Un conjunto de datos numéricos, puede ser una lista o un arreglo de numpy.
    
    Retorna:
    float: El Rango Intercuartílico (IQR) de los datos.
    """
    
    percentiles = percentil(valores)    # Se llama a la función percentiles()
    
    rangoInter = percentiles[74] - percentiles[24]
    
    return rangoInter   # Retorna el rango Intercuartílico

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
    
    

#def gradienteMse(x,y,theta):
#    pendiente, intercepto = theta
#    
#    y_pred = [pendiente *xv + intercepto for xv in x]
#    
#    g1 = 2/len(x) * sum([ (y_p - y_d) for x_d,y_d ])
#    
#    g2 = 2/len(x) *sum([(y_p - y_d) for x_d, y_d, y_p in zip(x,y,y_pred)])
#    
# 
#    return [g1,g2]
    


