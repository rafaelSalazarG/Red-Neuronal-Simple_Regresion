import numpy as np
import matplotlib.pyplot as plt

# Generador basado en ejemplo del curso CS231 de Stanford: 
# CS231n Convolutional Neural Networks for Visual Recognition
# (https://cs231n.github.io/neural-networks-case-study/)
def generar_datos_clasificacion(cantidad_ejemplos=500):
    randomgen = np.random.default_rng()
    A =2+randomgen.standard_normal()*0.1
    B= 3+randomgen.standard_normal()*0.1
    C= 1+randomgen.standard_normal()*0.1

    coords = np.zeros((cantidad_ejemplos, 2))

    u = np.random.rand(cantidad_ejemplos) * 2 * np.pi
    v = np.random.rand(cantidad_ejemplos) * np.pi

    x = A * np.cos(u) * np.cos(v)
    y = B * np.sin(u) * np.cos(v)
    z = C * np.sin(v)
    coords = np.c_[x,y] 

    return coords, z


def inicializar_pesos(n_entrada, n_capa_2, n_capa_3):
    randomgen = np.random.default_rng()

    w1 = 0.1 * randomgen.standard_normal((n_entrada, n_capa_2))
    b1 = 0.1 * randomgen.standard_normal((1, n_capa_2))

    w2 = 0.1 * randomgen.standard_normal((n_capa_2, n_capa_3))
    b2 = 0.1 * randomgen.standard_normal((1,n_capa_3))

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def ejecutar_adelante(x, pesos):
    # Funcion de entrada (a.k.a. "regla de propagacion") para la primera capa oculta
    z = x.dot(pesos["w1"]) + pesos["b1"]

    # Funcion de activacion ReLU para la capa oculta (h -> "hidden")
    h = np.maximum(0, z)

    # Salida de la red (funcion de activacion lineal). Esto incluye la salida de todas
    # las neuronas y para todos los ejemplos proporcionados
    y = h.dot(pesos["w2"]) + pesos["b2"]

    return {"z": z, "h": h, "y": y}


def clasificar(x, pesos):
    # Corremos la red "hacia adelante"
    resultados_feed_forward = ejecutar_adelante(x, pesos)
    
    # Buscamos la(s) clase(s) con scores mas altos (en caso de que haya mas de una con 
    # el mismo score estas podrian ser varias). Dado que se puede ejecutar en batch (x 
    # podria contener varios ejemplos), buscamos los maximos a lo largo del axis=1 
    # (es decir, por filas)
    max_scores = np.argmax(resultados_feed_forward["y"], axis=1)

    # Tomamos el primero de los maximos (podria usarse otro criterio, como ser eleccion aleatoria)
    # Nuevamente, dado que max_scores puede contener varios renglones (uno por cada ejemplo),
    # retornamos la primera columna
    try:
        return max_scores[:, 0]
    except:
        return max_scores[:]

def train(x, t, pesos, learning_rate, epochs, validacion = False, ejemplosVal = 300, periodoVal = 20):
    m = np.size(x, 0) 
    trainingLoss = []
    validationLoss = []
    
    if(validacion):   
        xVal, tVal = generar_datos_clasificacion(cantidad_ejemplos=ejemplosVal, cantidad_clases=3)
        mVal = np.size(xVal, 0)
    for i in range(epochs):
        resultados_feed_forward = ejecutar_adelante(x, pesos)
        y = resultados_feed_forward["y"]
        h = resultados_feed_forward["h"]
        z = resultados_feed_forward["z"]
        
        p = np.zeros((m,1))
        for j in range(m):
            p[j] = (t[j]-y[j])

        loss = ( 1 / m ) *np.sum( np.square(p) ) 
        trainingLoss.append(loss)
        if i %1000 == 0:
            print("Loss epoch", i, ":", loss)
        if i %periodoVal == 0 and validacion: 
            feedForward_Val = ejecutar_adelante(xVal,pesos)
            yVal = feedForward_Val["y"]
            pVal = np.power((tVal-yVal),2)
            lossVal = (1 / m) * np.sum(pVal)
            validationLoss.append(lossVal)

            if (i >epochs*0.3):
                numValidacion = (validationLoss[-1]-validationLoss[-2])
                derivadaVal = numValidacion/0.01
                freno = ((validationLoss[-1]-min(validationLoss))/validationLoss[-1])*100

                if (freno > 5) and derivadaVal>0:
                    return trainingLoss, validationLoss
            
        w1 = pesos["w1"]
        b1 = pesos["b1"]
        w2 = pesos["w2"]
        b2 = pesos["b2"]
        dL_dy =(-2/m) *p       

        dL_dw2 = h.T.dot(dL_dy)    #             
        dL_db2 = np.sum(dL_dy, axis=0, keepdims=True) #

        dL_dh = dL_dy.dot(w2.T) #
        
        dL_dz = dL_dh      #
        dL_dz[z <= 0] = 0  #

        dL_dw1 = x.T.dot(dL_dz)     #                    
        dL_db1 = np.sum(dL_dz, axis=0, keepdims=True)  #

        w1 += -learning_rate * dL_dw1
        b1 += -learning_rate * dL_db1
        w2 += -learning_rate * dL_dw2
        b2 += -learning_rate * dL_db2

        pesos["w1"] = w1
        pesos["b1"] = b1
        pesos["w2"] = w2
        pesos["b2"] = b2
    if(validacion):
        return trainingLoss, validationLoss
    else:
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.add_subplot(111,projection = '3d')
        ax.scatter(x[:,0], x[:,1],y,c='r',marker='o')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title('Salida de la red neuronal')
        plt.show()
        return trainingLoss     

def iniciar(numero_ejemplos, graficar_datos, validacion = False, periodoVal=20 ):
    # Generamos datos
    x, t = generar_datos_clasificacion(numero_ejemplos)

    if graficar_datos:
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.add_subplot(111,projection = '3d')
        ax.scatter(x[:,0], x[:,1],t,c='r',marker='o')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_title('Elipse inicial, de entrenamiento')
        plt.show()
    # Inicializa pesos de la red
    NEURONAS_CAPA_OCULTA = 100
    NEURONAS_ENTRADA = 2

    pesos = inicializar_pesos(n_entrada=NEURONAS_ENTRADA, n_capa_2=NEURONAS_CAPA_OCULTA, n_capa_3=1)

    # Entrena
    LEARNING_RATE=0.5
    EPOCHS=20000
    if validacion:
        trainingLoss, validationLoss = train(x, t, pesos, LEARNING_RATE, EPOCHS, True, periodoVal=periodoVal) 
    else: 
        trainingLoss = train(x, t, pesos, LEARNING_RATE, EPOCHS)
    if validacion:
        fig,ax = plt.subplots()
        trainX = [i for i in range(len(trainingLoss))]
        espaciamiento = (len(trainingLoss)/periodoVal)
        if(isinstance(espaciamiento,int) == False):
            espaciamiento =int(espaciamiento)+1
        valX = np.linspace(0,trainX[-1],espaciamiento)
        ax.plot(trainX,trainingLoss,'b',valX,validationLoss,'r')
        plt.show()
    else:
        fig,ax = plt.subplots()
        trainX = [i for i in range(len(trainingLoss))]
        ax.plot(trainX,trainingLoss,'b')
        plt.show()
    return pesos
print('Entrenando red neuronal...')
pesos = iniciar(numero_ejemplos=500, graficar_datos=True, validacion=False, periodoVal=30)
'''print('Clasificando elementos de test...')
x,t = generar_datos_clasificacion(cantidad_ejemplos=300,cantidad_clases=3)
predicted_class = clasificar(x,pesos)
b = predicted_class == t
print ('Precision de testeo: %.2f'  % (np.mean(b)))'''
pass