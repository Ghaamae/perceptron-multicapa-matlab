%Universidad Guadalajara Lamar
%Ingenieria en Computacion
%Redes Neuronales Avanzadas
%Maria Ines Calderon Zetter
%Programa: Perceptron Multicapa con funciones de aprendizaje 
%de Levenberg Marquardt y aprendizaje de Gradiente Descenciente.

%t1 =[-1 : 0.05 : 1]; %%Rangos de Valores

t =[-1 : 0.02 : 1];
t1 =[0 : 0.1 : 2*pi]; %%Rangos de Valores

%Funciones de evaluacion sigmoidal con y sin ruido
f = sin(2*pi*t); %%Funcion
%f = sin(2*pi*t)+0.05*randn(size(t));

%Creamos una red neuronal feedforward de dos capas.
%Los rangos de entrada son del minimo al maximo de t. La primera capa tiene 
%siete neuronas tansig, la segunda capa tiene una neurona purelin,
%hay que utilizar trainlm para entrenarle. (Algoritmo que actualiza los 
%pesos y las ganancias de acuerdo a la optimización de Levenberg-Marquardt. )

net.trainFcn='trainlm';
%Funcion de aprendizaje de Levenberg Marquardt
net = netwff(minmax(t),[7,1],{'tansig','purelin'},'trainlm');

%Funcion de aprendizaje de Gradiente Descendiente
%net = newff(minmax(t),[10,1],{'tansig','purelin'},'traingd');

%net.trainParam.show = 5;

%Maximo de epocas
net.trainParam.epochs = 1000;

%Error Deseado
net.trainParam.goal = 1e-3;

net.trainFcn = 'trainlm';
%net.init(net);
[net,tr] = train(net,t,f);
y = sim(net,t);
y1 = sim(net,t1);

%figure
%plot(t1,y,'r--')
%hold on
%plot(t,f)

figure
plot(t1,y1,'r--')
hold on
plot(t,f)
